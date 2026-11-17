//! The WebGPU engine: tensors held on the GPU and WGSL kernels that work on them. Natively wgpu runs
//! on Vulkan, Metal or DX12, which is how the kernels are tested against candle; in a browser the
//! same code runs on WebGPU.

use wgpu::util::DeviceExt;

pub mod bridge;
pub mod llama;
pub mod model;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// A tensor on the GPU: f32 values, row-major, of `shape`.
pub struct GpuTensor {
    pub shape: Vec<usize>,
    buffer: wgpu::Buffer,
}

impl GpuTensor {
    /// The same values under another shape of the same size, sharing the buffer.
    pub fn view(&self, shape: &[usize]) -> GpuTensor {
        assert_eq!(shape.iter().product::<usize>(), self.len(), "view {:?} of {:?}", shape, self.shape);
        GpuTensor { shape: shape.to_vec(), buffer: self.buffer.clone() }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A compiled kernel and its bind group layout, asked for once: a browser makes a new layout object on
/// every `get_bind_group_layout`.
struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// Uniform slots in the ring every launch writes its parameters into: 256 bytes each, the alignment a
/// uniform binding's offset needs.
const SLOTS: usize = 4096;
const SLOT: u64 = 256;

/// One GPU device and the kernels compiled for it. Work is RECORDED into one command encoder and
/// submitted once, when a value is read back: a forward pass is a single submission, not hundreds.
/// Every buffer recorded is held until that read completes — in a browser, dropping a buffer
/// destroys it, and a buffer destroyed under pending work fails the read.
pub struct Gpu {
    // Held for the device's lifetime: in a browser the device is lost ("a valid external Instance
    // reference no longer exists") once the instance that made it is collected.
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    encoder: std::cell::RefCell<Option<wgpu::CommandEncoder>>,
    held: std::cell::RefCell<Vec<wgpu::Buffer>>,
    errors: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    uniforms: wgpu::Buffer,
    slot: std::cell::Cell<usize>,
    matmul: Kernel,
    rms_norm: Kernel,
    embed: Kernel,
    rope: Kernel,
    attention: Kernel,
    silu_mul: Kernel,
    add: Kernel,
    place: Kernel,
}

/// y = x Wᵀ: x is m × k (from row `xoff` of its buffer), W is n × k (a linear layer's weight), y is m × n. Each 16 × 16 workgroup
/// computes one tile of y, stepping through k a tile at a time through workgroup memory.
const MATMUL: &str = r#"
struct Dims { m: u32, n: u32, k: u32, xoff: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;
const T: u32 = 16u;
var<workgroup> xs: array<array<f32, 16>, 16>;
var<workgroup> ws: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let row = wg.y * T + l.y;
    let col = wg.x * T + l.x;
    let wrow = wg.x * T + l.y;
    var acc = 0.0;
    let tiles = (d.k + T - 1u) / T;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let kk = t * T + l.x;
        var xv = 0.0;
        if (row < d.m && kk < d.k) { xv = x[(row + d.xoff) * d.k + kk]; }
        xs[l.y][l.x] = xv;
        var wv = 0.0;
        if (wrow < d.n && kk < d.k) { wv = w[wrow * d.k + kk]; }
        ws[l.y][l.x] = wv;
        workgroupBarrier();
        for (var i = 0u; i < T; i = i + 1u) { acc = acc + xs[l.y][i] * ws[l.x][i]; }
        workgroupBarrier();
    }
    if (row < d.m && col < d.n) { y[row * d.n + col] = acc; }
}
"#;

/// RMSNorm over each row of x (m × d), scaled by g (d): one workgroup per row, reducing the sum of
/// squares in workgroup memory.
const RMS_NORM: &str = r#"
struct P { d: u32, eps: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let base = wg.x * p.d;
    var s = 0.0;
    for (var i = l.x; i < p.d; i = i + 256u) { let v = x[base + i]; s = s + v * v; }
    part[l.x] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (l.x < stride) { part[l.x] = part[l.x] + part[l.x + stride]; }
        workgroupBarrier();
    }
    let r = inverseSqrt(part[0] / f32(p.d) + p.eps);
    for (var i = l.x; i < p.d; i = i + 256u) { y[base + i] = x[base + i] * r * g[i]; }
}
"#;

/// Rows of an embedding table, one per token id: y (n × h) = table[ids].
const EMBED: &str = r#"
struct P { h: u32, n: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= p.h * p.n) { return; }
    y[i] = table[ids[i / p.h] * p.h + i % p.h];
}
"#;

/// Rotary position embedding in place on x (n × heads × hd), rotate-half form, from cos/sin tables
/// (positions × hd/2) built on the host exactly as candle builds them.
const ROPE: &str = r#"
struct P { heads: u32, hd: u32, n: u32, pos: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let half = p.hd / 2u;
    let idx = g.x;
    if (idx >= p.n * p.heads * half) { return; }
    let j = idx % half;
    let rest = idx / half;
    let h = rest % p.heads;
    let i = rest / p.heads;
    let base = (i * p.heads + h) * p.hd;
    let c = cos_t[(p.pos + i) * half + j];
    let s = sin_t[(p.pos + i) * half + j];
    let x1 = x[base + j];
    let x2 = x[base + j + half];
    x[base + j] = x1 * c - x2 * s;
    x[base + j + half] = x2 * c + x1 * s;
}
"#;

/// Causal attention for query rows at positions pos..pos+n against the KV cache (max × kvh × hd), one
/// workgroup per (row, head), an online softmax over keys in chunks of 64 so no score matrix exists.
const ATTENTION: &str = r#"
struct P { heads: u32, kvh: u32, hd: u32, n: u32, pos: u32, scale: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> qs: array<f32, 256>;
var<workgroup> sc: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let i = wg.x;
    let h = wg.y;
    let t = l.x;
    let kvhead = h / (p.heads / p.kvh);
    let qbase = (i * p.heads + h) * p.hd;
    for (var d = t; d < p.hd; d = d + 64u) { qs[d] = q[qbase + d]; }
    workgroupBarrier();
    let keys = p.pos + i + 1u;
    var m = -1e30;
    var lsum = 0.0;
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var j0 = 0u; j0 < keys; j0 = j0 + 64u) {
        let j = j0 + t;
        var s = -1e30;
        if (j < keys) {
            let kb = (j * p.kvh + kvhead) * p.hd;
            var dot = 0.0;
            for (var d = 0u; d < p.hd; d = d + 1u) { dot = dot + qs[d] * kc[kb + d]; }
            s = dot * p.scale;
        }
        sc[t] = s;
        red[t] = s;
        workgroupBarrier();
        for (var st = 32u; st > 0u; st = st >> 1u) {
            if (t < st) { red[t] = max(red[t], red[t + st]); }
            workgroupBarrier();
        }
        let mnew = max(m, red[0]);
        let corr = exp(m - mnew);
        let cnt = min(64u, keys - j0);
        var psum = 0.0;
        for (var a = 0u; a < 4u; a = a + 1u) { acc[a] = acc[a] * corr; }
        for (var u = 0u; u < cnt; u = u + 1u) {
            let pw = exp(sc[u] - mnew);
            psum = psum + pw;
            let vb = ((j0 + u) * p.kvh + kvhead) * p.hd;
            for (var a = 0u; a < 4u; a = a + 1u) {
                let d = t + a * 64u;
                if (d < p.hd) { acc[a] = acc[a] + pw * vc[vb + d]; }
            }
        }
        lsum = lsum * corr + psum;
        m = mnew;
        workgroupBarrier();
    }
    for (var a = 0u; a < 4u; a = a + 1u) {
        let d = t + a * 64u;
        if (d < p.hd) { y[qbase + d] = acc[a] / lsum; }
    }
}
"#;

/// The MLP's gate: y = silu(g) · u, elementwise.
const SILU_MUL: &str = r#"
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let v = g[i];
    y[i] = v / (1.0 + exp(-v)) * u[i];
}
"#;

/// Columns of a wider matrix: dst[r, col + c] = src[r, c], for src n × w and dst n × width.
const PLACE: &str = r#"
struct P { n: u32, w: u32, width: u32, col: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n * p.w) { return; }
    let r = i / p.w;
    dst[r * p.width + p.col + i % p.w] = src[i];
}
"#;

/// The residual stream: x += y, elementwise, in place.
const ADD: &str = r#"
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    x[i] = x[i] + y[i];
}
"#;

impl Gpu {
    /// The most capable adapter there is, or an error naming why there is none.
    pub async fn new() -> Result<Gpu, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("prattle"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("the GPU refused a device: {e}"))?;
        let pipeline = |name: &str, source: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let layout = pipeline.get_bind_group_layout(0);
            Kernel { pipeline, layout }
        };
        let matmul = pipeline("matmul", MATMUL);
        let rms_norm = pipeline("rms_norm", RMS_NORM);
        let embed = pipeline("embed", EMBED);
        let rope = pipeline("rope", ROPE);
        let attention = pipeline("attention", ATTENTION);
        let silu_mul = pipeline("silu_mul", SILU_MUL);
        let add = pipeline("add", ADD);
        let place = pipeline("place", PLACE);
        let errors = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = errors.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            if let Ok(mut v) = seen.lock() {
                v.push(e.to_string());
            }
        }));
        let lost = errors.clone();
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut v) = lost.lock() {
                v.push(format!("device lost ({reason:?}): {message}"));
            }
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: SLOTS as u64 * SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Gpu {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            encoder: std::cell::RefCell::new(None),
            held: std::cell::RefCell::new(Vec::new()),
            errors,
            uniforms,
            slot: std::cell::Cell::new(0),
            matmul,
            rms_norm,
            embed,
            rope,
            attention,
            silu_mul,
            add,
            place,
        })
    }

    /// Token ids, as a u32 buffer the embedding kernel reads.
    pub fn upload_ids(&self, ids: &[u32]) -> GpuTensor {
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(ids),
            usage: wgpu::BufferUsages::STORAGE,
        });
        GpuTensor { shape: vec![ids.len()], buffer }
    }

    /// A zeroed tensor of `shape`.
    pub fn zeros(&self, shape: &[usize]) -> GpuTensor {
        self.empty(shape)
    }

    /// Copy all of `src` into `dst` starting at element `at`.
    pub fn copy_into(&self, src: &GpuTensor, dst: &GpuTensor, at: usize) {
        assert!(at + src.len() <= dst.len(), "copy of {} into {} at {at}", src.len(), dst.len());
        self.record(|e| e.copy_buffer_to_buffer(&src.buffer, 0, &dst.buffer, (at * 4) as u64, (src.len() * 4) as u64));
        self.held.borrow_mut().extend([src.buffer.clone(), dst.buffer.clone()]);
    }

    /// Record into the open command encoder, opening one if none is.
    fn record(&self, f: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let mut slot = self.encoder.borrow_mut();
        f(slot.get_or_insert_with(|| self.device.create_command_encoder(&Default::default())));
    }

    /// Submit everything recorded so far.
    pub fn flush(&self) {
        if let Some(e) = self.encoder.borrow_mut().take() {
            self.queue.submit([e.finish()]);
        }
    }

    pub fn upload(&self, values: &[f32], shape: &[usize]) -> GpuTensor {
        assert_eq!(values.len(), shape.iter().product::<usize>(), "{} values for shape {shape:?}", values.len());
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        GpuTensor { shape: shape.to_vec(), buffer }
    }

    fn empty(&self, shape: &[usize]) -> GpuTensor {
        let n: usize = shape.iter().product();
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GpuTensor { shape: shape.to_vec(), buffer }
    }

    /// The values of `t`, read back from the GPU.
    pub async fn download(&self, t: &GpuTensor) -> Vec<f32> {
        let size = (t.len() * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.record(|e| e.copy_buffer_to_buffer(&t.buffer, 0, &staging, 0, size));
        self.flush();
        let slice = staging.slice(..size);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Natively the read completes only when the device is polled; in a browser the page's event
        // loop completes it, and the future below resolves then.
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let read = rx.await.expect("the GPU dropped a read");
        if let Err(e) = read {
            let errors = self.errors.lock().map(|v| v.join("; ")).unwrap_or_default();
            panic!("the GPU refused a read: {e:?}; the device reported: {}", if errors.is_empty() { "nothing" } else { &errors });
        }
        let values = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        // The read completing means everything submitted before it has run: the held buffers may go,
        // and the uniform ring starts over.
        self.held.borrow_mut().clear();
        self.slot.set(0);
        values
    }

    fn run(&self, kernel: &Kernel, buffers: &[&wgpu::Buffer], uniform: &[u32], groups: (u32, u32)) {
        // A full ring is submitted before a slot is reused: queue writes run in order with
        // submissions, so a slot written after the submit cannot reach work recorded before it.
        if self.slot.get() == SLOTS {
            self.flush();
            self.slot.set(0);
        }
        let offset = self.slot.get() as u64 * SLOT;
        self.slot.set(self.slot.get() + 1);
        self.queue.write_buffer(&self.uniforms, offset, bytemuck::cast_slice(uniform));
        let size = std::num::NonZeroU64::new((uniform.len() * 4) as u64);
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: buffers.len() as u32,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.uniforms, offset, size }),
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &kernel.layout, entries: &entries });
        self.record(|e| {
            let mut pass = e.begin_compute_pass(&Default::default());
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        });
        self.held.borrow_mut().extend(buffers.iter().map(|b| (*b).clone()));
    }

    /// y = x Wᵀ for x (m × k) and a linear layer's weight W (n × k).
    pub fn matmul(&self, x: &GpuTensor, w: &GpuTensor) -> GpuTensor {
        self.matmul_rows(x, 0, x.shape[0], w)
    }

    /// y = x[from..from+m] Wᵀ: `m` rows of x starting at row `from`.
    pub fn matmul_rows(&self, x: &GpuTensor, from: usize, m: usize, w: &GpuTensor) -> GpuTensor {
        let k = x.shape[1];
        let n = w.shape[0];
        assert_eq!(w.shape[1], k, "matmul: x is {:?}, W is {:?}", x.shape, w.shape);
        assert!(from + m <= x.shape[0], "matmul: rows {from}..{} of {:?}", from + m, x.shape);
        let y = self.empty(&[m, n]);
        let groups = (n.div_ceil(16) as u32, m.div_ceil(16) as u32);
        self.run(&self.matmul, &[&x.buffer, &w.buffer, &y.buffer], &[m as u32, n as u32, k as u32, from as u32], groups);
        y
    }

    /// The rows of `table` (vocab × h) for `ids`.
    pub fn embed(&self, ids: &GpuTensor, table: &GpuTensor) -> GpuTensor {
        let (n, h) = (ids.len(), table.shape[1]);
        let y = self.empty(&[n, h]);
        self.run(&self.embed, &[&ids.buffer, &table.buffer, &y.buffer], &[h as u32, n as u32, 0, 0], ((n * h).div_ceil(256) as u32, 1));
        y
    }

    /// Rotary embedding in place on x (n × heads·hd) for positions pos..pos+n.
    pub fn rope(&self, x: &GpuTensor, heads: usize, cos: &GpuTensor, sin: &GpuTensor, pos: usize) {
        let n = x.shape[0];
        let hd = x.shape[1] / heads;
        let work = n * heads * hd / 2;
        self.run(&self.rope, &[&x.buffer, &cos.buffer, &sin.buffer], &[heads as u32, hd as u32, n as u32, pos as u32], (work.div_ceil(256) as u32, 1));
    }

    /// Causal attention of q (n × heads·hd, at positions pos..pos+n) over the caches' first pos+n rows.
    pub fn attention(&self, q: &GpuTensor, kc: &GpuTensor, vc: &GpuTensor, heads: usize, kvh: usize, pos: usize) -> GpuTensor {
        let n = q.shape[0];
        let hd = q.shape[1] / heads;
        assert!(hd <= 256, "attention: head width {hd} over 256");
        let y = self.empty(&q.shape);
        let scale = (1.0 / (hd as f32).sqrt()).to_bits();
        let p = [heads as u32, kvh as u32, hd as u32, n as u32, pos as u32, scale, 0, 0];
        self.run(&self.attention, &[&q.buffer, &kc.buffer, &vc.buffer, &y.buffer], &p, (n as u32, heads as u32));
        y
    }

    /// silu(g) · u, elementwise.
    pub fn silu_mul(&self, g: &GpuTensor, u: &GpuTensor) -> GpuTensor {
        let y = self.empty(&g.shape);
        self.run(&self.silu_mul, &[&g.buffer, &u.buffer, &y.buffer], &[g.len() as u32, 0, 0, 0], (g.len().div_ceil(256) as u32, 1));
        y
    }

    /// A copy of `t` in a buffer of its own.
    pub fn copy(&self, t: &GpuTensor) -> GpuTensor {
        let y = self.empty(&t.shape);
        self.copy_into(t, &y, 0);
        y
    }

    /// dst[:, col..col + w] = src, for src n × w and dst n × width.
    pub fn place(&self, src: &GpuTensor, dst: &GpuTensor, col: usize) {
        let (n, w, width) = (src.shape[0], src.shape[1], dst.shape[1]);
        assert!(dst.shape[0] == n && col + w <= width, "place {:?} at column {col} of {:?}", src.shape, dst.shape);
        self.run(&self.place, &[&src.buffer, &dst.buffer], &[n as u32, w as u32, width as u32, col as u32], ((n * w).div_ceil(256) as u32, 1));
    }

    /// x += y, in place.
    pub fn add(&self, x: &GpuTensor, y: &GpuTensor) {
        assert_eq!(x.len(), y.len(), "add: {:?} and {:?}", x.shape, y.shape);
        self.run(&self.add, &[&x.buffer, &y.buffer], &[x.len() as u32, 0, 0, 0], (x.len().div_ceil(256) as u32, 1));
    }

    /// RMSNorm over each row of x (m × d), scaled by g (d).
    pub fn rms_norm(&self, x: &GpuTensor, g: &GpuTensor, eps: f32) -> GpuTensor {
        let d = *x.shape.last().unwrap();
        let m = x.len() / d;
        assert_eq!(g.len(), d, "rms_norm: x is {:?}, g is {:?}", x.shape, g.shape);
        let y = self.empty(&x.shape);
        self.run(&self.rms_norm, &[&x.buffer, &g.buffer, &y.buffer], &[d as u32, eps.to_bits(), 0, 0], (m as u32, 1));
        y
    }
}
