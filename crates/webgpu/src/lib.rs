//! The WebGPU engine: tensors held on the GPU and WGSL kernels that work on them. Natively wgpu runs
//! on Vulkan, Metal or DX12, which is how the kernels are tested against candle; in a browser the
//! same code runs on WebGPU.

use wgpu::util::DeviceExt;

pub mod bridge;
pub mod decoder;
pub mod lfm2;
pub mod llama;
pub mod model;
pub mod qwen35;
pub mod stream;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// A tensor on the GPU, row-major, of `shape`: f32 values, or — `half` — f16 values packed two to a
/// u32, which only a weight read by `matmul` or `embed` may be.
#[derive(Clone)]
pub struct GpuTensor {
    pub shape: Vec<usize>,
    pub half: bool,
    buffer: wgpu::Buffer,
}

impl GpuTensor {
    /// The same values under another shape of the same size, sharing the buffer.
    pub fn view(&self, shape: &[usize]) -> GpuTensor {
        assert_eq!(shape.iter().product::<usize>(), self.len(), "view {:?} of {:?}", shape, self.shape);
        GpuTensor { shape: shape.to_vec(), half: self.half, buffer: self.buffer.clone() }
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
    matmul_h: Kernel,
    embed_h: Kernel,
    split_halves: Kernel,
    mul_sigmoid: Kernel,
    l2norm_heads: Kernel,
    gdn_gates: Kernel,
    conv_silu: Kernel,
    mul_cols: Kernel,
    delta: Kernel,
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

/// MATMUL with W in f16, two values to a u32: the same tiles, each weight unpacked as it is staged.
const MATMUL_H: &str = r#"
struct Dims { m: u32, n: u32, k: u32, xoff: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
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
        if (wrow < d.n && kk < d.k) {
            let e = wrow * d.k + kk;
            wv = unpack2x16float(w[e >> 1u])[e & 1u];
        }
        ws[l.y][l.x] = wv;
        workgroupBarrier();
        for (var i = 0u; i < T; i = i + 1u) { acc = acc + xs[l.y][i] * ws[l.x][i]; }
        workgroupBarrier();
    }
    if (row < d.m && col < d.n) { y[row * d.n + col] = acc; }
}
"#;

/// EMBED from an f16 table.
const EMBED_H: &str = r#"
struct P { h: u32, n: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= p.h * p.n) { return; }
    let e = ids[i / p.h] * p.h + i % p.h;
    y[i] = unpack2x16float(table[e >> 1u])[e & 1u];
}
"#;

/// Each head's two halves apart: x (n × heads × 2·hd) → a, b (n × heads × hd) — Qwen3.5's query and
/// its gate, which one projection writes side by side per head.
const SPLIT_HALVES: &str = r#"
struct P { n: u32, heads: u32, hd: u32, pad0: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= p.n * p.heads * p.hd) { return; }
    let src = (i / p.hd) * 2u * p.hd + i % p.hd;
    a[i] = x[src];
    b[i] = x[src + p.hd];
}
"#;

/// x ← x · σ(g), elementwise, in place: gated attention's output gate.
const MUL_SIGMOID: &str = r#"
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    x[i] = x[i] / (1.0 + exp(-g[i]));
}
"#;

/// L2 normalisation of the first heads of each row, in place: row r's head h is the `d` values from
/// r·width + h·d, divided by √(Σx² + 1e-6), and the first `scaled` heads multiplied by `scale` — Gated
/// DeltaNet's queries (scaled by 1/√d) and keys. One workgroup per (row, head).
const L2NORM_HEADS: &str = r#"
struct P { width: u32, d: u32, scaled: u32, scale: f32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
var<workgroup> part: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let base = wg.x * p.width + wg.y * p.d;
    var s = 0.0;
    for (var i = l.x; i < p.d; i = i + 64u) { let v = x[base + i]; s = s + v * v; }
    part[l.x] = s;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {
        if (l.x < st) { part[l.x] = part[l.x] + part[l.x + st]; }
        workgroupBarrier();
    }
    var r = 1.0 / sqrt(part[0] + 1e-6);
    if (wg.y < p.scaled) { r = r * p.scale; }
    for (var i = l.x; i < p.d; i = i + 64u) { x[base + i] = x[base + i] * r; }
}
"#;

/// Gated DeltaNet's per-step gates, in place over (n × heads): a ← exp(−exp(A_log)·softplus(a + dt_bias)),
/// the decay; b ← σ(b), beta. `neg_a` holds −exp(A_log). softplus as max(x, 0) + log(1 + e^−|x|), so a
/// large input gives a finite decay.
const GDN_GATES: &str = r#"
struct P { n: u32, heads: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read> neg_a: array<f32>;
@group(0) @binding(3) var<storage, read> dt: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let h = i % p.heads;
    let v = a[i] + dt[h];
    let sp = max(v, 0.0) + log(1.0 + exp(-abs(v)));
    a[i] = exp(neg_a[h] * sp);
    b[i] = 1.0 / (1.0 + exp(-b[i]));
}
"#;

/// A causal depthwise conv: raw (n × ch) after the state's k − 1 rows (oldest first),
/// y[t, c] = Σ_j w[c, j] · padded[t + j, c], then SiLU when `silu` is 1 (Gated DeltaNet) and nothing
/// when 0 (LFM2's short conv); `rows` gets the last k − 1 rows of the padded input — the state after
/// these tokens.
const CONV_SILU: &str = r#"
struct P { n: u32, ch: u32, k: u32, silu: u32 };
@group(0) @binding(0) var<storage, read> raw: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> st: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> rows: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
fn padded(r: u32, c: u32) -> f32 {
    if (r < p.k - 1u) { return st[r * p.ch + c]; }
    return raw[(r - (p.k - 1u)) * p.ch + c];
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = i / p.ch;
    let c = i % p.ch;
    if (t < p.n) {
        var acc = 0.0;
        for (var j = 0u; j < p.k; j = j + 1u) { acc = acc + w[c * p.k + j] * padded(t + j, c); }
        if (p.silu == 1u) { acc = acc / (1.0 + exp(-acc)); }
        y[i] = acc;
    }
    if (t < p.k - 1u) { rows[t * p.ch + c] = padded(p.n + t, c); }
}
"#;

/// Products of column ranges: y[r, c] = a[r·wa + oa + c] · b[r·wb + ob + c] for y n × w — LFM2's B·x and
/// C·conv, each a slice of one projection's row.
const MUL_COLS: &str = r#"
struct P { n: u32, w: u32, wa: u32, oa: u32, wb: u32, ob: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n * p.w) { return; }
    let r = i / p.w;
    let c = i % p.w;
    y[i] = a[r * p.wa + p.oa + c] * b[r * p.wb + p.ob + c];
}
"#;

/// The gated delta rule, token by token, its state S (heads × dk × dv) resident and updated in place.
/// One workgroup per value head, one invocation per value column j, holding S[:, j]: per token,
/// S ← decay·S; delta = (v − Sᵀk)·beta; S ← S + k·deltaᵀ; o = Sᵀq. q, k (already normalised) and v are
/// read from the conv's output row: [q heads | k heads | v heads]; value head h reads key head
/// h / (heads / key heads).
const DELTA: &str = r#"
struct P { n: u32, kk: u32, vh: u32, dk: u32, dv: u32, ch: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> mixed: array<f32>;
@group(0) @binding(1) var<storage, read> decay: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> st: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let h = wg.x;
    let j = l.x;
    let kh = h / (p.vh / p.kk);
    var s: array<f32, 128>;
    if (j < p.dv) {
        for (var i = 0u; i < p.dk; i = i + 1u) { s[i] = st[(h * p.dk + i) * p.dv + j]; }
    }
    for (var t = 0u; t < p.n; t = t + 1u) {
        let row = t * p.ch;
        if (j < p.dk) {
            qs[j] = mixed[row + kh * p.dk + j];
            ks[j] = mixed[row + p.kk * p.dk + kh * p.dk + j];
        }
        workgroupBarrier();
        if (j < p.dv) {
            let g = decay[t * p.vh + h];
            var kv = 0.0;
            for (var i = 0u; i < p.dk; i = i + 1u) { s[i] = s[i] * g; kv = kv + s[i] * ks[i]; }
            let dlt = (mixed[row + 2u * p.kk * p.dk + h * p.dv + j] - kv) * beta[t * p.vh + h];
            var acc = 0.0;
            for (var i = 0u; i < p.dk; i = i + 1u) { s[i] = s[i] + ks[i] * dlt; acc = acc + s[i] * qs[i]; }
            o[t * p.vh * p.dv + h * p.dv + j] = acc;
        }
        workgroupBarrier();
    }
    if (j < p.dv) {
        for (var i = 0u; i < p.dk; i = i + 1u) { st[(h * p.dk + i) * p.dv + j] = s[i]; }
    }
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

/// Rotary position embedding in place on x (n × heads × hd), rotate-half form over each head's first
/// `rot` dims (the rest untouched), from cos/sin tables (positions × rot/2) built on the host exactly
/// as candle builds them.
const ROPE: &str = r#"
struct P { heads: u32, hd: u32, n: u32, pos: u32, rot: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let half = p.rot / 2u;
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
        let matmul_h = pipeline("matmul_h", MATMUL_H);
        let embed_h = pipeline("embed_h", EMBED_H);
        let split_halves = pipeline("split_halves", SPLIT_HALVES);
        let mul_sigmoid = pipeline("mul_sigmoid", MUL_SIGMOID);
        let l2norm_heads = pipeline("l2norm_heads", L2NORM_HEADS);
        let gdn_gates = pipeline("gdn_gates", GDN_GATES);
        let conv_silu = pipeline("conv_silu", CONV_SILU);
        let mul_cols = pipeline("mul_cols", MUL_COLS);
        let delta = pipeline("delta", DELTA);
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
            matmul_h,
            embed_h,
            split_halves,
            mul_sigmoid,
            l2norm_heads,
            gdn_gates,
            conv_silu,
            mul_cols,
            delta,
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
        GpuTensor { shape: vec![ids.len()], half: false, buffer }
    }

    /// A zeroed tensor of `shape`.
    pub fn zeros(&self, shape: &[usize]) -> GpuTensor {
        self.empty(shape)
    }

    /// Copy all of `src` into `dst` starting at element `at`.
    pub fn copy_into(&self, src: &GpuTensor, dst: &GpuTensor, at: usize) {
        assert!(!src.half && !dst.half, "copy_into works on f32 tensors");
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
        GpuTensor { shape: shape.to_vec(), half: false, buffer }
    }

    /// An f16 tensor of `shape`, its values still to be written (`write`): `bytes` is its packed size.
    pub fn empty_half(&self, shape: &[usize]) -> GpuTensor {
        let n: usize = shape.iter().product();
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 2).next_multiple_of(4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GpuTensor { shape: shape.to_vec(), half: true, buffer }
    }

    /// f32 values stored as f16, two to a u32: a weight `matmul` and `embed` read at half the size.
    pub fn upload_half(&self, values: &[f32], shape: &[usize]) -> GpuTensor {
        assert_eq!(values.len(), shape.iter().product::<usize>(), "{} values for shape {shape:?}", values.len());
        let t = self.empty_half(shape);
        let mut bytes: Vec<u8> = values.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        self.write(&t, 0, &bytes);
        t
    }

    /// Raw little-endian bytes into `t` at byte `offset` (both multiples of 4): a streamed weight.
    pub fn write(&self, t: &GpuTensor, offset: u64, bytes: &[u8]) {
        self.queue.write_buffer(&t.buffer, offset, bytes);
    }

    /// Every value of `t` set to zero.
    pub fn clear(&self, t: &GpuTensor) {
        self.record(|e| e.clear_buffer(&t.buffer, 0, None));
        self.held.borrow_mut().push(t.buffer.clone());
    }

    fn empty(&self, shape: &[usize]) -> GpuTensor {
        let n: usize = shape.iter().product();
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GpuTensor { shape: shape.to_vec(), half: false, buffer }
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
        assert!(!w.half || k.is_multiple_of(2), "matmul: an f16 weight's rows must hold an even count, not {k}");
        let y = self.empty(&[m, n]);
        let groups = (n.div_ceil(16) as u32, m.div_ceil(16) as u32);
        let kernel = if w.half { &self.matmul_h } else { &self.matmul };
        self.run(kernel, &[&x.buffer, &w.buffer, &y.buffer], &[m as u32, n as u32, k as u32, from as u32], groups);
        y
    }

    /// The rows of `table` (vocab × h) for `ids`.
    pub fn embed(&self, ids: &GpuTensor, table: &GpuTensor) -> GpuTensor {
        let (n, h) = (ids.len(), table.shape[1]);
        let y = self.empty(&[n, h]);
        let kernel = if table.half { &self.embed_h } else { &self.embed };
        self.run(kernel, &[&ids.buffer, &table.buffer, &y.buffer], &[h as u32, n as u32, 0, 0], ((n * h).div_ceil(256) as u32, 1));
        y
    }

    /// Rotary embedding in place on x (n × heads·hd) for positions pos..pos+n.
    pub fn rope(&self, x: &GpuTensor, heads: usize, cos: &GpuTensor, sin: &GpuTensor, pos: usize) {
        self.rope_part(x, heads, x.shape[1] / heads, cos, sin, pos);
    }

    /// Rotary embedding in place on the first `rot` dims of each head of x (n × heads·hd), for
    /// positions pos..pos+n, from tables of positions × rot/2.
    pub fn rope_part(&self, x: &GpuTensor, heads: usize, rot: usize, cos: &GpuTensor, sin: &GpuTensor, pos: usize) {
        let n = x.shape[0];
        let hd = x.shape[1] / heads;
        assert!(rot <= hd && rot.is_multiple_of(2), "rope: {rot} rotated dims of a {hd}-wide head");
        let work = n * heads * rot / 2;
        let p = [heads as u32, hd as u32, n as u32, pos as u32, rot as u32, 0, 0, 0];
        self.run(&self.rope, &[&x.buffer, &cos.buffer, &sin.buffer], &p, (work.div_ceil(256) as u32, 1));
    }

    /// Each head's halves of x (n × heads·2hd) apart, as two tensors of n × heads·hd.
    pub fn split_halves(&self, x: &GpuTensor, heads: usize) -> (GpuTensor, GpuTensor) {
        let n = x.shape[0];
        let hd = x.shape[1] / heads / 2;
        let (a, b) = (self.empty(&[n, heads * hd]), self.empty(&[n, heads * hd]));
        let work = n * heads * hd;
        self.run(&self.split_halves, &[&x.buffer, &a.buffer, &b.buffer], &[n as u32, heads as u32, hd as u32, 0], (work.div_ceil(256) as u32, 1));
        (a, b)
    }

    /// x ← x · σ(g), in place.
    pub fn mul_sigmoid(&self, x: &GpuTensor, g: &GpuTensor) {
        assert_eq!(x.len(), g.len(), "mul_sigmoid: {:?} and {:?}", x.shape, g.shape);
        self.run(&self.mul_sigmoid, &[&x.buffer, &g.buffer], &[x.len() as u32, 0, 0, 0], (x.len().div_ceil(256) as u32, 1));
    }

    /// L2-normalise, in place, the first `heads` heads of width `d` in each row of x (n × width), the
    /// first `scaled` of them then multiplied by `scale`.
    pub fn l2norm_heads(&self, x: &GpuTensor, heads: usize, d: usize, scaled: usize, scale: f32) {
        let (n, width) = (x.shape[0], x.shape[1]);
        assert!(heads * d <= width, "l2norm: {heads} heads of {d} in rows of {width}");
        self.run(&self.l2norm_heads, &[&x.buffer], &[width as u32, d as u32, scaled as u32, scale.to_bits()], (n as u32, heads as u32));
    }

    /// Gated DeltaNet's gates in place over (n × heads): a becomes the decay, b becomes beta.
    pub fn gdn_gates(&self, a: &GpuTensor, b: &GpuTensor, neg_a: &GpuTensor, dt_bias: &GpuTensor) {
        let heads = neg_a.len();
        assert!(a.len() == b.len() && a.len().is_multiple_of(heads) && dt_bias.len() == heads, "gdn_gates: {:?}, {:?} over {heads} heads", a.shape, b.shape);
        self.run(&self.gdn_gates, &[&a.buffer, &b.buffer, &neg_a.buffer, &dt_bias.buffer], &[a.len() as u32, heads as u32, 0, 0], (a.len().div_ceil(256) as u32, 1));
    }

    /// The causal depthwise conv and SiLU over raw (n × ch) after `state` (k − 1 × ch), whose rows
    /// become the last k − 1 of the input: weights `w` are ch × k.
    pub fn conv_silu(&self, raw: &GpuTensor, w: &GpuTensor, state: &GpuTensor) -> GpuTensor {
        self.conv(raw, w, state, true)
    }

    /// The causal depthwise conv over raw (n × ch) after `state` (k − 1 × ch), whose rows become the
    /// last k − 1 of the input, then SiLU if `silu`: weights `w` are ch × k.
    pub fn conv(&self, raw: &GpuTensor, w: &GpuTensor, state: &GpuTensor, silu: bool) -> GpuTensor {
        let (n, ch, k) = (raw.shape[0], raw.shape[1], w.shape[1]);
        assert!(w.shape[0] == ch && state.len() == (k - 1) * ch, "conv: raw {:?}, weights {:?}, state {:?}", raw.shape, w.shape, state.shape);
        let (out, next) = (self.empty(&[n, ch]), self.empty(&state.shape));
        let work = n.max(k - 1) * ch;
        self.run(&self.conv_silu, &[&raw.buffer, &w.buffer, &state.buffer, &out.buffer, &next.buffer], &[n as u32, ch as u32, k as u32, u32::from(silu)], (work.div_ceil(256) as u32, 1));
        self.copy_into(&next, state, 0);
        out
    }

    /// y (n × w) = a[:, oa..oa + w] · b[:, ob..ob + w], elementwise: column ranges of two matrices of
    /// n rows (which may be the same matrix).
    pub fn mul_cols(&self, a: &GpuTensor, oa: usize, b: &GpuTensor, ob: usize, w: usize) -> GpuTensor {
        let (n, wa, wb) = (a.shape[0], a.shape[1], b.shape[1]);
        assert!(b.shape[0] == n && oa + w <= wa && ob + w <= wb, "mul_cols: {w} columns at {oa} of {:?} and at {ob} of {:?}", a.shape, b.shape);
        let y = self.empty(&[n, w]);
        let p = [n as u32, w as u32, wa as u32, oa as u32, wb as u32, ob as u32, 0, 0];
        self.run(&self.mul_cols, &[&a.buffer, &b.buffer, &y.buffer], &p, ((n * w).div_ceil(256) as u32, 1));
        y
    }

    /// The gated delta rule over n tokens from `state` (vh × dk × dv), updated in place: q, k and v
    /// from the conv's output rows (n × ch), the decay and beta (n × vh). The outputs, n × vh·dv.
    #[allow(clippy::too_many_arguments)]
    pub fn delta(&self, mixed: &GpuTensor, decay: &GpuTensor, beta: &GpuTensor, state: &GpuTensor, kk: usize, vh: usize, dk: usize, dv: usize) -> GpuTensor {
        let (n, ch) = (mixed.shape[0], mixed.shape[1]);
        assert!(dk <= 128 && dv <= 128 && vh.is_multiple_of(kk), "delta: heads {kk} → {vh}, widths {dk} × {dv}");
        assert!(ch >= 2 * kk * dk + vh * dv && state.len() == vh * dk * dv, "delta: rows of {ch}, state {:?}", state.shape);
        let o = self.empty(&[n, vh * dv]);
        let p = [n as u32, kk as u32, vh as u32, dk as u32, dv as u32, ch as u32, 0, 0];
        self.run(&self.delta, &[&mixed.buffer, &decay.buffer, &beta.buffer, &state.buffer, &o.buffer], &p, (vh as u32, 1));
        o
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

/// A model's weights as a stream left them: the big ones already on the GPU, the small ones as host
/// values for the decoder to adjust (a norm's 1 + w, a decay's −exp) before it uploads them.
#[derive(Default)]
pub struct Weights {
    gpu: std::collections::HashMap<String, GpuTensor>,
    host: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    /// A weight already on the GPU, by its name in the file.
    pub fn take(&mut self, name: &str) -> Result<GpuTensor, String> {
        self.gpu.remove(name).ok_or_else(|| format!("the weights have no {name}"))
    }

    /// A small weight's shape and values, by its name in the file.
    pub fn host(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        self.host.remove(name).ok_or_else(|| format!("the weights have no {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.gpu.contains_key(name) || self.host.contains_key(name)
    }
}

/// Which tensors of a file a decoder keeps, and as what.
pub type Rule = fn(&str, &[usize]) -> Option<stream::Store>;

/// The `stream::Sink` that puts a file's tensors on the GPU as they arrive.
pub struct Upload<'a> {
    pub gpu: &'a Gpu,
    pub weights: &'a mut Weights,
    pub rule: Rule,
}

impl stream::Sink for Upload<'_> {
    fn store(&self, name: &str, shape: &[usize]) -> Option<stream::Store> {
        (self.rule)(name, shape)
    }

    fn begin(&mut self, name: &str, store: stream::Store, shape: &[usize], _bytes: u64) {
        let t = match store {
            stream::Store::Half => self.gpu.empty_half(shape),
            _ => self.gpu.empty(shape),
        };
        self.weights.gpu.insert(name.to_string(), t);
    }

    fn write(&mut self, name: &str, offset: u64, bytes: &[u8]) {
        if let Some(t) = self.weights.gpu.get(name) {
            self.gpu.write(t, offset, bytes);
        }
    }

    fn small(&mut self, name: &str, shape: &[usize], values: Vec<f32>) {
        self.weights.host.insert(name.to_string(), (shape.to_vec(), values));
    }
}

/// Every tensor of a whole file in memory, through the same stream a browser feeds piece by piece.
pub fn upload_file(gpu: &Gpu, bytes: &[u8], rule: Rule, weights: &mut Weights) -> Result<(), String> {
    let mut s = stream::Stream::new();
    s.feed(bytes, &mut Upload { gpu, weights, rule })?;
    s.finish()
}
