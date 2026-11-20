//! A fitted KV bridge on the GPU: the source's residual streams in, the target's memory out. The
//! arithmetic is prattle-core's `bridge::translate_hidden`, step for step: per target layer, `taps`
//! consecutive source streams and an intercept, then
//! - an attention layer: a ridge map to keys and values, the keys rotated to their positions, then
//!   the paper's learned correction per head;
//! - a short-conv layer (LFM2): a ridge map to its B·x rows, the last `conv_rows` of them its state;
//! - a layer with no map (Qwen3.5's Gated DeltaNet): the source's own state, copied as it is.

use crate::decoder::Decoder;
use crate::{Gpu, GpuTensor};

struct LayerMap {
    source: usize,
    /// The ridge maps transposed to out × in, the layout `Gpu::matmul` takes.
    k: GpuTensor,
    /// Absent for a short-conv layer, whose map is `k` alone.
    v: Option<GpuTensor>,
    /// The folded corrections transposed (dim × dim), when the file carries them.
    ck: Option<GpuTensor>,
    cv: Option<GpuTensor>,
}

/// What a bridge puts in one target layer.
pub enum Carry {
    /// Keys and values, each n × kv width, positions 0..n.
    Kv(GpuTensor, GpuTensor),
    /// A short conv's state: its last rows of B·x (rows × width), oldest first.
    Rows(GpuTensor),
    /// Nothing translated: the source's own state crosses as it is.
    Source,
}

/// What a target layer takes from a bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Kv,
    Rows,
    Source,
}

pub struct Bridge {
    taps: usize,
    dim: usize,
    /// How many B·x rows a short-conv layer's state holds; 0 when no layer is one.
    conv_rows: usize,
    /// Per target layer; `None` for a recurrent layer, whose state is the source's own.
    layers: Vec<Option<LayerMap>>,
}

fn floats(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>, String> {
    let b = view.data();
    Ok(match view.dtype() {
        safetensors::Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        safetensors::Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        safetensors::Dtype::BF16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => return Err(format!("bridge tensors in {other:?} are not read")),
    })
}

fn u32s(view: &safetensors::tensor::TensorView) -> Vec<usize> {
    view.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect()
}

/// A rows × cols matrix as cols × rows.
fn transposed(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

impl Bridge {
    /// From a bridge file's safetensors bytes. A layer with no map takes the source's own state
    /// (Qwen3.5); a layer with a key map alone writes a short conv's rows (LFM2). Granite's input-row
    /// maps belong to a family this engine does not run.
    pub fn load(gpu: &Gpu, bytes: &[u8]) -> Result<Bridge, String> {
        let st = safetensors::SafeTensors::deserialize(bytes).map_err(|e| format!("bridge file: {e}"))?;
        let get = |n: &str| st.tensor(n).map_err(|_| format!("the bridge file has no {n}"));
        let meta = u32s(&get("meta")?);
        let layer_of = u32s(&get("layer_of")?);
        let (dim, taps, conv_rows) = (meta[2], meta.get(3).copied().unwrap_or(1), meta.get(4).copied().unwrap_or(0));
        if meta.get(5).copied().unwrap_or(0) != 0 {
            return Err("this bridge maps a recurrent layer's inputs (Granite), which the WebGPU engine does not run".into());
        }
        let matrix = |n: &str| -> Result<Option<GpuTensor>, String> {
            let Ok(view) = st.tensor(n) else { return Ok(None) };
            let (rows, cols) = (view.shape()[0], view.shape()[1]);
            Ok(Some(gpu.upload(&transposed(&floats(&view)?, rows, cols), &[cols, rows])))
        };
        let mut layers = Vec::with_capacity(layer_of.len());
        for (l, &source) in layer_of.iter().enumerate() {
            let Some(k) = matrix(&format!("k.{l}"))? else {
                layers.push(None);
                continue;
            };
            let v = matrix(&format!("v.{l}"))?;
            if v.is_none() && conv_rows == 0 {
                return Err(format!("layer {l} has a key map and no value map"));
            }
            layers.push(Some(LayerMap { source, k, v, ck: matrix(&format!("ck.{l}"))?, cv: matrix(&format!("cv.{l}"))? }));
        }
        Ok(Bridge { taps, dim, conv_rows, layers })
    }

    fn kind(&self, l: usize) -> Kind {
        match &self.layers[l] {
            None => Kind::Source,
            Some(m) if m.v.is_none() => Kind::Rows,
            Some(_) => Kind::Kv,
        }
    }

    /// Whether this bridge fits `target`: per layer, the bridge writes what the layer takes.
    pub fn fits(&self, target: &Decoder) -> Result<(), String> {
        if self.layers.len() != target.layer_count() {
            return Err(format!("the bridge maps {} layers, the target has {}", self.layers.len(), target.layer_count()));
        }
        match (0..self.layers.len()).find(|&l| self.kind(l) != target.kind(l)) {
            Some(l) => Err(format!("layer {l}: the bridge writes {:?}, and the target's layer takes {:?}", self.kind(l), target.kind(l))),
            None if self.conv_rows != 0 && self.conv_rows != target.conv_rows() => Err(format!("the bridge writes {} conv rows, the target keeps {}", self.conv_rows, target.conv_rows())),
            None => Ok(()),
        }
    }

    /// Per target layer, what it takes: keys and values, a short conv's rows, or the source's state.
    pub fn translate(&self, gpu: &Gpu, hidden: &[GpuTensor], target: &Decoder) -> Vec<Carry> {
        assert_eq!(self.layers.len(), target.layer_count(), "the bridge maps {} layers, the target has {}", self.layers.len(), target.layer_count());
        let (n, h) = (hidden[0].shape[0], hidden[0].shape[1]);
        let ones = gpu.upload(&vec![1.0; n], &[n, 1]);
        let last = hidden.len() - 1;
        let width = target.kv_width();
        let heads = width / self.dim;
        self.layers
            .iter()
            .map(|m| {
                let Some(m) = m.as_ref() else { return Carry::Source };
                let f = gpu.zeros(&[n, self.taps * h + 1]);
                for t in 0..self.taps {
                    gpu.place(&hidden[(m.source + t).min(last)], &f, t * h);
                }
                gpu.place(&ones, &f, self.taps * h);
                let Some(wv) = &m.v else {
                    // the last `conv_rows` tokens' B·x rows, zeros in front of a shorter prompt
                    let (r, take) = (self.conv_rows, n.min(self.conv_rows));
                    let rows = gpu.matmul_rows(&f, n - take, take, &m.k);
                    let width = rows.shape[1];
                    let state = gpu.zeros(&[r, width]);
                    gpu.copy_into(&rows, &state, (r - take) * width);
                    return Carry::Rows(state);
                };
                let k = gpu.matmul(&f, &m.k);
                target.rotate(gpu, &k, 0);
                let v = gpu.matmul(&f, wv);
                let correct = |x: GpuTensor, c: &Option<GpuTensor>| match c {
                    Some(c) => gpu.matmul(&x.view(&[n * heads, self.dim]), c).view(&[n, width]),
                    None => x,
                };
                Carry::Kv(correct(k, &m.ck), correct(v, &m.cv))
            })
            .collect()
    }
}
