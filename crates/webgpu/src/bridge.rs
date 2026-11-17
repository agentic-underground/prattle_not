//! A fitted KV bridge on the GPU: the source's residual streams in, the target's keys and values out.
//! The arithmetic is prattle-core's `bridge::translate_hidden` for a Llama target, step for step: per
//! target layer, `taps` consecutive source streams and an intercept, a ridge map to keys and values,
//! the keys rotated to their positions, then the paper's learned correction per head.

use crate::llama::Llama;
use crate::{Gpu, GpuTensor};

struct LayerMap {
    source: usize,
    /// The ridge maps transposed to out × in, the layout `Gpu::matmul` takes.
    k: GpuTensor,
    v: GpuTensor,
    /// The folded corrections transposed (dim × dim), when the file carries them.
    ck: Option<GpuTensor>,
    cv: Option<GpuTensor>,
}

pub struct Bridge {
    taps: usize,
    dim: usize,
    layers: Vec<LayerMap>,
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
    /// From a bridge file's safetensors bytes. Only key/value maps are read: a recurrent layer's
    /// state maps (LFM2's conv rows, Granite's inputs) belong to families this engine does not run.
    pub fn load(gpu: &Gpu, bytes: &[u8]) -> Result<Bridge, String> {
        let st = safetensors::SafeTensors::deserialize(bytes).map_err(|e| format!("bridge file: {e}"))?;
        let get = |n: &str| st.tensor(n).map_err(|_| format!("the bridge file has no {n}"));
        let meta = u32s(&get("meta")?);
        let layer_of = u32s(&get("layer_of")?);
        let (dim, taps) = (meta[2], meta.get(3).copied().unwrap_or(1));
        if meta.get(4).copied().unwrap_or(0) != 0 {
            return Err("this bridge maps recurrent state, which the WebGPU engine does not run".into());
        }
        let matrix = |n: &str| -> Result<Option<GpuTensor>, String> {
            let Ok(view) = st.tensor(n) else { return Ok(None) };
            let (rows, cols) = (view.shape()[0], view.shape()[1]);
            Ok(Some(gpu.upload(&transposed(&floats(&view)?, rows, cols), &[cols, rows])))
        };
        let mut layers = Vec::with_capacity(layer_of.len());
        for (l, &source) in layer_of.iter().enumerate() {
            let k = matrix(&format!("k.{l}"))?.ok_or_else(|| format!("layer {l} has no key map: a recurrent layer"))?;
            let v = matrix(&format!("v.{l}"))?.ok_or_else(|| format!("layer {l} has no value map"))?;
            layers.push(LayerMap { source, k, v, ck: matrix(&format!("ck.{l}"))?, cv: matrix(&format!("cv.{l}"))? });
        }
        Ok(Bridge { taps, dim, layers })
    }

    /// One (keys, values) pair per target layer, each n × kv-width, as positions 0..n.
    pub fn translate(&self, gpu: &Gpu, hidden: &[GpuTensor], target: &Llama) -> Vec<(GpuTensor, GpuTensor)> {
        assert_eq!(self.layers.len(), target.layer_count(), "the bridge maps {} layers, the target has {}", self.layers.len(), target.layer_count());
        let (n, h) = (hidden[0].shape[0], hidden[0].shape[1]);
        let ones = gpu.upload(&vec![1.0; n], &[n, 1]);
        let last = hidden.len() - 1;
        let width = target.kv_width();
        let heads = width / self.dim;
        self.layers
            .iter()
            .map(|m| {
                let f = gpu.zeros(&[n, self.taps * h + 1]);
                for t in 0..self.taps {
                    gpu.place(&hidden[(m.source + t).min(last)], &f, t * h);
                }
                gpu.place(&ones, &f, self.taps * h);
                let k = gpu.matmul(&f, &m.k);
                target.rotate(gpu, &k, 0);
                let v = gpu.matmul(&f, &m.v);
                let correct = |x: GpuTensor, c: &Option<GpuTensor>| match c {
                    Some(c) => gpu.matmul(&x.view(&[n * heads, self.dim]), c).view(&[n, width]),
                    None => x,
                };
                (correct(k, &m.ck), correct(v, &m.cv))
            })
            .collect()
    }
}
