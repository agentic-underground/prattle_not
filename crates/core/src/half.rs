//! Weights at whatever precision they were loaded in, activations always f32. A model whose weights do
//! not fit a card in f32 loads them at half precision; every matmul casts its input to the weights'
//! precision and its output back, so norms, RoPE, softmax and the K/V a bridge reads stay f32.
//!
//! A model too large even at half precision has its LINEAR layers quantized at load (`quantizing`):
//! each one's weight is read on the host, quantized there and only the quantized form crosses to the
//! card — a 32B never exists on its GPU at full width. Norms and embeddings stay at half precision.

use std::cell::RefCell;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Embedding, Linear, Module, VarBuilder};

pub enum Lin {
    Dense(Linear),
    Quant(QMatMul),
}

impl Lin {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Lin::Dense(Linear::new(weight, bias))
    }
}

impl Module for Lin {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Lin::Dense(l) => l.forward(&x.to_dtype(l.weight().dtype())?)?.to_dtype(DType::F32),
            // Inside a gradient graph (the correction trains through a frozen target) the quantized kernel
            // runs as `QGrad`, whose backward dequantizes the weight only while it is used — a backward
            // that kept every layer's f16 weight held a whole 32B in f16 and ran its card out of memory.
            Lin::Quant(QMatMul::QTensor(w)) if x.track_op() => x.to_dtype(DType::F32)?.contiguous()?.apply_op1(QGrad(w.clone())),
            Lin::Quant(q) if x.track_op() => q.forward_via_f16(x)?.to_dtype(DType::F32),
            Lin::Quant(q) => q.forward(&x.to_dtype(DType::F32)?.contiguous()?),
        }
    }
}

/// The quantized matmul `x · Wᵀ` with a backward: the forward is candle's own quantized kernel; the
/// backward `∂x = ∂y · W` dequantizes W to f16 for the one product and lets it go. The weight is frozen,
/// so it gets no gradient.
struct QGrad(Arc<QTensor>);

impl candle_core::CustomOp1 for QGrad {
    fn name(&self) -> &'static str {
        "qmatmul-with-input-grad"
    }
    fn cpu_fwd(&self, storage: &candle_core::CpuStorage, layout: &candle_core::Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::CustomOp1::cpu_fwd(&*self.0, storage, layout)
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, storage: &candle_core::CudaStorage, layout: &candle_core::Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        candle_core::CustomOp1::cuda_fwd(&*self.0, storage, layout)
    }
    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        let w = self.0.dequantize_f16(grad.device())?; // [out, in]
        let g = grad.to_dtype(DType::F16)?;
        let w = match *g.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?,
            [b, _, _] => w.broadcast_left(b)?,
            _ => w,
        };
        Ok(Some(g.matmul(&w)?.to_dtype(DType::F32)?))
    }
}

/// While a model is being built under `quantizing`: the quantization, and every linear weight already
/// quantized on the host (taken as each layer is built).
struct Quantizing {
    quant: GgmlDType,
    ready: std::collections::HashMap<String, QTensor>,
}

impl Quantizing {
    /// The weight `name`, quantized ahead on the host, placed on `device` — its layer's card, which is
    /// the VarBuilder's, so one model's layers can span several cards.
    fn take(&mut self, name: &str, dims: &[usize], device: &Device) -> Result<QMatMul> {
        let host = self.ready.remove(name).ok_or_else(|| candle_core::Error::Msg(format!("{name}: not quantized ahead of the build")))?;
        if host.shape().dims() != dims {
            candle_core::bail!("{name}: shape {:?}, expected {dims:?}", host.shape());
        }
        let storage = QStorage::from_data(host.data()?, device, self.quant)?;
        QMatMul::from_qtensor(QTensor::new(storage, host.shape().clone())?)
    }
}

/// A mixture of experts' stacked weights `[experts, out, in]`: one tensor at the loaded precision, or
/// each expert quantized on its own (`quantizing`).
pub enum Experts {
    Dense(Tensor),
    Quant(Vec<QMatMul>),
}

impl Experts {
    /// Expert `e`'s `xs · Wₑᵀ`, `xs` `[t, in]` in f32, out f32.
    pub fn forward(&self, e: usize, xs: &Tensor) -> Result<Tensor> {
        match self {
            Experts::Dense(w) => xs.to_dtype(w.dtype())?.matmul(&w.get(e)?.t()?)?.to_dtype(DType::F32),
            // inside a gradient graph, the backward `QGrad` gives a linear layer — or no gradient flows
            // through the experts to the layers below
            Experts::Quant(q) => match &q[e] {
                QMatMul::QTensor(w) if xs.track_op() => xs.contiguous()?.apply_op1(QGrad(w.clone())),
                m => m.forward(&xs.contiguous()?),
            },
        }
    }
}

/// Stacked expert weights `name` `[experts, out, in]` from `vb` — each expert quantized while a model is
/// built under `quantizing`.
pub fn experts(vb: &VarBuilder, (n, out_dim, in_dim): (usize, usize, usize), name: &str) -> Result<Experts> {
    let full = format!("{}.{name}", vb.prefix());
    let quantized = QUANTIZING.with(|q| -> Result<Option<Experts>> {
        let mut q = q.borrow_mut();
        let Some(q) = q.as_mut() else { return Ok(None) };
        (0..n).map(|e| q.take(&format!("{full}#{e}"), &[out_dim, in_dim], vb.device())).collect::<Result<Vec<_>>>().map(|v| Some(Experts::Quant(v)))
    })?;
    match quantized {
        Some(x) => Ok(x),
        None => Ok(Experts::Dense(vb.get((n, out_dim, in_dim), name)?)),
    }
}

thread_local! {
    static QUANTIZING: RefCell<Option<Quantizing>> = const { RefCell::new(None) };
}

/// Build a model with every `linear` quantized to `quant`, placed on `device`. Every 2-D `.weight` in
/// `weights` that is not an embedding is quantized first, on all cores at once (quantizing a 32B on one
/// core took minutes); everything else `build` reads comes from its own VarBuilder, unchanged.
pub fn quantizing<T>(quant: GgmlDType, weights: &Shared, device: &Device, build: impl FnOnce() -> Result<T>) -> Result<T> {
    with_quantized(quant, weights.quantize_linears(quant)?, device, build)
}

/// `with_quantized` when the weights come already quantized (`ready`), `quantizing` otherwise.
pub fn quantize_with<T>(quant: GgmlDType, weights: &Shared, ready: Option<std::collections::HashMap<String, QTensor>>, device: &Device, build: impl FnOnce() -> Result<T>) -> Result<T> {
    match ready {
        Some(r) => with_quantized(quant, r, device, build),
        None => quantizing(quant, weights, device, build),
    }
}

/// As `quantizing`, with the linear weights already quantized — read from a pre-quantized file
/// (`crate::fast_llm`), keyed as `Shared::quantize_linears` keys them — so none is quantized again.
pub fn with_quantized<T>(quant: GgmlDType, ready: std::collections::HashMap<String, QTensor>, device: &Device, build: impl FnOnce() -> Result<T>) -> Result<T> {
    let _ = device; // each weight goes to its own builder's device
    QUANTIZING.with(|q| *q.borrow_mut() = Some(Quantizing { quant, ready }));
    let out = build();
    QUANTIZING.with(|q| *q.borrow_mut() = None);
    out
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Lin> {
    let quantized = QUANTIZING.with(|q| -> Result<Option<Lin>> {
        let mut q = q.borrow_mut();
        let Some(q) = q.as_mut() else { return Ok(None) };
        let name = format!("{}.weight", vb.prefix());
        if !q.ready.contains_key(&name) {
            return Ok(None); // not quantized ahead (a router): it stays at the builder's precision
        }
        Ok(Some(Lin::Quant(q.take(&name, &[out_dim, in_dim], vb.device())?)))
    })?;
    match quantized {
        Some(l) => Ok(l),
        None => candle_nn::linear_no_bias(in_dim, out_dim, vb).map(Lin::Dense),
    }
}

/// One safetensors buffer read by several VarBuilders at once — the card's and the host's — without a
/// second copy of a model that may be tens of GB. A PART of a larger file (a FAST_LLM's `source.` or
/// `target.`) is read in place: its tensors are the file's names under `prefix`, the prefix removed,
/// less any under `hidden`.
#[derive(Clone)]
pub struct Shared {
    file: Arc<candle_core::safetensors::BufferedSafetensors>,
    prefix: String,
    hidden: Option<String>,
}

impl Shared {
    pub fn new(weights: Vec<u8>) -> Result<Self> {
        Ok(Shared { file: Arc::new(candle_core::safetensors::BufferedSafetensors::new(weights)?), prefix: String::new(), hidden: None })
    }

    /// The tensors under `prefix`, less those under `hidden` (both full names in the file), as a model
    /// of their own — the same buffer, never copied.
    pub fn part(&self, prefix: &str, hidden: Option<&str>) -> Self {
        Shared { file: self.file.clone(), prefix: format!("{}{prefix}", self.prefix), hidden: hidden.map(str::to_string) }
    }

    /// Whether `self` and `other` read one buffer.
    pub fn shares_buffer(&self, other: &Shared) -> bool {
        Arc::ptr_eq(&self.file, &other.file)
    }

    pub fn builder(&self, dtype: DType, device: &Device) -> VarBuilder<'static> {
        VarBuilder::from_backend(Box::new(self.clone()), dtype, device.clone())
    }

    fn full(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    /// Every tensor this reads, by its name here, with its shape.
    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        self.file
            .tensors()
            .into_iter()
            .filter(|(n, _)| self.hidden.as_ref().is_none_or(|h| !n.starts_with(h.as_str())))
            .filter_map(|(n, v)| n.strip_prefix(self.prefix.as_str()).map(|rest| (rest.to_string(), v.shape().to_vec())))
            .collect()
    }

    /// One tensor's stored bytes.
    pub fn data(&self, name: &str) -> Result<&[u8]> {
        Ok(self.file.get(&self.full(name))?.data())
    }

    /// Every 2-D `.weight` that is neither an embedding nor a router, and every expert of every stacked
    /// `block_sparse_moe` weight (keyed `name#e`), quantized on the host — one worker per allowed thread (`device::cpu_workers`),
    /// each taking the next tensor, at most 16 so the f32 copies in flight stay a few GB. A router stays at the
    /// builder's precision: quantizing it could change which experts a token picks.
    pub fn quantize_linears(&self, quant: GgmlDType) -> Result<std::collections::HashMap<String, QTensor>> {
        // one item per tensor: a linear weight, or a whole expert stack (loaded once, its experts quantized
        // in turn — loading the stack per expert re-read it 72 times for a 72-expert layer)
        let mut items: Vec<(String, bool)> = Vec::new();
        // a multimodal checkpoint's towers are not loaded, and their widths need not divide a block
        let tensors = self.tensors();
        let language_only = tensors.iter().any(|(n, _)| n.contains("language_model."));
        for (n, v) in tensors {
            let rank = v.len();
            if !n.ends_with(".weight") || n.contains("embed") || n.contains("router") || (language_only && !n.contains("language_model.")) {
                continue;
            }
            if rank == 2 {
                items.push((n, false));
            } else if rank == 3 && n.contains("block_sparse_moe") {
                items.push((n, true));
            }
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let workers = crate::device::cpu_workers().min(items.len().max(1));
        let parts: Vec<Result<Vec<(String, QTensor)>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| -> Result<Vec<(String, QTensor)>> {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some((name, stack)) = items.get(i) else { break Ok(out) };
                            if !stack {
                                let w = SimpleBackend::get_unchecked(self, name, DType::F32, &Device::Cpu)?;
                                out.push((name.clone(), QTensor::quantize(&w, quant)?));
                                continue;
                            }
                            // the stack at its stored precision, each expert widened to f32 on its own — as a
                            // fresh tensor, not a view: quantizing reads the storage from its start, so a view at
                            // expert e's offset (and `copy`, which keeps the offset) gives every expert expert 0's
                            let all = SimpleBackend::get_unchecked(self, name, DType::BF16, &Device::Cpu)?;
                            for e in 0..all.dim(0)? {
                                let slice = all.get(e)?;
                                let w = Tensor::from_vec(slice.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, slice.dims(), &Device::Cpu)?;
                                out.push((format!("{name}#{e}"), QTensor::quantize(&w, quant)?));
                            }
                        }
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a quantizing worker panicked")).collect()
        });
        let mut ready = std::collections::HashMap::new();
        for part in parts {
            ready.extend(part?);
        }
        Ok(ready)
    }
}

impl SimpleBackend for Shared {
    fn get(&self, s: Shape, name: &str, h: candle_nn::Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        SimpleBackend::get(&*self.file, s, &self.full(name), h, dtype, dev)
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        SimpleBackend::get_unchecked(&*self.file, &self.full(name), dtype, dev)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        SimpleBackend::contains_tensor(&*self.file, &self.full(name))
    }
}

pub struct Embed(Embedding);

impl Embed {
    pub fn embeddings(&self) -> &Tensor {
        self.0.embeddings()
    }
}

impl Module for Embed {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.0.forward(ids)?.to_dtype(DType::F32)
    }
}

pub fn embedding(vocab: usize, hidden: usize, vb: VarBuilder) -> Result<Embed> {
    candle_nn::embedding(vocab, hidden, vb).map(Embed)
}

/// A small parameter — a norm's scale, a bias — always in f32.
pub fn small(vb: &VarBuilder, shape: impl Into<candle_core::Shape>, name: &str) -> Result<Tensor> {
    vb.get(shape, name)?.to_dtype(DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quantized layer passes the input the gradient its dequantized weight would.
    #[test]
    fn a_quantized_layer_passes_back_its_dequantized_weights_gradient() {
        let cpu = Device::Cpu;
        let w = Tensor::randn(0f32, 1.0, (64, 256), &cpu).unwrap();
        let q = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
        let dense = q.dequantize(&cpu).unwrap();
        let lin = Lin::Quant(QMatMul::from_qtensor(q).unwrap());
        let x = candle_core::Var::randn(0f32, 1.0, (2, 3, 256), &cpu).unwrap();
        let g = |y: Tensor| y.sqr().unwrap().sum_all().unwrap().backward().unwrap().get(x.as_tensor()).unwrap().clone();
        let quantized = g(lin.forward(x.as_tensor()).unwrap());
        let reference = g(x.as_tensor().broadcast_matmul(&dense.t().unwrap()).unwrap());
        let rel = ((&quantized - &reference).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap()
            / reference.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap())
        .sqrt();
        assert!(rel < 2e-2, "the quantized layer's input gradient is {rel:.2e} from its dequantized weight's");
    }
}
