//! A Llama-architecture decoder whose attention state is OPEN: the bridge reads one model's K/V
//! and installs it into another, which candle-transformers' own `Cache` does not allow.
//!
//! Derived from candle-transformers 0.11 `models/llama.rs` (MIT OR Apache-2.0, Copyright the
//! candle authors). Default RoPE only, no flash attention, f32. `tests::matches_candle` pins the
//! arithmetic to the original.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use crate::half::{Embed, Lin, embedding, linear};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::llama::Config;

/// Per-layer attention state, `[batch, kv_heads, seq, head_dim]` for K and for V, keys rotated.
#[derive(Clone)]
pub struct Kv {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl Kv {
    pub fn new(num_layers: usize) -> Self {
        Self { layers: vec![None; num_layers] }
    }

    /// How many positions the state holds.
    pub fn seq_len(&self) -> usize {
        self.layers.first().and_then(|l| l.as_ref()).map(|(k, _)| k.dims()[2]).unwrap_or(0)
    }

    pub fn layers(&self) -> &[Option<(Tensor, Tensor)>] {
        &self.layers
    }

    pub fn from_layers(layers: Vec<Option<(Tensor, Tensor)>>) -> Self {
        Self { layers }
    }
}

/// What a state must match to be installed into another model by direct copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta_bits: u32,
}

/// RMS norm whose arithmetic is the fused kernel, or — `differentiable` — basic ops.
struct Norm {
    weight: Tensor,
    eps: f32,
}

impl Norm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { weight: vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?.to_dtype(DType::F32)?, eps: eps as f32 })
    }

    fn forward(&self, x: &Tensor, differentiable: bool) -> Result<Tensor> {
        if differentiable {
            candle_nn::ops::rms_norm_slow(x, &self.weight, self.eps)
        } else {
            candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps)
        }
    }
}

struct Attention {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    /// Qwen3's per-head RMS norm on queries and keys, before RoPE; absent in Llama-family models.
    q_norm: Option<Norm>,
    k_norm: Option<Norm>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &Config, head_dim: usize) -> Result<Self> {
        let norm = |name: &str| -> Result<Option<Norm>> {
            if vb.contains_tensor(&format!("{name}.weight")) { Norm::load(head_dim, cfg.rms_norm_eps, vb.pp(name)).map(Some) } else { Ok(None) }
        };
        Ok(Self {
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            q: linear(cfg.hidden_size, head_dim * cfg.num_attention_heads, vb.pp("q_proj"))?,
            k: linear(cfg.hidden_size, head_dim * cfg.num_key_value_heads, vb.pp("k_proj"))?,
            v: linear(cfg.hidden_size, head_dim * cfg.num_key_value_heads, vb.pp("v_proj"))?,
            o: linear(head_dim * cfg.num_attention_heads, cfg.hidden_size, vb.pp("o_proj"))?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, rope: &Rope, slot: &mut Option<(Tensor, Tensor)>, diff: bool) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let split = |t: Tensor, h: usize| t.reshape((b, n, h, self.head_dim))?.transpose(1, 2)?.contiguous();
        let normed = |t: Tensor, norm: &Option<Norm>| match norm {
            Some(nm) => nm.forward(&t, diff),
            None => Ok(t),
        };
        let q = rope.apply(&normed(split(self.q.forward(x)?, self.heads)?, &self.q_norm)?, pos, diff)?;
        let mut k = rope.apply(&normed(split(self.k.forward(x)?, self.kv_heads)?, &self.k_norm)?, pos, diff)?;
        let mut v = split(self.v.forward(x)?, self.kv_heads)?;
        if let Some((pk, pv)) = slot.as_ref() {
            k = Tensor::cat(&[pk, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[pv, &v], 2)?.contiguous()?;
        }
        *slot = Some((k.clone(), v.clone()));

        let group = self.heads / self.kv_heads;
        let k = candle_transformers::utils::repeat_kv(k, group)?;
        let v = candle_transformers::utils::repeat_kv(v, group)?;
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if n == 1 { att } else { causal_mask(&att, n, pos)? };
        let p = if diff { candle_nn::ops::softmax(&att, candle_core::D::Minus1)? } else { candle_nn::ops::softmax_last_dim(&att)? };
        crate::observe::attention(&p)?;
        let y = p.matmul(&v.contiguous()?)?;
        self.o.forward(&y.transpose(1, 2)?.reshape((b, n, self.heads * self.head_dim))?)
    }
}

/// Query row i (at absolute position pos+i) may see keys 0..=pos+i.
fn causal_mask(att: &Tensor, n: usize, pos: usize) -> Result<Tensor> {
    let total = pos + n;
    let mask: Vec<u8> = (0..n).flat_map(|i| (0..total).map(move |j| u8::from(j > pos + i))).collect();
    let mask = Tensor::from_slice(&mask, (n, total), att.device())?.broadcast_as(att.shape())?;
    let neg = Tensor::new(f32::NEG_INFINITY, att.device())?.broadcast_as(att.shape())?;
    mask.where_cond(&neg, att)
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(cfg: &Config, head_dim: usize, device: &Device) -> Result<Self> {
        let mut inv: Vec<f32> =
            (0..head_dim).step_by(2).map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32)).collect();
        // Llama 3's `rope_type: llama3`: long wavelengths slowed by `factor`, short ones kept, the band
        // between blended — Meta's remap, as transformers and candle apply it.
        if let Some(r) = cfg.rope_scaling.as_ref().filter(|r| matches!(r.rope_type, candle_transformers::models::llama::Llama3RopeType::Llama3)) {
            let old = r.original_max_position_embeddings as f32;
            let (low, high) = (old / r.low_freq_factor, old / r.high_freq_factor);
            for f in inv.iter_mut() {
                let wavelen = 2.0 * std::f32::consts::PI / *f;
                *f = if wavelen < high {
                    *f
                } else if wavelen > low {
                    *f / r.factor
                } else {
                    let smooth = (old / wavelen - r.low_freq_factor) / (r.high_freq_factor - r.low_freq_factor);
                    (1.0 - smooth) * *f / r.factor + smooth * *f
                };
            }
        }
        let inv = Tensor::new(inv.as_slice(), device)?;
        let t = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.reshape((cfg.max_position_embeddings, 1))?.matmul(&inv.reshape((1, inv.elem_count()))?)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    fn apply(&self, x: &Tensor, pos: usize, diff: bool) -> Result<Tensor> {
        let n = x.dim(2)?;
        let (cos, sin) = (self.cos.narrow(0, pos, n)?, self.sin.narrow(0, pos, n)?);
        if !diff {
            return candle_nn::rotary_emb::rope(x, &cos, &sin);
        }
        // x·cos + rotate_half(x)·sin, with rotate_half(x) = [−x₂, x₁] — the fused kernel's arithmetic.
        let half = x.dim(3)? / 2;
        let (x1, x2) = (x.narrow(3, 0, half)?, x.narrow(3, half, half)?);
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
        let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?.unsqueeze(0)?;
        x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
    }
}

struct Block {
    norm1: Norm,
    attn: Attention,
    norm2: Norm,
    gate: Lin,
    up: Lin,
    down: Lin,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &Config, head_dim: usize) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            norm1: Norm::load(h, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Attention::load(vb.pp("self_attn"), cfg, head_dim)?,
            norm2: Norm::load(h, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            gate: linear(h, i, vb.pp("mlp.gate_proj"))?,
            up: linear(h, i, vb.pp("mlp.up_proj"))?,
            down: linear(i, h, vb.pp("mlp.down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, rope: &Rope, slot: &mut Option<(Tensor, Tensor)>, diff: bool) -> Result<Tensor> {
        let x = (self.attn.forward(&self.norm1.forward(x, diff)?, pos, rope, slot, diff)? + x)?;
        let h = self.norm2.forward(&x, diff)?;
        let mlp = self.down.forward(&(candle_nn::ops::silu(&self.gate.forward(&h)?)? * self.up.forward(&h)?)?)?;
        mlp + x
    }
}

pub struct Llama {
    /// Build RMS norm, softmax and RoPE from basic ops so gradients flow (candle's fused kernels have
    /// no backward). Inference leaves it off.
    differentiable: bool,
    embed: Embed,
    blocks: Vec<Block>,
    norm: Norm,
    head: Lin,
    rope: Rope,
    geometry: Geometry,
}

impl Llama {
    /// A Llama-family model, its head width hidden / heads.
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Self::load_with_head_dim(vb, cfg, cfg.hidden_size / cfg.num_attention_heads)
    }

    /// With the head width given — Qwen3 declares `head_dim` apart from hidden / heads.
    pub fn load_with_head_dim(vb: VarBuilder, cfg: &Config, head_dim: usize) -> Result<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let head = if cfg.tie_word_embeddings {
            Lin::new(embed.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let blocks =
            (0..cfg.num_hidden_layers).map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg, head_dim)).collect::<Result<_>>()?;
        Ok(Self {
            differentiable: false,
            norm: Norm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            rope: Rope::new(cfg, head_dim, vb.device())?,
            geometry: Geometry {
                layers: cfg.num_hidden_layers,
                kv_heads: cfg.num_key_value_heads,
                head_dim,
                rope_theta_bits: cfg.rope_theta.to_bits(),
            },
            embed,
            blocks,
            head,
        })
    }

    /// Rotate (or, with `inverse`, de-rotate) keys `[batch, kv_heads, n, head_dim]` that start at
    /// absolute position `pos` — the bridge maps keys in de-rotated space.
    pub fn rotate_keys(&self, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        let n = k.dim(2)?;
        let cos = self.rope.cos.narrow(0, pos, n)?;
        let sin = self.rope.sin.narrow(0, pos, n)?;
        // rope(x) = x·cos + rotate_half(x)·sin; its inverse is the same with −sin.
        let sin = if inverse { sin.neg()? } else { sin };
        candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)
    }

    pub fn set_differentiable(&mut self, on: bool) {
        self.differentiable = on;
    }

    /// The token embeddings of `ids` (`[batch, n]`) → `[batch, n, hidden]`.
    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed.forward(ids)
    }

    pub fn hidden_size(&self) -> usize {
        self.embed.embeddings().dim(1).unwrap_or(0)
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub fn empty_state(&self) -> Kv {
        Kv::new(self.blocks.len())
    }

    /// Run `ids` (`[batch, n]`) from absolute position `pos`, extending `kv`; logits of the last
    /// position, `[batch, vocab]`.
    pub fn forward(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        self.forward_capture(ids, pos, kv, None)
    }

    /// Logits at EVERY position of `ids`, `[batch, n, vocab]` — what a distribution-matching loss needs.
    pub fn forward_all(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        let mut x = self.embed.forward(ids)?;
        for (block, slot) in self.blocks.iter().zip(kv.layers.iter_mut()) {
            x = block.forward(&x, pos, &self.rope, slot, self.differentiable)?;
        }
        self.head.forward(&self.norm.forward(&x, self.differentiable)?)?.to_dtype(DType::F32)
    }

    /// As `forward`, also pushing each layer's INPUT residual stream `[batch, n, hidden]` into
    /// `capture` — the features a bridge can map from.
    pub fn forward_capture(&self, ids: &Tensor, pos: usize, kv: &mut Kv, mut capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let n = ids.dim(1)?;
        let mut x = self.embed.forward(ids)?;
        for (block, slot) in self.blocks.iter().zip(kv.layers.iter_mut()) {
            if let Some(c) = capture.as_deref_mut() {
                c.push(x.clone());
            }
            x = block.forward(&x, pos, &self.rope, slot, self.differentiable)?;
        }
        let x = self.norm.forward(&x, self.differentiable)?.i((.., n - 1, ..))?.contiguous()?;
        self.head.forward(&x)?.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use candle_transformers::models::llama as upstream;

    fn tiny() -> Config {
        let raw: upstream::LlamaConfig = serde_json::from_str(
            r#"{"hidden_size":32,"intermediate_size":64,"vocab_size":50,"num_hidden_layers":2,
                "num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-5,"rope_theta":10000.0,
                "max_position_embeddings":64,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        raw.into_config(false)
    }

    /// A listener hears every attention layer's probabilities — one `[1, heads, queries, keys]` per
    /// layer, each row summing to 1 over the keys — and listening changes no logit.
    #[test]
    fn a_listener_hears_each_layers_attention_and_changes_nothing() {
        let cfg = tiny();
        let map = VarMap::new();
        let model = Llama::load(VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu), &cfg).unwrap();
        let ids = Tensor::new(&[3u32, 7, 11, 2, 9], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        let plain = model.forward(&ids, 0, &mut model.empty_state()).unwrap();
        let heard = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let into = heard.clone();
        crate::observe::listen(move |p| Ok(into.borrow_mut().push(p.clone())));
        let traced = model.forward(&ids, 0, &mut model.empty_state()).unwrap();
        crate::observe::stop();
        let rows = heard.borrow();
        assert_eq!(rows.len(), 2, "one hearing per attention layer");
        for p in rows.iter() {
            assert_eq!(p.dims(), &[1, 4, 5, 5], "a hearing is [1, heads, queries, keys]");
            for s in p.sum(3).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap() {
                assert!((s - 1.0).abs() < 1e-5, "a row sums to {s}");
            }
        }
        let diff = (&plain - &traced).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(diff, 0.0, "listening changed the logits");
    }

    #[test]
    fn inverse_rope_undoes_rope_and_rope_changes_keys() {
        let cfg = tiny();
        let map = VarMap::new();
        let model = Llama::load(VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu), &cfg).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 5, 8), &Device::Cpu).unwrap();
        let rotated = model.rotate_keys(&k, 3, false).unwrap();
        let back = model.rotate_keys(&rotated, 3, true).unwrap();
        let moved = (rotated - &k).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let err = (back - &k).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(moved > 1e-2, "rotation at position 3 did not change the keys");
        assert!(err < 1e-5, "de-rotation left an error of {err}");
    }

    #[test]
    fn the_differentiable_forward_agrees_and_carries_gradients() {
        let cfg = tiny();
        let map = VarMap::new();
        let mut model = Llama::load(VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu), &cfg).unwrap();
        let ids = Tensor::new(&[3u32, 7, 11, 2, 9], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        let fast = model.forward(&ids, 0, &mut model.empty_state()).unwrap();
        model.set_differentiable(true);
        let slow = model.forward(&ids, 0, &mut model.empty_state()).unwrap();
        let diff = (&fast - &slow).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-4, "the differentiable forward differs by {diff}");
        let grads = slow.sum_all().unwrap().backward().expect("backward through the model");
        let w = map.data().lock().unwrap().get("model.layers.0.input_layernorm.weight").unwrap().as_tensor().clone();
        let g = grads.get(&w).expect("no gradient reached the first norm").abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(g > 0.0, "the gradient at the first norm is zero");
    }

    #[test]
    fn weights_at_half_precision_give_the_same_logits_and_an_f32_state() {
        let cfg = tiny();
        let map = VarMap::new();
        let full = Llama::load(VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu), &cfg).unwrap();
        let tensors: std::collections::HashMap<String, Tensor> =
            map.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().to_dtype(DType::F16).unwrap())).collect();
        let half = Llama::load(VarBuilder::from_tensors(tensors, DType::F16, &Device::Cpu), &cfg).unwrap();
        let ids = Tensor::new(&[3u32, 7, 11, 2, 9], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        let a = full.forward(&ids, 0, &mut full.empty_state()).unwrap();
        let mut kv = half.empty_state();
        let b = half.forward(&ids, 0, &mut kv).expect("a half-precision model runs");
        assert_eq!(b.dtype(), DType::F32, "logits come out in f32");
        assert_eq!(kv.layers()[0].as_ref().unwrap().0.dtype(), DType::F32, "the state a bridge reads stays f32");
        let scale = a.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let worst = (&a - &b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(worst < 0.02 * scale, "half-precision weights move the logits by {worst} (largest {scale})");
    }

    #[test]
    fn matches_candle() {
        let cfg = tiny();
        let map = VarMap::new();
        let vb = VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu);
        let ours = Llama::load(vb.clone(), &cfg).unwrap();
        let theirs = upstream::Llama::load(vb, &cfg).unwrap();
        let mut cache = upstream::Cache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();
        let mut kv = ours.empty_state();
        for (ids, pos) in [(vec![3u32, 7, 11, 2, 9], 0), (vec![5], 5), (vec![1], 6)] {
            let t = Tensor::new(ids.as_slice(), &Device::Cpu).unwrap().unsqueeze(0).unwrap();
            let a = ours.forward(&t, pos, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = theirs.forward(&t, pos, &mut cache).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            assert!(worst < 1e-4, "logits differ from candle by {worst} at pos {pos}");
        }
        assert_eq!(kv.seq_len(), 7);
    }
}
