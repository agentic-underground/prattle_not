//! Liquid AI's LFM2 text decoder (`lfm2` — `LFM2-350M`, `LFM2-1.2B`) from safetensors, its
//! persistent state OPEN. Follows transformers' `modeling_lfm2.py` (its reference torch paths);
//! `tests/hf_reference.rs` pins it.
//!
//! Most layers are short convolutions: one input projection to [B | C | x], a causal depthwise conv
//! (no bias, no activation) over B·x, then C · conv and the output projection. The rest
//! (`full_attn_idxs`) are attention with RMS q/k norms per head and full rotate-half RoPE. The MLP is
//! SwiGLU; the final norm is `embedding_norm`; norms scale by w; the head is tied.
//!
//! STATE, in the open `Kv`: an attention layer's slot is (K, V) `[1, kv_heads, n, head_dim]`, keys
//! rotated; a conv layer's slot is (C, ∅) — C the last `L − 1` rows of B·x `[1, hidden, L − 1]`, and
//! an empty second tensor — what a bridge carries across.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::half::{Embed, Lin, embedding, linear, small};
use crate::llama::{Geometry, Kv};

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub mlp_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub eps: f64,
    pub vocab_size: usize,
    /// Per layer: true for attention, false for a short convolution.
    pub attention: Vec<bool>,
    pub conv: usize,
}

impl Config {
    pub fn from_json(config_json: &[u8]) -> std::result::Result<Config, String> {
        let t: serde_json::Value = serde_json::from_slice(config_json).map_err(|e| e.to_string())?;
        let u = |k: &str| t[k].as_u64().map(|v| v as usize).ok_or(format!("config has no {k}"));
        let hidden_size = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let layers = u("num_hidden_layers")?;
        let attention = match t["layer_types"].as_array() {
            Some(types) => types.iter().map(|v| v == "full_attention").collect(),
            None => {
                let idx: Vec<u64> = t["full_attn_idxs"].as_array().ok_or("config has no full_attn_idxs")?.iter().filter_map(|v| v.as_u64()).collect();
                (0..layers as u64).map(|i| idx.contains(&i)).collect()
            }
        };
        // transformers' `block_auto_adjust_ff_dim`: two thirds of `block_ff_dim`, × the multiplier,
        // rounded up to `block_multiple_of`
        let mut mlp_size = t["block_ff_dim"].as_u64().or(t["intermediate_size"].as_u64()).ok_or("config has no block_ff_dim")? as usize;
        if t["block_auto_adjust_ff_dim"].as_bool().unwrap_or(false) {
            mlp_size = 2 * mlp_size / 3;
            if let Some(m) = t["block_ffn_dim_multiplier"].as_f64() {
                let of = t["block_multiple_of"].as_u64().unwrap_or(256) as usize;
                mlp_size = ((m * mlp_size as f64) as usize).div_ceil(of) * of;
            }
        }
        Ok(Config {
            hidden_size,
            mlp_size,
            layers,
            heads,
            kv_heads: u("num_key_value_heads")?,
            head_dim: hidden_size / heads,
            rope_theta: t["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            eps: t["norm_eps"].as_f64().or(t["block_norm_eps"].as_f64()).unwrap_or(1e-5),
            vocab_size: u("vocab_size")?,
            attention,
            conv: u("conv_L_cache")?,
        })
    }
}

struct Norm {
    weight: Tensor,
    eps: f32,
}

impl Norm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { weight: small(&vb, size, "weight")?, eps: eps as f32 })
    }
    /// From basic ops when `diff`, so gradients flow (the fused kernel has none).
    fn forward(&self, x: &Tensor, diff: bool) -> Result<Tensor> {
        if diff { candle_nn::ops::rms_norm_slow(x, &self.weight, self.eps) } else { candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps) }
    }
}

struct Attention {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: Norm,
    k_norm: Norm,
}

struct ShortConv {
    in_proj: Lin,
    /// `[hidden, L]`
    conv: Tensor,
    out: Lin,
}

enum Mixer {
    Attention(Attention),
    Conv(ShortConv),
}

struct Layer {
    mixer: Mixer,
    operator_norm: Norm,
    ffn_norm: Norm,
    w1: Lin,
    w2: Lin,
    w3: Lin,
}

pub struct Lfm2 {
    differentiable: bool,
    cfg: Config,
    embed: Embed,
    layers: Vec<Layer>,
    norm: Norm,
    head: Lin,
    cos: Tensor,
    sin: Tensor,
}

const MAX_POSITIONS: usize = 8192;

impl Lfm2 {
    pub fn load(config_json: &[u8], weights: impl Into<crate::weights::Weights>, device: &Device) -> std::result::Result<Lfm2, String> {
        Self::load_in(config_json, weights, device, DType::F32)
    }

    /// As `load`, its weights held in `dtype` (`crate::half`).
    pub fn load_in(config_json: &[u8], weights: impl Into<crate::weights::Weights>, device: &Device, dtype: DType) -> std::result::Result<Lfm2, String> {
        Self::from_builder(config_json, weights.into().builder(dtype, device)?)
    }

    /// As `load_in`, from a VarBuilder already made.
    pub fn from_builder(config_json: &[u8], vb: VarBuilder) -> std::result::Result<Lfm2, String> {
        let cfg = Config::from_json(config_json)?;
        let device = vb.device().clone();
        Self::build(cfg, vb, &device).map_err(|e| e.to_string())
    }

    fn build(cfg: Config, root: VarBuilder, device: &Device) -> Result<Lfm2> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let vb = root.pp("model");
        let embed = embedding(cfg.vocab_size, h, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = vb.pp(format!("layers.{i}"));
            let mixer = if cfg.attention[i] {
                let a = l.pp("self_attn");
                Mixer::Attention(Attention {
                    q: linear(h, cfg.heads * hd, a.pp("q_proj"))?,
                    k: linear(h, cfg.kv_heads * hd, a.pp("k_proj"))?,
                    v: linear(h, cfg.kv_heads * hd, a.pp("v_proj"))?,
                    o: linear(cfg.heads * hd, h, a.pp("out_proj"))?,
                    q_norm: Norm::load(hd, cfg.eps, a.pp("q_layernorm"))?,
                    k_norm: Norm::load(hd, cfg.eps, a.pp("k_layernorm"))?,
                })
            } else {
                let c = l.pp("conv");
                Mixer::Conv(ShortConv {
                    in_proj: linear(h, 3 * h, c.pp("in_proj"))?,
                    conv: small(&c, (h, 1, cfg.conv), "conv.weight")?.squeeze(1)?,
                    out: linear(h, h, c.pp("out_proj"))?,
                })
            };
            layers.push(Layer {
                mixer,
                operator_norm: Norm::load(h, cfg.eps, l.pp("operator_norm"))?,
                ffn_norm: Norm::load(h, cfg.eps, l.pp("ffn_norm"))?,
                w1: linear(h, cfg.mlp_size, l.pp("feed_forward.w1"))?,
                w2: linear(cfg.mlp_size, h, l.pp("feed_forward.w2"))?,
                w3: linear(h, cfg.mlp_size, l.pp("feed_forward.w3"))?,
            });
        }
        let head = if root.contains_tensor("lm_head.weight") { linear(h, cfg.vocab_size, root.pp("lm_head"))? } else { Lin::new(embed.embeddings().clone(), None) };
        let inv: Vec<f32> = (0..hd).step_by(2).map(|i| 1.0 / cfg.rope_theta.powf(i as f32 / hd as f32)).collect();
        let inv = Tensor::new(inv.as_slice(), device)?;
        let t = Tensor::arange(0, MAX_POSITIONS as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.reshape((MAX_POSITIONS, 1))?.matmul(&inv.reshape((1, inv.elem_count()))?)?;
        Ok(Lfm2 {
            differentiable: false,
            norm: Norm::load(h, cfg.eps, vb.pp("embedding_norm"))?,
            head,
            cos: freqs.cos()?,
            sin: freqs.sin()?,
            embed,
            layers,
            cfg,
        })
    }

    pub fn set_differentiable(&mut self, on: bool) {
        self.differentiable = on;
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn empty_state(&self) -> Kv {
        Kv::new(self.layers.len())
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// The attention layers' shape (the conv layers' state is carried separately).
    pub fn geometry(&self) -> Geometry {
        Geometry { layers: self.cfg.layers, kv_heads: self.cfg.kv_heads, head_dim: self.cfg.head_dim, rope_theta_bits: self.cfg.rope_theta.to_bits() }
    }

    /// Rotate-half RoPE over all of `[b, heads, n, head_dim]` from position `pos`; `sign` −1 inverts.
    fn rope(&self, x: &Tensor, pos: usize, sign: f64) -> Result<Tensor> {
        let n = x.dim(2)?;
        let (cos, sin) = (self.cos.narrow(0, pos, n)?, (self.sin.narrow(0, pos, n)? * sign)?);
        let x = x.contiguous()?;
        if self.differentiable {
            let half = self.cfg.head_dim / 2;
            let turned = Tensor::cat(&[&x.narrow(3, half, half)?.neg()?, &x.narrow(3, 0, half)?], 3)?;
            let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?.unsqueeze(0)?;
            let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?.unsqueeze(0)?;
            x.broadcast_mul(&cos)? + turned.broadcast_mul(&sin)?
        } else {
            candle_nn::rotary_emb::rope(&x, &cos, &sin)
        }
    }

    /// Rotate (or, `inverse`, de-rotate) keys of an attention layer from absolute position `pos`.
    pub fn rotate_keys(&self, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        self.rope(k, pos, if inverse { -1.0 } else { 1.0 })
    }

    /// Short-conv layer `layer`'s B·x rows `[n, hidden]` — what its state is made of — from the
    /// residual stream `[1, n, hidden]` entering it (`forward_capture`); None for an attention layer.
    pub fn conv_rows(&self, layer: usize, residual: &Tensor) -> Result<Option<Tensor>> {
        let Mixer::Conv(c) = &self.layers[layer].mixer else { return Ok(None) };
        let h = self.cfg.hidden_size;
        let bcx = c.in_proj.forward(&self.layers[layer].operator_norm.forward(residual, false)?)?.squeeze(0)?; // [n, 3h]
        Ok(Some((bcx.narrow(1, 0, h)? * bcx.narrow(1, 2 * h, h)?)?))
    }

    /// How many B·x rows a short-conv layer's state holds.
    pub fn conv_state_rows(&self) -> usize {
        self.cfg.conv - 1
    }

    pub fn forward(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        self.forward_capture(ids, pos, kv, None)
    }

    pub fn forward_all(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        let x = self.run(ids, pos, kv, None)?;
        self.head.forward(&x)
    }

    /// Logits from residual rows `[.., hidden]`: the final norm, the head, and the family's own scaling —
    /// what `forward_capture` does to the last layer's output (a logit lens reads any layer's).
    pub fn unembed(&self, x: &Tensor) -> Result<Tensor> {
        self.head.forward(&self.norm.forward(x, self.differentiable)?)
    }

    pub fn forward_capture(&self, ids: &Tensor, pos: usize, kv: &mut Kv, capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let x = self.run(ids, pos, kv, capture)?;
        let n = x.dim(1)?;
        self.head.forward(&x.narrow(1, n - 1, 1)?)?.squeeze(1)
    }

    fn run(&self, ids: &Tensor, pos: usize, kv: &mut Kv, mut capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let (b, n) = ids.dims2()?;
        let diff = self.differentiable;
        let mut x = self.embed.forward(ids)?;
        let mut slots = kv.layers().to_vec();
        for (layer, slot) in self.layers.iter().zip(slots.iter_mut()) {
            if let Some(cap) = capture.as_deref_mut() {
                cap.push(x.clone());
            }
            let y = layer.operator_norm.forward(&x, diff)?;
            let mixed = match &layer.mixer {
                Mixer::Attention(a) => self.attention(a, &y, pos, slot, b, n)?,
                Mixer::Conv(c) => self.short_conv(c, &y, slot, b, n)?,
            };
            x = (x + mixed)?;
            let y = layer.ffn_norm.forward(&x, diff)?;
            x = (&x + layer.w2.forward(&(candle_nn::ops::silu(&layer.w1.forward(&y)?)? * layer.w3.forward(&y)?)?)?)?;
        }
        *kv = Kv::from_layers(slots);
        self.norm.forward(&x, diff)
    }

    fn attention(&self, a: &Attention, y: &Tensor, pos: usize, slot: &mut Option<(Tensor, Tensor)>, b: usize, n: usize) -> Result<Tensor> {
        let c = &self.cfg;
        let (hd, diff) = (c.head_dim, self.differentiable);
        let q = a.q_norm.forward(&a.q.forward(y)?.reshape((b, n, c.heads, hd))?, diff)?.transpose(1, 2)?;
        let q = self.rope(&q, pos, 1.0)?;
        let k = a.k_norm.forward(&a.k.forward(y)?.reshape((b, n, c.kv_heads, hd))?, diff)?.transpose(1, 2)?;
        let mut k = self.rope(&k, pos, 1.0)?;
        let mut v = a.v.forward(y)?.reshape((b, n, c.kv_heads, hd))?.transpose(1, 2)?.contiguous()?;
        if let Some((pk, pv)) = slot.as_ref() {
            k = Tensor::cat(&[pk, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[pv, &v], 2)?.contiguous()?;
        }
        *slot = Some((k.clone(), v.clone()));
        let o = crate::attend::attend(&q, k, v, pos, 1.0 / (hd as f64).sqrt(), None, diff)?;
        a.o.forward(&o)
    }

    fn short_conv(&self, s: &ShortConv, y: &Tensor, slot: &mut Option<(Tensor, Tensor)>, b: usize, n: usize) -> Result<Tensor> {
        let (h, l) = (self.cfg.hidden_size, self.cfg.conv);
        let bcx = s.in_proj.forward(y)?.transpose(1, 2)?.contiguous()?; // [b, 3h, n]
        let (bg, cg, x) = (bcx.narrow(1, 0, h)?, bcx.narrow(1, h, h)?, bcx.narrow(1, 2 * h, h)?);
        let bx = (bg * x)?;
        let prev = match slot.as_ref() {
            Some((pc, _)) => pc.clone(),
            None => Tensor::zeros((b, h, l - 1), DType::F32, y.device())?,
        };
        let padded = Tensor::cat(&[&prev, &bx], 2)?; // [b, h, l − 1 + n]
        let new_c = padded.narrow(2, n, l - 1)?.contiguous()?;
        // depthwise causal conv: out[t] = Σ_j w[j] · padded[t + j]
        let mut conv = Tensor::zeros((b, h, n), DType::F32, y.device())?;
        for j in 0..l {
            let w = s.conv.narrow(1, j, 1)?.reshape((1, h, 1))?;
            conv = (conv + padded.narrow(2, j, n)?.broadcast_mul(&w)?)?;
        }
        *slot = Some((new_c, Tensor::zeros(0, DType::F32, y.device())?));
        s.out.forward(&(cg * conv)?.transpose(1, 2)?.contiguous()?)
    }
}
