//! Gemma 3's text decoder with its attention state OPEN — BROADEN_SUPPORT's second loader, for Gemma 3
//! and TranslateGemma. Follows transformers' `Gemma3TextModel` (as candle-transformers 0.11
//! `models/gemma3.rs` does, MIT OR Apache-2.0, Copyright the candle authors); `tests/hf_reference.rs`
//! pins the arithmetic to transformers. f32, forward only.
//!
//! What differs from a Llama-family decoder: embeddings scaled by √hidden; every RMS norm scales by
//! `1 + w`, before AND after attention and the MLP; per-head q/k norms; a GELU-tanh MLP; local
//! (sliding-window) layers with their own RoPE base, and global layers every `sliding_window_pattern`
//! (or as `layer_types` lists) with `rope_theta` and linear RoPE scaling; scores scaled by
//! `query_pre_attn_scalar^-½`; the output head tied to the embeddings.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear_no_bias as linear};

use crate::llama::{Geometry, Kv};

/// The text config, with transformers' `Gemma3TextConfig` defaults for what a multimodal
/// checkpoint's `text_config` leaves out.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_pre_attn_scalar: f64,
    pub rope_theta: f32,
    pub rope_local_base: f32,
    /// Linear RoPE scaling on the GLOBAL layers (1 when absent).
    pub rope_factor: f32,
    pub sliding_window: usize,
    pub vocab_size: usize,
    pub eps: f64,
    /// Per layer: true for global (full) attention, false for sliding-window.
    pub global: Vec<bool>,
}

impl Config {
    pub fn from_json(config_json: &[u8]) -> std::result::Result<Config, String> {
        let root: serde_json::Value = serde_json::from_slice(config_json).map_err(|e| e.to_string())?;
        let t = if root["text_config"].is_object() { &root["text_config"] } else { &root };
        let u = |k: &str, d: usize| t[k].as_u64().map_or(d, |v| v as usize);
        let f = |k: &str, d: f64| t[k].as_f64().unwrap_or(d);
        let layers = u("num_hidden_layers", 26);
        let global = match t["layer_types"].as_array() {
            Some(types) => types.iter().map(|v| v == "full_attention").collect(),
            None => {
                let pattern = u("sliding_window_pattern", 6);
                (0..layers).map(|i| (i + 1) % pattern == 0).collect()
            }
        };
        // the global layers' scaling: `rope_scaling` (older configs) or `rope_parameters.full_attention`
        // (newer ones, e.g. TranslateGemma)
        let scaling = if t["rope_scaling"].is_object() { &t["rope_scaling"] } else { &t["rope_parameters"]["full_attention"] };
        let rope_factor = match scaling["rope_type"].as_str() {
            Some("linear") => scaling["factor"].as_f64().unwrap_or(1.0) as f32,
            None | Some("default") => 1.0,
            Some(other) => return Err(format!("Gemma 3 RoPE scaling {other:?} is not supported")),
        };
        Ok(Config {
            hidden_size: u("hidden_size", 2304),
            intermediate_size: u("intermediate_size", 9216),
            layers,
            heads: u("num_attention_heads", 8),
            kv_heads: u("num_key_value_heads", 4),
            head_dim: u("head_dim", 256),
            query_pre_attn_scalar: f("query_pre_attn_scalar", 256.0),
            rope_theta: f("rope_theta", 1_000_000.0) as f32,
            rope_local_base: f("rope_local_base_freq", 10_000.0) as f32,
            rope_factor,
            sliding_window: u("sliding_window", 4096),
            vocab_size: u("vocab_size", 262_208),
            eps: f("rms_norm_eps", 1e-6),
            global,
        })
    }
}

/// RMS norm scaling by `1 + w`.
struct Norm {
    weight: Tensor,
    eps: f64,
}

impl Norm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { weight: (vb.get(size, "weight")? + 1.0)?, eps })
    }
    fn forward(&self, x: &Tensor, diff: bool) -> Result<Tensor> {
        if diff {
            candle_nn::ops::rms_norm_slow(x, &self.weight, self.eps as f32)
        } else {
            candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
        }
    }
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(base: f32, factor: f32, head_dim: usize, max: usize, device: &Device) -> Result<Self> {
        let inv: Vec<f32> = (0..head_dim).step_by(2).map(|i| 1.0 / base.powf(i as f32 / head_dim as f32) / factor).collect();
        let inv = Tensor::new(inv.as_slice(), device)?;
        let t = Tensor::arange(0, max as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.reshape((max, 1))?.matmul(&inv.reshape((1, inv.elem_count()))?)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }
    /// Rotate `[batch, heads, n, head_dim]` from absolute position `pos`; `sign` −1 de-rotates. With
    /// `diff`, from basic ops (the fused kernel has no backward).
    fn apply(&self, x: &Tensor, pos: usize, sign: f64, diff: bool) -> Result<Tensor> {
        let n = x.dim(2)?;
        let (cos, sin) = (self.cos.narrow(0, pos, n)?, (self.sin.narrow(0, pos, n)? * sign)?);
        if !diff {
            return candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin);
        }
        let half = x.dim(3)? / 2;
        let (x1, x2) = (x.narrow(3, 0, half)?, x.narrow(3, half, half)?);
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
        let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?.unsqueeze(0)?;
        x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
    }
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Norm,
    k_norm: Norm,
    input_norm: Norm,
    post_attention_norm: Norm,
    pre_ffn_norm: Norm,
    post_ffn_norm: Norm,
    gate: Linear,
    up: Linear,
    down: Linear,
    global: bool,
}

pub struct Gemma3 {
    /// Build norms, softmax and RoPE from basic ops so gradients flow; inference leaves it off.
    differentiable: bool,
    cfg: Config,
    embed: Embedding,
    layers: Vec<Layer>,
    norm: Norm,
    head: Linear,
    local_rope: Rope,
    global_rope: Rope,
}

/// Positions the RoPE tables cover — the prompts bridged here are far shorter.
const MAX_POSITIONS: usize = 8192;

impl Gemma3 {
    /// From `config.json` and ONE safetensors buffer — a text-only checkpoint (`model.*`) or a
    /// multimodal one (`language_model.model.*`, the vision tower ignored).
    pub fn load(config_json: &[u8], weights: Vec<u8>, device: &Device) -> std::result::Result<Gemma3, String> {
        let cfg = Config::from_json(config_json)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device).map_err(|e| e.to_string())?;
        let prefix = if vb.contains_tensor("language_model.model.embed_tokens.weight") { "language_model.model" } else { "model" };
        Self::build(cfg, vb.pp(prefix), device).map_err(|e| e.to_string())
    }

    fn build(cfg: Config, vb: VarBuilder, device: &Device) -> Result<Gemma3> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let embed = embedding(cfg.vocab_size, h, vb.pp("embed_tokens"))?;
        let layers = (0..cfg.layers)
            .map(|i| {
                let l = vb.pp(format!("layers.{i}"));
                let a = l.pp("self_attn");
                Ok(Layer {
                    q: linear(h, cfg.heads * hd, a.pp("q_proj"))?,
                    k: linear(h, cfg.kv_heads * hd, a.pp("k_proj"))?,
                    v: linear(h, cfg.kv_heads * hd, a.pp("v_proj"))?,
                    o: linear(cfg.heads * hd, h, a.pp("o_proj"))?,
                    q_norm: Norm::load(hd, cfg.eps, a.pp("q_norm"))?,
                    k_norm: Norm::load(hd, cfg.eps, a.pp("k_norm"))?,
                    input_norm: Norm::load(h, cfg.eps, l.pp("input_layernorm"))?,
                    post_attention_norm: Norm::load(h, cfg.eps, l.pp("post_attention_layernorm"))?,
                    pre_ffn_norm: Norm::load(h, cfg.eps, l.pp("pre_feedforward_layernorm"))?,
                    post_ffn_norm: Norm::load(h, cfg.eps, l.pp("post_feedforward_layernorm"))?,
                    gate: linear(h, cfg.intermediate_size, l.pp("mlp.gate_proj"))?,
                    up: linear(h, cfg.intermediate_size, l.pp("mlp.up_proj"))?,
                    down: linear(cfg.intermediate_size, h, l.pp("mlp.down_proj"))?,
                    global: cfg.global[i],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Gemma3 {
            differentiable: false,
            norm: Norm::load(h, cfg.eps, vb.pp("norm"))?,
            head: Linear::new(embed.embeddings().clone(), None),
            local_rope: Rope::new(cfg.rope_local_base, 1.0, hd, MAX_POSITIONS, device)?,
            global_rope: Rope::new(cfg.rope_theta, cfg.rope_factor, hd, MAX_POSITIONS, device)?,
            embed,
            layers,
            cfg,
        })
    }

    pub fn empty_state(&self) -> Kv {
        Kv::new(self.layers.len())
    }

    pub fn set_differentiable(&mut self, on: bool) {
        self.differentiable = on;
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// The attention-state shape a bridge maps into. `rope_theta_bits` names the GLOBAL base; the
    /// local layers rotate with `rope_local_base` — `rotate_keys` takes the layer for that reason.
    pub fn geometry(&self) -> Geometry {
        Geometry { layers: self.cfg.layers, kv_heads: self.cfg.kv_heads, head_dim: self.cfg.head_dim, rope_theta_bits: self.cfg.rope_theta.to_bits() }
    }

    /// Rotate (or, `inverse`, de-rotate) layer `layer`'s keys `[batch, kv_heads, n, head_dim]` that
    /// start at absolute position `pos`, with that layer's RoPE.
    pub fn rotate_keys(&self, layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        let rope = if self.cfg.global[layer] { &self.global_rope } else { &self.local_rope };
        rope.apply(k, pos, if inverse { -1.0 } else { 1.0 }, false)
    }

    /// Run `ids` (`[batch, n]`) from absolute position `pos`, extending `kv`; logits of the last
    /// position, `[batch, vocab]`.
    pub fn forward(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        self.forward_capture(ids, pos, kv, None)
    }

    /// Logits at EVERY position of `ids`, `[batch, n, vocab]` — what a distribution-matching loss needs.
    pub fn forward_all(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        let x = self.run(ids, pos, kv, None)?;
        self.head.forward(&x)?.to_dtype(DType::F32)
    }

    /// As `forward`, also pushing each layer's INPUT residual stream `[batch, n, hidden]` into
    /// `capture` — the features a bridge maps from.
    pub fn forward_capture(&self, ids: &Tensor, pos: usize, kv: &mut Kv, capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let x = self.run(ids, pos, kv, capture)?;
        let n = x.dim(1)?;
        self.head.forward(&x.narrow(1, n - 1, 1)?)?.squeeze(1)?.to_dtype(DType::F32)
    }

    /// The decoder: final-normed hidden states at every position, `[batch, n, hidden]`.
    fn run(&self, ids: &Tensor, pos: usize, kv: &mut Kv, mut capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let (b, n) = ids.dims2()?;
        let (c, diff) = (&self.cfg, self.differentiable);
        let mut x = (self.embed.forward(ids)? * (c.hidden_size as f64).sqrt())?;
        let mut slots = kv.layers().to_vec();
        for (layer, slot) in self.layers.iter().zip(slots.iter_mut()) {
            if let Some(cap) = capture.as_deref_mut() {
                cap.push(x.clone());
            }
            let r = x.clone();
            let y = layer.input_norm.forward(&x, diff)?;
            let split = |t: Tensor, heads: usize| t.reshape((b, n, heads, c.head_dim))?.transpose(1, 2)?.contiguous();
            let rope = if layer.global { &self.global_rope } else { &self.local_rope };
            let q = rope.apply(&layer.q_norm.forward(&split(layer.q.forward(&y)?, c.heads)?, diff)?, pos, 1.0, diff)?;
            let mut k = rope.apply(&layer.k_norm.forward(&split(layer.k.forward(&y)?, c.kv_heads)?, diff)?, pos, 1.0, diff)?;
            let mut v = split(layer.v.forward(&y)?, c.kv_heads)?;
            if let Some((pk, pv)) = slot.as_ref() {
                k = Tensor::cat(&[pk, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[pv, &v], 2)?.contiguous()?;
            }
            *slot = Some((k.clone(), v.clone()));
            let group = c.heads / c.kv_heads;
            let k = candle_transformers::utils::repeat_kv(k, group)?;
            let v = candle_transformers::utils::repeat_kv(v, group)?;
            let att = (q.matmul(&k.t()?)? * c.query_pre_attn_scalar.powf(-0.5))?;
            let att = mask(&att, n, pos, if layer.global { None } else { Some(c.sliding_window) })?;
            let p = if diff { candle_nn::ops::softmax(&att, D::Minus1)? } else { candle_nn::ops::softmax_last_dim(&att)? };
            let a = p.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((b, n, c.heads * c.head_dim))?;
            x = (r + layer.post_attention_norm.forward(&layer.o.forward(&a)?, diff)?)?;
            let r = x.clone();
            let y = layer.pre_ffn_norm.forward(&x, diff)?;
            let mlp = layer.down.forward(&(layer.gate.forward(&y)?.gelu()? * layer.up.forward(&y)?)?)?;
            x = (r + layer.post_ffn_norm.forward(&mlp, diff)?)?;
        }
        *kv = Kv::from_layers(slots);
        self.norm.forward(&x, diff)
    }
}

/// Causal mask, and for a sliding-window layer keys no older than `window` positions: query row i (at
/// absolute position pos+i) sees key j when j ≤ pos+i and pos+i − j < window.
fn mask(att: &Tensor, n: usize, pos: usize, window: Option<usize>) -> Result<Tensor> {
    let total = att.dim(D::Minus1)?;
    let hidden: Vec<u8> = (0..n)
        .flat_map(|i| (0..total).map(move |j| u8::from(j > pos + i || window.is_some_and(|w| pos + i - j.min(pos + i) >= w))))
        .collect();
    if !hidden.contains(&1) {
        return Ok(att.clone());
    }
    let m = Tensor::from_slice(&hidden, (n, total), att.device())?.broadcast_as(att.shape())?;
    let neg = Tensor::new(f32::NEG_INFINITY, att.device())?.broadcast_as(att.shape())?;
    m.where_cond(&neg, att)
}
