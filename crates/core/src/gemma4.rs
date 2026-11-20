//! Gemma 4's text decoder with its attention state OPEN — BROADEN_SUPPORT's Gemma 4 loader. Follows
//! transformers' `Gemma4TextModel`; `tests/hf_reference.rs` pins the arithmetic to transformers. The
//! vision and audio towers are ignored.
//!
//! What differs from Gemma 3: RMS norms scale by `w`, not `1 + w`; values are RMS-normed too (no scale);
//! scores are not scaled (the q/k norms carry it); GLOBAL layers have heads of their own width
//! (`global_head_dim`), their own KV-head count when `attention_k_eq_v` (and then V is K's projection),
//! and "proportional" RoPE, which rotates only the first quarter of each head's frequency pairs; every
//! layer ends by scaling the residual by its `layer_scalar`; the logits are soft-capped.
//!
//! The small models (E2B) add PER-LAYER INPUTS — a second, per-layer embedding of each token, mixed with
//! a projection of the main embedding, gated into the residual after every MLP — and KV SHARING: the
//! last `num_kv_shared_layers` layers compute no keys or values, and read those of the last unshared
//! layer of their own kind. A shared layer's slot in `Kv` stays empty. Those layers' MLPs may be twice
//! as wide (`use_double_wide_mlp`).

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::half::{Embed, Lin, embedding, linear, small};
use crate::llama::{Geometry, Kv};

/// One layer's shape.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub global: bool,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    /// For a KV-shared layer, the layer whose keys and values it reads.
    pub kv_from: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub heads: usize,
    pub vocab_size: usize,
    pub eps: f64,
    pub sliding_window: usize,
    pub softcap: Option<f64>,
    /// V is K's projection on the global layers (no `v_proj` there).
    pub k_eq_v: bool,
    pub local_theta: f32,
    pub global_theta: f32,
    /// The fraction of a global head's frequency pairs that rotate.
    pub global_rotary: f64,
    /// Per-layer input width (0: none).
    pub ple: usize,
    pub layers: Vec<LayerSpec>,
}

impl Config {
    pub fn from_json(config_json: &[u8]) -> std::result::Result<Config, String> {
        let root: serde_json::Value = serde_json::from_slice(config_json).map_err(|e| e.to_string())?;
        let t = if root["text_config"].is_object() { &root["text_config"] } else { &root };
        let u = |k: &str, d: usize| t[k].as_u64().map_or(d, |v| v as usize);
        let f = |k: &str, d: f64| t[k].as_f64().unwrap_or(d);
        let n = u("num_hidden_layers", 35);
        let types: Vec<bool> = match t["layer_types"].as_array() {
            Some(types) => types.iter().map(|v| v == "full_attention").collect(),
            None => return Err("Gemma 4: config has no layer_types".into()),
        };
        if types.len() != n {
            return Err(format!("Gemma 4: {} layer_types for {n} layers", types.len()));
        }
        let k_eq_v = t["attention_k_eq_v"].as_bool().unwrap_or(false);
        let kv_heads = u("num_key_value_heads", 1);
        let global_kv = if k_eq_v { t["num_global_key_value_heads"].as_u64().map_or(kv_heads, |v| v as usize) } else { kv_heads };
        let (head_dim, global_dim) = (u("head_dim", 256), u("global_head_dim", 512));
        let intermediate = u("intermediate_size", 6144);
        let shared = u("num_kv_shared_layers", 0);
        let first_shared = n - shared;
        let double = t["use_double_wide_mlp"].as_bool().unwrap_or(false);
        let layers = (0..n)
            .map(|i| {
                let global = types[i];
                let is_shared = shared > 0 && i >= first_shared;
                let kv_from = is_shared.then(|| (0..first_shared).rev().find(|&j| types[j] == global)).flatten();
                if is_shared && kv_from.is_none() {
                    return Err(format!("Gemma 4: shared layer {i} has no earlier layer of its kind"));
                }
                Ok(LayerSpec {
                    global,
                    kv_heads: if global { global_kv } else { kv_heads },
                    head_dim: if global { global_dim } else { head_dim },
                    intermediate: intermediate * if double && is_shared { 2 } else { 1 },
                    kv_from,
                })
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;
        let rope = &t["rope_parameters"];
        let global_type = rope["full_attention"]["rope_type"].as_str().unwrap_or("default");
        let global_rotary = match global_type {
            "proportional" => rope["full_attention"]["partial_rotary_factor"].as_f64().unwrap_or(1.0),
            "default" => 1.0,
            other => return Err(format!("Gemma 4 RoPE type {other:?} is not supported")),
        };
        if rope["full_attention"]["factor"].as_f64().is_some_and(|x| x != 1.0) {
            return Err("Gemma 4: scaled RoPE is not supported".into());
        }
        if t["enable_moe_block"].as_bool().unwrap_or(false) {
            return Err("Gemma 4: the mixture-of-experts block (26B-A4B) is not supported yet".into());
        }
        Ok(Config {
            hidden_size: u("hidden_size", 1536),
            heads: u("num_attention_heads", 8),
            vocab_size: u("vocab_size", 262_144),
            eps: f("rms_norm_eps", 1e-6),
            sliding_window: u("sliding_window", 512),
            softcap: t["final_logit_softcapping"].as_f64(),
            k_eq_v,
            local_theta: rope["sliding_attention"]["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
            global_theta: rope["full_attention"]["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            global_rotary,
            ple: u("hidden_size_per_layer_input", 0),
            layers,
        })
    }
}

/// RMS norm scaling by `w` (or by nothing: a value norm).
struct Norm {
    weight: Tensor,
    eps: f32,
}

impl Norm {
    fn load(size: usize, eps: f64, vb: &VarBuilder) -> Result<Self> {
        Ok(Self { weight: small(vb, size, "weight")?, eps: eps as f32 })
    }
    fn unscaled(size: usize, eps: f64, device: &Device) -> Result<Self> {
        Ok(Self { weight: Tensor::ones(size, DType::F32, device)?, eps: eps as f32 })
    }
    fn forward(&self, x: &Tensor, diff: bool) -> Result<Tensor> {
        if diff { candle_nn::ops::rms_norm_slow(x, &self.weight, self.eps) } else { candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps) }
    }
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    /// The first `rotated` of `head_dim / 2` frequency pairs rotate, at exponents over the WHOLE head
    /// width; the rest have frequency 0 and pass unrotated (transformers' "proportional" RoPE).
    fn new(base: f32, head_dim: usize, rotated: usize, max: usize, device: &Device) -> Result<Self> {
        let inv: Vec<f32> = (0..head_dim / 2).map(|i| if i < rotated { 1.0 / base.powf((2 * i) as f32 / head_dim as f32) } else { 0.0 }).collect();
        let inv = Tensor::new(inv.as_slice(), device)?;
        let t = Tensor::arange(0, max as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.reshape((max, 1))?.matmul(&inv.reshape((1, inv.elem_count()))?)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }
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

/// A layer's own key and value projections — absent on a KV-shared layer.
struct KvProj {
    k: Lin,
    /// None when V is K's projection.
    v: Option<Lin>,
    k_norm: Norm,
}

/// A layer's per-layer-input gate.
struct PerLayer {
    gate: Lin,
    projection: Lin,
    norm: Norm,
}

struct Layer {
    q: Lin,
    kv: Option<KvProj>,
    o: Lin,
    q_norm: Norm,
    v_norm: Norm,
    input_norm: Norm,
    post_attention_norm: Norm,
    pre_ffn_norm: Norm,
    post_ffn_norm: Norm,
    gate: Lin,
    up: Lin,
    down: Lin,
    per_layer: Option<PerLayer>,
    scalar: Tensor,
}

/// The model-level half of the per-layer inputs.
struct PerLayerInputs {
    embed: Embed,
    projection: Lin,
    norm: Norm,
}

pub struct Gemma4 {
    differentiable: bool,
    cfg: Config,
    embed: Embed,
    per_layer: Option<PerLayerInputs>,
    layers: Vec<Layer>,
    norm: Norm,
    head: Lin,
    local_rope: Rope,
    global_rope: Rope,
}

const MAX_POSITIONS: usize = 8192;

impl Gemma4 {
    pub fn load(config_json: &[u8], weights: impl Into<crate::weights::Weights>, device: &Device) -> std::result::Result<Gemma4, String> {
        Self::load_in(config_json, weights, device, DType::F32)
    }

    /// As `load`, its weights held in `dtype` (`crate::half`).
    pub fn load_in(config_json: &[u8], weights: impl Into<crate::weights::Weights>, device: &Device, dtype: DType) -> std::result::Result<Gemma4, String> {
        let vb = weights.into().builder(dtype, device)?;
        Self::from_builder(config_json, vb)
    }

    /// From a VarBuilder already made — `model::load_decoder_q` builds under `crate::half::quantizing`.
    /// A text-only checkpoint (`model.*`) or a multimodal one (`model.language_model.*`).
    pub fn from_builder(config_json: &[u8], vb: VarBuilder) -> std::result::Result<Gemma4, String> {
        let cfg = Config::from_json(config_json)?;
        let prefix = if vb.contains_tensor("model.language_model.embed_tokens.weight") { "model.language_model" } else { "model" };
        Self::build(cfg, vb.pp(prefix)).map_err(|e| e.to_string())
    }

    fn build(cfg: Config, vb: VarBuilder) -> Result<Gemma4> {
        let (h, eps, device) = (cfg.hidden_size, cfg.eps, vb.device().clone());
        let n = cfg.layers.len();
        let embed = embedding(cfg.vocab_size, h, vb.pp("embed_tokens"))?;
        let per_layer = if cfg.ple > 0 {
            Some(PerLayerInputs {
                embed: embedding(cfg.vocab_size, n * cfg.ple, vb.pp("embed_tokens_per_layer"))?,
                projection: linear(h, n * cfg.ple, vb.pp("per_layer_model_projection"))?,
                norm: Norm::load(cfg.ple, eps, &vb.pp("per_layer_projection_norm"))?,
            })
        } else {
            None
        };
        let mut layers = Vec::with_capacity(n);
        for (i, spec) in cfg.layers.iter().enumerate() {
            let l = vb.pp(format!("layers.{i}"));
            let a = l.pp("self_attn");
            let hd = spec.head_dim;
            let kv = if spec.kv_from.is_none() {
                let k_eq_v = cfg.k_eq_v && spec.global;
                Some(KvProj {
                    k: linear(h, spec.kv_heads * hd, a.pp("k_proj"))?,
                    v: if k_eq_v { None } else { Some(linear(h, spec.kv_heads * hd, a.pp("v_proj"))?) },
                    k_norm: Norm::load(hd, eps, &a.pp("k_norm"))?,
                })
            } else {
                None
            };
            let per = if cfg.ple > 0 {
                Some(PerLayer {
                    gate: linear(h, cfg.ple, l.pp("per_layer_input_gate"))?,
                    projection: linear(cfg.ple, h, l.pp("per_layer_projection"))?,
                    norm: Norm::load(h, eps, &l.pp("post_per_layer_input_norm"))?,
                })
            } else {
                None
            };
            layers.push(Layer {
                q: linear(h, cfg.heads * hd, a.pp("q_proj"))?,
                kv,
                o: linear(cfg.heads * hd, h, a.pp("o_proj"))?,
                q_norm: Norm::load(hd, eps, &a.pp("q_norm"))?,
                v_norm: Norm::unscaled(hd, eps, &device)?,
                input_norm: Norm::load(h, eps, &l.pp("input_layernorm"))?,
                post_attention_norm: Norm::load(h, eps, &l.pp("post_attention_layernorm"))?,
                pre_ffn_norm: Norm::load(h, eps, &l.pp("pre_feedforward_layernorm"))?,
                post_ffn_norm: Norm::load(h, eps, &l.pp("post_feedforward_layernorm"))?,
                gate: linear(h, spec.intermediate, l.pp("mlp.gate_proj"))?,
                up: linear(h, spec.intermediate, l.pp("mlp.up_proj"))?,
                down: linear(spec.intermediate, h, l.pp("mlp.down_proj"))?,
                per_layer: per,
                scalar: small(&l, 1, "layer_scalar")?,
            });
        }
        let global_dim = cfg.layers.iter().find(|s| s.global).map_or(512, |s| s.head_dim);
        let local_dim = cfg.layers.iter().find(|s| !s.global).map_or(256, |s| s.head_dim);
        let rotated = (cfg.global_rotary * global_dim as f64 / 2.0) as usize;
        Ok(Gemma4 {
            differentiable: false,
            norm: Norm::load(h, eps, &vb.pp("norm"))?,
            head: Lin::new(embed.embeddings().clone(), None),
            local_rope: Rope::new(cfg.local_theta, local_dim, local_dim / 2, MAX_POSITIONS, &device)?,
            global_rope: Rope::new(cfg.global_theta, global_dim, rotated, MAX_POSITIONS, &device)?,
            embed,
            per_layer,
            layers,
            cfg,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
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

    /// The SLIDING layers' attention shape (most layers); a global layer's is `layer_shape`.
    /// `rope_theta_bits` names the global base.
    pub fn geometry(&self) -> Geometry {
        let local = self.cfg.layers.iter().find(|s| !s.global).unwrap_or(&self.cfg.layers[0]);
        Geometry { layers: self.cfg.layers.len(), kv_heads: local.kv_heads, head_dim: local.head_dim, rope_theta_bits: self.cfg.global_theta.to_bits() }
    }

    /// Layer `layer`'s (kv heads, head width).
    pub fn layer_shape(&self, layer: usize) -> (usize, usize) {
        let s = &self.cfg.layers[layer];
        (s.kv_heads, s.head_dim)
    }

    pub fn rotate_keys(&self, layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        let rope = if self.cfg.layers[layer].global { &self.global_rope } else { &self.local_rope };
        rope.apply(k, pos, if inverse { -1.0 } else { 1.0 }, false)
    }

    pub fn forward(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        self.forward_capture(ids, pos, kv, None)
    }

    pub fn forward_all(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        let x = self.run(ids, pos, kv, None)?;
        self.logits(&x)
    }

    /// Logits from residual rows `[.., hidden]`: the final norm, the head, and the family's own scaling —
    /// what `forward_capture` does to the last layer's output (a logit lens reads any layer's).
    pub fn unembed(&self, x: &Tensor) -> Result<Tensor> {
        self.logits(&self.norm.forward(x, self.differentiable)?)
    }

    pub fn forward_capture(&self, ids: &Tensor, pos: usize, kv: &mut Kv, capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let x = self.run(ids, pos, kv, capture)?;
        let n = x.dim(1)?;
        self.logits(&x.narrow(1, n - 1, 1)?)?.squeeze(1)
    }

    fn logits(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.head.forward(x)?.to_dtype(DType::F32)?;
        match self.cfg.softcap {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }
    }

    /// Per-layer inputs `[batch, n, layers, ple]` from the token ids and the scaled main embedding.
    fn per_layer_inputs(&self, p: &PerLayerInputs, ids: &Tensor, x: &Tensor) -> Result<Tensor> {
        let (b, n) = ids.dims2()?;
        let (layers, ple) = (self.cfg.layers.len(), self.cfg.ple);
        let tokens = (p.embed.forward(ids)? * (ple as f64).sqrt())?.reshape((b, n, layers, ple))?;
        let projected = (p.projection.forward(x)? * (self.cfg.hidden_size as f64).powf(-0.5))?.reshape((b, n, layers, ple))?;
        let projected = p.norm.forward(&projected, self.differentiable)?;
        (projected + tokens)? * 0.5f64.sqrt()
    }

    /// The decoder: final-normed hidden states at every position, `[batch, n, hidden]`.
    fn run(&self, ids: &Tensor, pos: usize, kv: &mut Kv, mut capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let (b, n) = ids.dims2()?;
        let (c, diff) = (&self.cfg, self.differentiable);
        let mut x = (self.embed.forward(ids)? * (c.hidden_size as f64).sqrt())?;
        let ple = self.per_layer.as_ref().map(|p| self.per_layer_inputs(p, ids, &x)).transpose()?;
        let mut slots = kv.layers().to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cap) = capture.as_deref_mut() {
                cap.push(x.clone());
            }
            let spec = &c.layers[i];
            let hd = spec.head_dim;
            let rope = if spec.global { &self.global_rope } else { &self.local_rope };
            let r = x.clone();
            let y = layer.input_norm.forward(&x, diff)?;
            let split = |t: Tensor, heads: usize| t.reshape((b, n, heads, hd))?.transpose(1, 2)?.contiguous();
            let q = rope.apply(&layer.q_norm.forward(&split(layer.q.forward(&y)?, c.heads)?, diff)?, pos, 1.0, diff)?;
            let (k, v) = match (&layer.kv, spec.kv_from) {
                (Some(p), _) => {
                    let raw_k = p.k.forward(&y)?;
                    let raw_v = match &p.v {
                        Some(v) => v.forward(&y)?,
                        None => raw_k.clone(),
                    };
                    let mut k = rope.apply(&p.k_norm.forward(&split(raw_k, spec.kv_heads)?, diff)?, pos, 1.0, diff)?;
                    let mut v = layer.v_norm.forward(&split(raw_v, spec.kv_heads)?, diff)?;
                    if let Some((pk, pv)) = slots[i].as_ref() {
                        k = Tensor::cat(&[pk, &k], 2)?.contiguous()?;
                        v = Tensor::cat(&[pv, &v], 2)?.contiguous()?;
                    }
                    slots[i] = Some((k.clone(), v.clone()));
                    (k, v)
                }
                (None, Some(from)) => slots[from].clone().ok_or_else(|| candle_core::Error::Msg(format!("layer {i}: layer {from}'s keys are not there yet")))?,
                (None, None) => candle_core::bail!("layer {i}: no keys and no layer to share them from"),
            };
            let window = if spec.global { None } else { Some(c.sliding_window) };
            let a = crate::attend::attend(&q, k, v, pos, 1.0, window, diff)?;
            x = (r + layer.post_attention_norm.forward(&layer.o.forward(&a)?, diff)?)?;
            let r = x.clone();
            let y = layer.pre_ffn_norm.forward(&x, diff)?;
            let mlp = layer.down.forward(&(layer.gate.forward(&y)?.gelu()? * layer.up.forward(&y)?)?)?;
            x = (r + layer.post_ffn_norm.forward(&mlp, diff)?)?;
            if let (Some(per), Some(ple)) = (&layer.per_layer, &ple) {
                let input = ple.narrow(2, i, 1)?.squeeze(2)?;
                let g = (per.gate.forward(&x)?.gelu()? * input)?;
                x = (&x + per.norm.forward(&per.projection.forward(&g)?, diff)?)?;
            }
            x = x.broadcast_mul(&layer.scalar)?;
        }
        *kv = Kv::from_layers(slots);
        self.norm.forward(&x, diff)
    }
}
