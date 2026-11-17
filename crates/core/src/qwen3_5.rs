//! Qwen3.5's text decoder from safetensors, its persistent state OPEN — BROADEN_SUPPORT's Qwen3.5
//! loader, for pair C (0.8B → 2B) and the paper's 4B → 9B. Follows transformers'
//! `modeling_qwen3_5.py` (its reference torch paths); `tests/hf_reference.rs` pins it. f32, forward
//! only. (`qwen35.rs` is the GGUF instrument checked against upstream's oracle; this is the
//! Hugging Face checkpoint.)
//!
//! Every `full_attention_interval`-th layer is gated attention: q and its gate from one projection
//! (per head, query half then gate half), RMS(1 + w) q/k norms, RoPE over the first
//! `partial_rotary_factor` of each head (rotate-half), output × σ(gate). The rest are Gated
//! DeltaNet: a causal depthwise conv (SiLU) over [q | k | v], L2-normalised q and k, q ÷ √d_k,
//! β = σ(b), decay = exp(−exp(A_log)·softplus(a + dt_bias)), the delta rule per value head, then
//! RMS(o)·w · SiLU(z) and the output projection. Norms elsewhere scale by 1 + w; the head is tied.
//!
//! STATE, in the open `Kv`: an attention layer's slot is (K, V) `[1, kv_heads, n, head_dim]`, keys
//! rotated; a GDN layer's slot is (C, R) — C the last `conv − 1` pre-conv rows `[1, channels,
//! conv − 1]`, R the recurrent state `[1, v_heads, d_k, d_v]` — what a bridge copies across.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use crate::half::{Embed, Lin, embedding, linear};
use candle_nn::VarBuilder;

use crate::llama::{Geometry, Kv};

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot: usize,
    pub rope_theta: f32,
    pub eps: f64,
    pub vocab_size: usize,
    /// Per layer: true for gated attention, false for Gated DeltaNet.
    pub attention: Vec<bool>,
    pub k_heads: usize,
    pub v_heads: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub conv: usize,
}

impl Config {
    pub fn from_json(config_json: &[u8]) -> std::result::Result<Config, String> {
        let root: serde_json::Value = serde_json::from_slice(config_json).map_err(|e| e.to_string())?;
        let t = if root["text_config"].is_object() { &root["text_config"] } else { &root };
        let u = |k: &str| t[k].as_u64().map(|v| v as usize).ok_or(format!("text_config has no {k}"));
        let rope = if t["rope_parameters"].is_object() { &t["rope_parameters"] } else { t };
        let head_dim = u("head_dim")?;
        let layers = u("num_hidden_layers")?;
        let attention = match t["layer_types"].as_array() {
            Some(types) => types.iter().map(|v| v == "full_attention").collect(),
            None => {
                let every = u("full_attention_interval")?;
                (0..layers).map(|i| (i + 1) % every == 0).collect()
            }
        };
        Ok(Config {
            hidden_size: u("hidden_size")?,
            intermediate_size: u("intermediate_size")?,
            layers,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            rot: (head_dim as f64 * rope["partial_rotary_factor"].as_f64().unwrap_or(0.25)) as usize,
            rope_theta: rope["rope_theta"].as_f64().unwrap_or(10_000_000.0) as f32,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            vocab_size: u("vocab_size")?,
            attention,
            k_heads: u("linear_num_key_heads")?,
            v_heads: u("linear_num_value_heads")?,
            d_k: u("linear_key_head_dim")?,
            d_v: u("linear_value_head_dim")?,
            conv: u("linear_conv_kernel_dim")?,
        })
    }
    fn channels(&self) -> usize {
        2 * self.k_heads * self.d_k + self.v_heads * self.d_v
    }
}

/// RMS norm; `offset` 1.0 for Qwen3.5's zero-centred norms (× (1 + w)), 0.0 for the gated norm (× w).
struct Norm {
    weight: Tensor,
    eps: f32,
}

impl Norm {
    fn load(size: usize, eps: f64, offset: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { weight: (vb.get(size, "weight")?.to_dtype(DType::F32)? + offset)?, eps: eps as f32 })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps)
    }
    /// From basic ops, so gradients flow (the fused kernel has none).
    fn forward_diff(&self, x: &Tensor, diff: bool) -> Result<Tensor> {
        if diff { candle_nn::ops::rms_norm_slow(x, &self.weight, self.eps) } else { self.forward(x) }
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

struct Gdn {
    qkv: Lin,
    z: Lin,
    a: Lin,
    b: Lin,
    /// `[channels, conv]`
    conv: Tensor,
    /// −exp(A_log), `[v_heads]`
    neg_a: Tensor,
    dt_bias: Tensor,
    norm: Norm,
    out: Lin,
}

enum Mixer {
    Attention(Attention),
    Gdn(Gdn),
}

struct Layer {
    mixer: Mixer,
    input_norm: Norm,
    post_norm: Norm,
    gate: Lin,
    up: Lin,
    down: Lin,
}

pub struct Qwen35 {
    /// Norms, softmax and RoPE from basic ops so gradients flow; inference leaves it off.
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

impl Qwen35 {
    /// From `config.json` and ONE safetensors buffer — `model.language_model.*` (the vision tower and
    /// the multi-token-prediction layer ignored) or `model.*`.
    pub fn load(config_json: &[u8], weights: Vec<u8>, device: &Device) -> std::result::Result<Qwen35, String> {
        Self::load_in(config_json, weights, device, DType::F32)
    }

    /// As `load`, its weights held in `dtype` (`crate::half`).
    pub fn load_in(config_json: &[u8], weights: Vec<u8>, device: &Device, dtype: DType) -> std::result::Result<Qwen35, String> {
        let cfg = Config::from_json(config_json)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, dtype, device).map_err(|e| e.to_string())?;
        let prefix = if vb.contains_tensor("model.language_model.embed_tokens.weight") { "model.language_model" } else { "model" };
        // An untied output head (the 9B) is its own `lm_head`; otherwise it is the embeddings.
        let head = if vb.contains_tensor("lm_head.weight") {
            Some(linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head")).map_err(|e| e.to_string())?)
        } else {
            None
        };
        Self::build(cfg, vb.pp(prefix), head, device).map_err(|e| e.to_string())
    }

    fn build(cfg: Config, vb: VarBuilder, head: Option<Lin>, device: &Device) -> Result<Qwen35> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let embed = embedding(cfg.vocab_size, h, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = vb.pp(format!("layers.{i}"));
            let mixer = if cfg.attention[i] {
                let a = l.pp("self_attn");
                Mixer::Attention(Attention {
                    q: linear(h, cfg.heads * hd * 2, a.pp("q_proj"))?,
                    k: linear(h, cfg.kv_heads * hd, a.pp("k_proj"))?,
                    v: linear(h, cfg.kv_heads * hd, a.pp("v_proj"))?,
                    o: linear(cfg.heads * hd, h, a.pp("o_proj"))?,
                    q_norm: Norm::load(hd, cfg.eps, 1.0, a.pp("q_norm"))?,
                    k_norm: Norm::load(hd, cfg.eps, 1.0, a.pp("k_norm"))?,
                })
            } else {
                let g = l.pp("linear_attn");
                let ch = cfg.channels();
                Mixer::Gdn(Gdn {
                    qkv: linear(h, ch, g.pp("in_proj_qkv"))?,
                    z: linear(h, cfg.v_heads * cfg.d_v, g.pp("in_proj_z"))?,
                    a: linear(h, cfg.v_heads, g.pp("in_proj_a"))?,
                    b: linear(h, cfg.v_heads, g.pp("in_proj_b"))?,
                    conv: g.get((ch, 1, cfg.conv), "conv1d.weight")?.to_dtype(DType::F32)?.squeeze(1)?,
                    neg_a: g.get(cfg.v_heads, "A_log")?.to_dtype(DType::F32)?.exp()?.neg()?,
                    dt_bias: g.get(cfg.v_heads, "dt_bias")?.to_dtype(DType::F32)?,
                    norm: Norm::load(cfg.d_v, cfg.eps, 0.0, g.pp("norm"))?,
                    out: linear(cfg.v_heads * cfg.d_v, h, g.pp("out_proj"))?,
                })
            };
            layers.push(Layer {
                mixer,
                input_norm: Norm::load(h, cfg.eps, 1.0, l.pp("input_layernorm"))?,
                post_norm: Norm::load(h, cfg.eps, 1.0, l.pp("post_attention_layernorm"))?,
                gate: linear(h, cfg.intermediate_size, l.pp("mlp.gate_proj"))?,
                up: linear(h, cfg.intermediate_size, l.pp("mlp.up_proj"))?,
                down: linear(cfg.intermediate_size, h, l.pp("mlp.down_proj"))?,
            });
        }
        let inv: Vec<f32> = (0..cfg.rot).step_by(2).map(|i| 1.0 / cfg.rope_theta.powf(i as f32 / cfg.rot as f32)).collect();
        let inv = Tensor::new(inv.as_slice(), device)?;
        let t = Tensor::arange(0, MAX_POSITIONS as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.reshape((MAX_POSITIONS, 1))?.matmul(&inv.reshape((1, inv.elem_count()))?)?;
        Ok(Qwen35 {
            differentiable: false,
            norm: Norm::load(h, cfg.eps, 1.0, vb.pp("norm"))?,
            head: match head {
                Some(head) => head,
                None => Lin::new(embed.embeddings().clone(), None),
            },
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

    /// The attention layers' shape (the GDN layers' state is copied, not mapped).
    pub fn geometry(&self) -> Geometry {
        Geometry { layers: self.cfg.layers, kv_heads: self.cfg.kv_heads, head_dim: self.cfg.head_dim, rope_theta_bits: self.cfg.rope_theta.to_bits() }
    }

    /// RoPE on the first `rot` dims of `[b, heads, n, head_dim]` from position `pos`; `sign` −1 inverts.
    fn rope(&self, x: &Tensor, pos: usize, sign: f64) -> Result<Tensor> {
        let (n, rot) = (x.dim(2)?, self.cfg.rot);
        let (cos, sin) = (self.cos.narrow(0, pos, n)?, (self.sin.narrow(0, pos, n)? * sign)?);
        let a = x.narrow(3, 0, rot)?.contiguous()?;
        let rotated = if self.differentiable {
            let half = rot / 2;
            let turned = Tensor::cat(&[&a.narrow(3, half, half)?.neg()?, &a.narrow(3, 0, half)?], 3)?;
            let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?.unsqueeze(0)?;
            let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?.unsqueeze(0)?;
            (a.broadcast_mul(&cos)? + turned.broadcast_mul(&sin)?)?
        } else {
            candle_nn::rotary_emb::rope(&a, &cos, &sin)?
        };
        Tensor::cat(&[&rotated, &x.narrow(3, rot, self.cfg.head_dim - rot)?], 3)
    }

    /// Rotate (or, `inverse`, de-rotate) keys of an attention layer from absolute position `pos`.
    pub fn rotate_keys(&self, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        self.rope(k, pos, if inverse { -1.0 } else { 1.0 })
    }

    pub fn forward(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        self.forward_capture(ids, pos, kv, None)
    }

    pub fn forward_all(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        let x = self.run(ids, pos, kv, None)?;
        self.head.forward(&x)?.to_dtype(DType::F32)
    }

    pub fn forward_capture(&self, ids: &Tensor, pos: usize, kv: &mut Kv, capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let x = self.run(ids, pos, kv, capture)?;
        let n = x.dim(1)?;
        self.head.forward(&x.narrow(1, n - 1, 1)?)?.squeeze(1)?.to_dtype(DType::F32)
    }

    fn run(&self, ids: &Tensor, pos: usize, kv: &mut Kv, mut capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let (b, n) = ids.dims2()?;
        let c = &self.cfg;
        let mut x = self.embed.forward(ids)?;
        let mut slots = kv.layers().to_vec();
        for (layer, slot) in self.layers.iter().zip(slots.iter_mut()) {
            if let Some(cap) = capture.as_deref_mut() {
                cap.push(x.clone());
            }
            let y = layer.input_norm.forward_diff(&x, self.differentiable)?;
            let mixed = match &layer.mixer {
                Mixer::Attention(a) => self.attention(a, &y, pos, slot, b, n)?,
                Mixer::Gdn(g) => self.gdn(g, &y, slot, b, n)?,
            };
            x = (x + mixed)?;
            let y = layer.post_norm.forward_diff(&x, self.differentiable)?;
            let mlp = layer.down.forward(&(candle_nn::ops::silu(&layer.gate.forward(&y)?)? * layer.up.forward(&y)?)?)?;
            x = (x + mlp)?;
        }
        *kv = Kv::from_layers(slots);
        let _ = c;
        self.norm.forward_diff(&x, self.differentiable)
    }

    fn attention(&self, a: &Attention, y: &Tensor, pos: usize, slot: &mut Option<(Tensor, Tensor)>, b: usize, n: usize) -> Result<Tensor> {
        let c = &self.cfg;
        let hd = c.head_dim;
        let qg = a.q.forward(y)?.reshape((b, n, c.heads, hd * 2))?;
        let q = qg.narrow(3, 0, hd)?;
        let gate = qg.narrow(3, hd, hd)?.reshape((b, n, c.heads * hd))?;
        let diff = self.differentiable;
        let q = self.rope(&a.q_norm.forward_diff(&q, diff)?.transpose(1, 2)?.contiguous()?, pos, 1.0)?;
        let k = a.k_norm.forward_diff(&a.k.forward(y)?.reshape((b, n, c.kv_heads, hd))?, diff)?.transpose(1, 2)?.contiguous()?;
        let mut k = self.rope(&k, pos, 1.0)?;
        let mut v = a.v.forward(y)?.reshape((b, n, c.kv_heads, hd))?.transpose(1, 2)?.contiguous()?;
        if let Some((pk, pv)) = slot.as_ref() {
            k = Tensor::cat(&[pk, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[pv, &v], 2)?.contiguous()?;
        }
        *slot = Some((k.clone(), v.clone()));
        let group = c.heads / c.kv_heads;
        let k = candle_transformers::utils::repeat_kv(k, group)?;
        let v = candle_transformers::utils::repeat_kv(v, group)?;
        // contiguous: at one token RoPE hands back a strided query, which CUDA's matmul refuses
        let att = (q.contiguous()?.matmul(&k.t()?)? / (hd as f64).sqrt())?;
        let att = if n == 1 { att } else { causal(&att, n, pos)? };
        let p = if diff { candle_nn::ops::softmax(&att, D::Minus1)? } else { candle_nn::ops::softmax_last_dim(&att)? };
        crate::observe::attention(&p)?;
        let o = p.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((b, n, c.heads * hd))?;
        a.o.forward(&(o * candle_nn::ops::sigmoid(&gate)?)?)
    }

    fn gdn(&self, g: &Gdn, y: &Tensor, slot: &mut Option<(Tensor, Tensor)>, b: usize, n: usize) -> Result<Tensor> {
        let c = &self.cfg;
        let (ch, kk, dk, dv, vh) = (c.channels(), c.k_heads, c.d_k, c.d_v, c.v_heads);
        let raw = g.qkv.forward(y)?.transpose(1, 2)?.contiguous()?; // [b, ch, n]
        let (prev_c, prev_r) = match slot.as_ref() {
            Some((pc, pr)) => (pc.clone(), Some(pr.clone())),
            None => (Tensor::zeros((b, ch, c.conv - 1), DType::F32, y.device())?, None),
        };
        let padded = Tensor::cat(&[&prev_c, &raw], 2)?; // [b, ch, conv − 1 + n]
        let new_c = padded.narrow(2, n, c.conv - 1)?.contiguous()?;
        // depthwise causal conv: out[t] = Σ_j w[j] · padded[t + j]
        let mut conv = Tensor::zeros((b, ch, n), DType::F32, y.device())?;
        for j in 0..c.conv {
            let w = g.conv.narrow(1, j, 1)?.reshape((1, ch, 1))?;
            conv = (conv + padded.narrow(2, j, n)?.broadcast_mul(&w)?)?;
        }
        let mixed = candle_nn::ops::silu(&conv)?.transpose(1, 2)?.contiguous()?; // [b, n, ch]
        let l2 = |t: Tensor| -> Result<Tensor> { t.broadcast_mul(&(t.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?) };
        let q = l2(mixed.narrow(2, 0, kk * dk)?.reshape((b, n, kk, dk))?)?;
        let k = l2(mixed.narrow(2, kk * dk, kk * dk)?.reshape((b, n, kk, dk))?)?;
        let v = mixed.narrow(2, 2 * kk * dk, vh * dv)?.reshape((b, n, vh, dv))?;
        let q = (q / (dk as f64).sqrt())?;
        let (q, k) = if vh > kk {
            let r = vh / kk;
            let rep = |t: Tensor| t.unsqueeze(3)?.repeat((1, 1, 1, r, 1))?.reshape((b, n, vh, dk));
            (rep(q)?, rep(k)?)
        } else {
            (q, k)
        };
        let beta = candle_nn::ops::sigmoid(&g.b.forward(y)?)?; // [b, n, vh]
        // the log of each step's decay
        let log_decay = softplus(&g.a.forward(y)?.broadcast_add(&g.dt_bias)?)?.broadcast_mul(&g.neg_a)?;
        let state = match prev_r {
            Some(r) => r,
            None => Tensor::zeros((b, vh, dk, dv), DType::F32, y.device())?,
        };
        let (o, state) = if n < CHUNK_FROM { delta_steps(&q, &k, &v, &log_decay, &beta, state)? } else { delta_chunks(&q, &k, &v, &log_decay, &beta, state)? };
        *slot = Some((new_c, state));
        let z = g.z.forward(y)?.reshape((b, n, vh, dv))?;
        let o = (g.norm.forward_diff(&o, self.differentiable)? * candle_nn::ops::silu(&z)?)?.reshape((b, n, vh * dv))?;
        g.out.forward(&o)
    }
}

/// From this many tokens the recurrence runs in chunks: a step is ~10 small launches, so one token at a
/// time leaves a GPU waiting on launches and a wider model costs the same as a narrow one.
const CHUNK_FROM: usize = 16;
const CHUNK: usize = 64;

/// The gated delta rule one token at a time: q, k `[b, n, h, dk]` (q scaled), v `[b, n, h, dv]`, the
/// log decay and beta `[b, n, h]`, state `[b, h, dk, dv]`. Returns the outputs `[b, n, h, dv]` and the
/// state after the last token.
fn delta_steps(q: &Tensor, k: &Tensor, v: &Tensor, log_decay: &Tensor, beta: &Tensor, mut state: Tensor) -> Result<(Tensor, Tensor)> {
    let (b, n, h, _) = q.dims4()?;
    let decay = log_decay.exp()?;
    let mut outs = Vec::with_capacity(n);
    for t in 0..n {
        let (qt, kt, vt) = (q.narrow(1, t, 1)?.squeeze(1)?, k.narrow(1, t, 1)?.squeeze(1)?, v.narrow(1, t, 1)?.squeeze(1)?);
        let dt = decay.narrow(1, t, 1)?.squeeze(1)?.reshape((b, h, 1, 1))?;
        let bt = beta.narrow(1, t, 1)?.squeeze(1)?.reshape((b, h, 1))?;
        state = state.broadcast_mul(&dt)?;
        let kv_mem = state.broadcast_mul(&kt.unsqueeze(3)?)?.sum(2)?; // [b, h, dv]
        let delta = (vt - kv_mem)?.broadcast_mul(&bt)?;
        state = (state + kt.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;
        outs.push(state.broadcast_mul(&qt.unsqueeze(3)?)?.sum(2)?); // [b, h, dv]
    }
    Ok((Tensor::stack(&outs, 1)?, state))
}

/// The same recurrence in chunks of `CHUNK` tokens — the chunked form transformers uses for Qwen3.5.
/// Within a chunk the delta rule's corrections are one triangular inverse, `(I + L)⁻¹`, built by doubling
/// diagonal blocks, and the state crosses chunk boundaries once per chunk.
fn delta_chunks(q: &Tensor, k: &Tensor, v: &Tensor, log_decay: &Tensor, beta: &Tensor, state: Tensor) -> Result<(Tensor, Tensor)> {
    let (b, n, h, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let dev = q.device();
    let chunks = n.div_ceil(CHUNK);
    let pad = chunks * CHUNK - n;
    // [b·h, chunks, CHUNK, d], zero-padded: a padded step has beta 0 and decay 1, so it changes nothing
    let blocks = |x: &Tensor| -> Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?;
        let d = x.dim(3)?;
        let x = if pad > 0 { Tensor::cat(&[&x, &Tensor::zeros((b, h, pad, d), DType::F32, dev)?], 2)? } else { x };
        x.reshape((b * h, chunks, CHUNK, d))
    };
    let (q, k, v) = (blocks(q)?, blocks(k)?, blocks(v)?);
    let per_step = |x: &Tensor| -> Result<Tensor> { blocks(&x.unsqueeze(3)?)?.squeeze(3) }; // [b·h, chunks, CHUNK]
    let (g, beta) = (per_step(log_decay)?.cumsum(2)?, per_step(beta)?);
    let lower: Vec<f32> = (0..CHUNK * CHUNK).map(|i| f32::from(i % CHUNK <= i / CHUNK)).collect();
    let lower = Tensor::from_vec(lower, (CHUNK, CHUNK), dev)?;
    let eye = Tensor::eye(CHUNK, DType::F32, dev)?;
    let strict = (&lower - &eye)?;
    // decay from step j to step i (j ≤ i) within a chunk; zero above the diagonal
    let diff = g.unsqueeze(3)?.broadcast_sub(&g.unsqueeze(2)?)?;
    let decay = diff.broadcast_mul(&lower)?.exp()?.broadcast_mul(&lower)?;
    let k_beta = k.broadcast_mul(&beta.unsqueeze(3)?)?;
    let v_beta = v.broadcast_mul(&beta.unsqueeze(3)?)?;
    let l = k_beta.matmul(&k.transpose(2, 3)?.contiguous()?)?.mul(&decay)?.broadcast_mul(&strict)?;
    // (I + L)⁻¹ by doubling diagonal blocks: from the inverses D of the size-s blocks, the size-2s ones
    // are D − D·C·D, C the lower-left parts the 2s blocks add. Every product is a true sub-block inverse,
    // so nothing grows and cancels (a power series in L reaches ~1e20 when a chunk's keys repeat).
    let m = l.broadcast_add(&eye)?;
    let block = |w: usize| -> Vec<f32> { (0..CHUNK * CHUNK).map(|i| f32::from(i / CHUNK / w == i % CHUNK / w)).collect() };
    let mut inverse = eye.broadcast_as(m.shape())?.contiguous()?;
    let mut w = 1;
    while w < CHUNK {
        let added: Vec<f32> = block(2 * w).iter().zip(block(w)).map(|(a, b)| a - b).collect();
        let c = m.broadcast_mul(&Tensor::from_vec(added, (CHUNK, CHUNK), dev)?)?;
        inverse = (&inverse - inverse.matmul(&c)?.matmul(&inverse)?)?;
        w *= 2;
    }
    let v_fixed = inverse.matmul(&v_beta)?;
    let k_decayed = inverse.matmul(&k_beta.broadcast_mul(&g.exp()?.unsqueeze(3)?)?)?;
    let mut state = state.reshape((b * h, dk, dv))?;
    let mut outs = Vec::with_capacity(chunks);
    for c in 0..chunks {
        let at = |x: &Tensor| -> Result<Tensor> { x.narrow(1, c, 1)?.squeeze(1)?.contiguous() };
        let (qc, kc, gc) = (at(&q)?, at(&k)?, at(&g)?);
        let inner = qc.matmul(&kc.transpose(1, 2)?.contiguous()?)?.mul(&at(&decay)?)?;
        let v_new = (at(&v_fixed)? - at(&k_decayed)?.matmul(&state)?)?;
        let from_state = qc.broadcast_mul(&gc.exp()?.unsqueeze(2)?)?.matmul(&state)?;
        outs.push((from_state + inner.matmul(&v_new)?)?);
        let last = gc.narrow(1, CHUNK - 1, 1)?; // [b·h, 1]
        let k_to_end = kc.broadcast_mul(&last.broadcast_sub(&gc)?.exp()?.unsqueeze(2)?)?;
        state = (state.broadcast_mul(&last.exp()?.unsqueeze(2)?)? + k_to_end.transpose(1, 2)?.contiguous()?.matmul(&v_new)?)?;
    }
    let o = Tensor::cat(&outs, 1)?.narrow(1, 0, n)?.reshape((b, h, n, dv))?.transpose(1, 2)?.contiguous()?;
    Ok((o, state.reshape((b, h, dk, dv))?))
}

/// log(1 + eˣ) as max(x, 0) + log(1 + e^−|x|): no exp overflows, so a large input gives a finite decay.
fn softplus(x: &Tensor) -> Result<Tensor> {
    x.relu()? + (x.abs()?.neg()?.exp()? + 1.0)?.log()?
}

fn causal(att: &Tensor, n: usize, pos: usize) -> Result<Tensor> {
    let total = pos + n;
    let mask: Vec<u8> = (0..n).flat_map(|i| (0..total).map(move |j| u8::from(j > pos + i))).collect();
    let mask = Tensor::from_slice(&mask, (n, total), att.device())?.broadcast_as(att.shape())?;
    let neg = Tensor::new(f32::NEG_INFINITY, att.device())?.broadcast_as(att.shape())?;
    mask.where_cond(&neg, att)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// softplus stays finite where exp overflows (a step's log-decay of −∞ makes the chunked
    /// recurrence's within-chunk decay −∞ − (−∞) = NaN), and is exact at the ends and the middle.
    #[test]
    fn softplus_is_finite_and_exact_for_large_inputs() {
        let x = Tensor::new(&[100f32, 20.0, 0.0, -100.0], &Device::Cpu).unwrap();
        let y = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        let want = [100.0, 20.0 + (-20f32).exp().ln_1p(), std::f32::consts::LN_2, (-100f32).exp().ln_1p()];
        for (got, want) in y.iter().zip(want) {
            assert!(got.is_finite() && (got - want).abs() <= 1e-6 * want.abs().max(1.0), "softplus gave {got}, want {want}");
        }
    }

    /// A deterministic tensor of `shape`, values in (−1, 1).
    fn wave(shape: &[usize], seed: f32) -> Tensor {
        let len: usize = shape.iter().product();
        let v: Vec<f32> = (0..len).map(|i| ((i as f32 + 1.0) * 0.7071 + seed).sin()).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn the_chunked_recurrence_is_the_step_by_step_one() {
        // 150 tokens: two whole chunks and a padded third, from a state already holding something
        let (b, n, h, dk, dv) = (1, 150, 2, 8, 6);
        let unit = |t: Tensor| t.broadcast_div(&t.sqr().unwrap().sum_keepdim(3).unwrap().sqrt().unwrap()).unwrap();
        let q = (unit(wave(&[b, n, h, dk], 0.1)) / (dk as f64).sqrt()).unwrap();
        let k = unit(wave(&[b, n, h, dk], 0.2));
        let v = wave(&[b, n, h, dv], 0.3);
        let log_decay = ((wave(&[b, n, h], 0.4) - 1.5).unwrap() * 0.1).unwrap();
        let beta = candle_nn::ops::sigmoid(&wave(&[b, n, h], 0.5)).unwrap();
        let state = wave(&[b, h, dk, dv], 0.6);
        let (o1, s1) = delta_steps(&q, &k, &v, &log_decay, &beta, state.clone()).unwrap();
        let (o2, s2) = delta_chunks(&q, &k, &v, &log_decay, &beta, state).unwrap();
        let worst = |a: &Tensor, b: &Tensor| (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(o1.dims(), o2.dims());
        let (wo, ws) = (worst(&o1, &o2), worst(&s1, &s2));
        assert!(wo < 1e-4 && ws < 1e-4, "chunked differs from step by step: outputs {wo:e}, state {ws:e}");
    }

    /// Keys that nearly repeat, beta and decay near 1 — a chunk whose delta-rule corrections are large —
    /// still give the step-by-step answer, finite.
    #[test]
    fn the_chunked_recurrence_holds_when_keys_repeat() {
        let (b, n, h, dk, dv) = (1, 128, 1, 8, 4);
        let unit = |t: Tensor| t.broadcast_div(&t.sqr().unwrap().sum_keepdim(3).unwrap().sqrt().unwrap()).unwrap();
        let base = Tensor::ones((b, n, h, dk), DType::F32, &Device::Cpu).unwrap();
        let k = unit((base + (wave(&[b, n, h, dk], 0.2) * 0.05).unwrap()).unwrap());
        let q = (k.clone() / (dk as f64).sqrt()).unwrap();
        let v = wave(&[b, n, h, dv], 0.3);
        let log_decay = (Tensor::ones((b, n, h), DType::F32, &Device::Cpu).unwrap() * -1e-3).unwrap();
        let beta = (Tensor::ones((b, n, h), DType::F32, &Device::Cpu).unwrap() * 0.99).unwrap();
        let state = Tensor::zeros((b, h, dk, dv), DType::F32, &Device::Cpu).unwrap();
        let (o1, s1) = delta_steps(&q, &k, &v, &log_decay, &beta, state.clone()).unwrap();
        let (o2, s2) = delta_chunks(&q, &k, &v, &log_decay, &beta, state).unwrap();
        let worst = |a: &Tensor, b: &Tensor| (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let (wo, ws) = (worst(&o1, &o2), worst(&s1, &s2));
        assert!(wo.is_finite() && ws.is_finite() && wo < 1e-3 && ws < 1e-3, "chunked differs from step by step: outputs {wo:e}, state {ws:e}");
    }
}
