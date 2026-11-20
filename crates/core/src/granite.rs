//! IBM Granite 4.0's hybrid text decoder (`granitemoehybrid` — `h-350m`, `h-1b`, `h-tiny`) from
//! safetensors, its persistent state OPEN. Follows transformers' `modeling_granitemoehybrid.py`
//! (its reference torch paths); `tests/hf_reference.rs` pins it.
//!
//! Most layers are Mamba-2: one input projection to [gate | x B C | dt], a causal depthwise conv
//! (with bias, SiLU) over x B C, dt = softplus(dt + dt_bias), A = −exp(A_log), and per head the
//! selective scan `S ← exp(dt·A)·S + dt·x ⊗ B`, `y = S·C + D·x`; then RMS(y · SiLU(gate))·w and the
//! output projection. The rest are attention with NO position encoding, scores scaled by
//! `attention_multiplier`. μP multipliers: embeddings × `embedding_multiplier`, each sublayer's
//! output × `residual_multiplier`, logits ÷ `logits_scaling`. The MLP is one projection to
//! [gate | up]; with experts (`h-tiny`) a router's top-k logits, softmaxed, weight that many expert
//! MLPs of the same shape, added to the shared one. Norms scale by w; the head is tied.
//!
//! STATE, in the open `Kv`: an attention layer's slot is (K, V) `[1, kv_heads, n, head_dim]`; a
//! Mamba layer's slot is (C, S) — C the last `conv − 1` pre-conv rows `[1, conv_channels, conv − 1]`,
//! S the scan state `[1, heads, head_dim, d_state]` — what a bridge copies across.

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
    pub eps: f64,
    pub vocab_size: usize,
    /// Per layer: true for attention, false for Mamba-2.
    pub attention: Vec<bool>,
    pub embedding_multiplier: f64,
    pub residual_multiplier: f64,
    pub attention_multiplier: f64,
    pub logits_scaling: f64,
    pub m_heads: usize,
    pub m_head_dim: usize,
    pub d_state: usize,
    pub groups: usize,
    pub conv: usize,
    /// Experts per layer (0 = dense), how many each token uses, and each one's MLP width.
    pub experts: usize,
    pub top_k: usize,
    pub expert_size: usize,
}

impl Config {
    pub fn from_json(config_json: &[u8]) -> std::result::Result<Config, String> {
        let t: serde_json::Value = serde_json::from_slice(config_json).map_err(|e| e.to_string())?;
        let u = |k: &str| t[k].as_u64().map(|v| v as usize).ok_or(format!("config has no {k}"));
        let f = |k: &str| t[k].as_f64().ok_or(format!("config has no {k}"));
        if t["position_embedding_type"].as_str() == Some("rope") {
            return Err("granitemoehybrid with RoPE is not supported yet — Granite 4.0's hybrids use none".into());
        }
        let hidden_size = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        let layers = u("num_hidden_layers")?;
        let types = t["layer_types"].as_array().ok_or("config has no layer_types")?;
        let m_heads = u("mamba_n_heads")?;
        let inner = u("mamba_expand")? * hidden_size;
        Ok(Config {
            hidden_size,
            mlp_size: u("shared_intermediate_size")?,
            layers,
            heads,
            kv_heads: u("num_key_value_heads")?,
            head_dim: hidden_size / heads,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            vocab_size: u("vocab_size")?,
            attention: types.iter().map(|v| v == "attention").collect(),
            embedding_multiplier: f("embedding_multiplier")?,
            residual_multiplier: f("residual_multiplier")?,
            attention_multiplier: f("attention_multiplier")?,
            logits_scaling: f("logits_scaling")?,
            m_heads,
            m_head_dim: t["mamba_d_head"].as_u64().map_or(inner / m_heads, |v| v as usize),
            d_state: u("mamba_d_state")?,
            groups: u("mamba_n_groups")?,
            conv: u("mamba_d_conv")?,
            experts: t["num_local_experts"].as_u64().unwrap_or(0) as usize,
            top_k: t["num_experts_per_tok"].as_u64().unwrap_or(0) as usize,
            expert_size: t["intermediate_size"].as_u64().unwrap_or(0) as usize,
        })
    }
    fn inner(&self) -> usize {
        self.m_heads * self.m_head_dim
    }
    fn conv_channels(&self) -> usize {
        self.inner() + 2 * self.groups * self.d_state
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
}

struct Mamba {
    in_proj: Lin,
    /// `[conv_channels, conv]`
    conv: Tensor,
    conv_bias: Tensor,
    /// −exp(A_log), `[heads]`
    neg_a: Tensor,
    dt_bias: Tensor,
    d: Tensor,
    norm: Norm,
    out: Lin,
}

enum Mixer {
    Attention(Attention),
    Mamba(Mamba),
}

/// A mixture of experts: the router `[experts, hidden]`, and every expert's [gate | up] `[experts,
/// 2·size, hidden]` and down `[experts, hidden, size]` projections, at the loaded precision.
struct Moe {
    router: Lin,
    router_experts: usize,
    gate_up: crate::half::Experts,
    down: crate::half::Experts,
    top_k: usize,
    size: usize,
}

impl Moe {
    /// `y` `[b, n, hidden]`. The routing is read on the host — which experts, and their weights — and
    /// each expert runs once over the tokens that chose it.
    fn forward(&self, y: &Tensor) -> Result<Tensor> {
        let (b, n, h) = y.dims3()?;
        let flat = y.reshape((b * n, h))?;
        let logits: Vec<Vec<f32>> = self.router.forward(&flat)?.to_vec2()?;
        let experts = self.router_experts;
        let mut chosen: Vec<(Vec<u32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); experts];
        for (t, row) in logits.iter().enumerate() {
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_by(|&i, &j| row[j].total_cmp(&row[i]));
            let top = &order[..self.top_k];
            let peak = row[top[0]];
            let total: f32 = top.iter().map(|&e| (row[e] - peak).exp()).sum();
            for &e in top {
                chosen[e].0.push(t as u32);
                chosen[e].1.push((row[e] - peak).exp() / total);
            }
        }
        let mut out = flat.zeros_like()?;
        for (e, (tokens, weights)) in chosen.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let idx = Tensor::new(tokens.as_slice(), y.device())?;
            let xs = flat.index_select(&idx, 0)?;
            let gu = self.gate_up.forward(e, &xs)?;
            let act = (candle_nn::ops::silu(&gu.narrow(1, 0, self.size)?)? * gu.narrow(1, self.size, self.size)?)?;
            let o = self.down.forward(e, &act)?;
            let w = Tensor::new(weights.as_slice(), y.device())?.unsqueeze(1)?;
            out = out.index_add(&idx, &o.broadcast_mul(&w)?, 0)?;
        }
        out.reshape((b, n, h))
    }
}

struct Layer {
    mixer: Mixer,
    moe: Option<Moe>,
    input_norm: Norm,
    post_norm: Norm,
    /// `[2·mlp_size, hidden]`: gate rows, then up rows
    mlp_in: Lin,
    mlp_out: Lin,
}

pub struct Granite {
    differentiable: bool,
    cfg: Config,
    embed: Embed,
    layers: Vec<Layer>,
    norm: Norm,
    head: Lin,
}

impl Granite {
    pub fn load(config_json: &[u8], weights: impl Into<crate::weights::Weights>, device: &Device) -> std::result::Result<Granite, String> {
        Self::load_in(config_json, weights, device, DType::F32)
    }

    /// As `load`, its weights held in `dtype` (`crate::half`).
    pub fn load_in(config_json: &[u8], weights: impl Into<crate::weights::Weights>, device: &Device, dtype: DType) -> std::result::Result<Granite, String> {
        let cfg = Config::from_json(config_json)?;
        let vb = weights.into().builder(dtype, device)?;
        Self::build(cfg, vb).map_err(|e| e.to_string())
    }

    /// As `load_in`, from a VarBuilder already made — `model::load_decoder_q` builds under
    /// `crate::half::quantizing` with one.
    pub fn from_builder(config_json: &[u8], vb: VarBuilder) -> std::result::Result<Granite, String> {
        Self::build(Config::from_json(config_json)?, vb).map_err(|e| e.to_string())
    }

    fn build(cfg: Config, root: VarBuilder) -> Result<Granite> {
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
                    o: linear(cfg.heads * hd, h, a.pp("o_proj"))?,
                })
            } else {
                let m = l.pp("mamba");
                let (ch, inner, mh) = (cfg.conv_channels(), cfg.inner(), cfg.m_heads);
                Mixer::Mamba(Mamba {
                    in_proj: linear(h, inner + ch + mh, m.pp("in_proj"))?,
                    conv: small(&m, (ch, 1, cfg.conv), "conv1d.weight")?.squeeze(1)?,
                    conv_bias: small(&m, ch, "conv1d.bias")?,
                    neg_a: small(&m, mh, "A_log")?.exp()?.neg()?,
                    dt_bias: small(&m, mh, "dt_bias")?,
                    d: small(&m, mh, "D")?,
                    norm: Norm::load(inner, cfg.eps, m.pp("norm"))?,
                    out: linear(inner, h, m.pp("out_proj"))?,
                })
            };
            let moe = if cfg.experts > 0 {
                let e = l.pp("block_sparse_moe");
                Some(Moe {
                    router: linear(h, cfg.experts, e.pp("router.layer"))?,
                    router_experts: cfg.experts,
                    gate_up: crate::half::experts(&e, (cfg.experts, 2 * cfg.expert_size, h), "input_linear.weight")?,
                    down: crate::half::experts(&e, (cfg.experts, h, cfg.expert_size), "output_linear.weight")?,
                    top_k: cfg.top_k,
                    size: cfg.expert_size,
                })
            } else {
                None
            };
            layers.push(Layer {
                mixer,
                moe,
                input_norm: Norm::load(h, cfg.eps, l.pp("input_layernorm"))?,
                post_norm: Norm::load(h, cfg.eps, l.pp("post_attention_layernorm"))?,
                mlp_in: linear(h, 2 * cfg.mlp_size, l.pp("shared_mlp.input_linear"))?,
                mlp_out: linear(cfg.mlp_size, h, l.pp("shared_mlp.output_linear"))?,
            });
        }
        let head = if root.contains_tensor("lm_head.weight") { linear(h, cfg.vocab_size, root.pp("lm_head"))? } else { Lin::new(embed.embeddings().clone(), None) };
        Ok(Granite { differentiable: false, norm: Norm::load(h, cfg.eps, vb.pp("norm"))?, head, embed, layers, cfg })
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

    /// The attention layers' shape (the Mamba layers' state is copied, not mapped). No RoPE: theta 0.
    pub fn geometry(&self) -> Geometry {
        Geometry { layers: self.cfg.layers, kv_heads: self.cfg.kv_heads, head_dim: self.cfg.head_dim, rope_theta_bits: 0 }
    }

    /// Keys carry no position here, so there is nothing to rotate.
    pub fn rotate_keys(&self, k: &Tensor, _pos: usize, _inverse: bool) -> Result<Tensor> {
        Ok(k.clone())
    }

    /// Mamba-2 layer `layer`'s normed input rows `[n, hidden]` from the residual stream `[1, n, hidden]`
    /// entering it (`forward_capture`) — what a bridge fits where the scan state cannot be mapped; None
    /// for an attention layer.
    pub fn layer_input(&self, layer: usize, residual: &Tensor) -> Result<Option<Tensor>> {
        let l = &self.layers[layer];
        if !matches!(l.mixer, Mixer::Mamba(_)) {
            return Ok(None);
        }
        Ok(Some(l.input_norm.forward(residual, self.differentiable)?.squeeze(0)?))
    }

    /// Mamba-2 layer `layer`'s state (C, S) built from its normed inputs `[1, n, hidden]`: the layer's own
    /// projection, conv and scan over every row, from an empty state.
    pub fn state_from_inputs(&self, layer: usize, inputs: &Tensor) -> Result<(Tensor, Tensor)> {
        let Mixer::Mamba(m) = &self.layers[layer].mixer else { candle_core::bail!("layer {layer} is attention, not Mamba-2") };
        let (b, n, _) = inputs.dims3()?;
        let mut slot = None;
        self.mamba(m, inputs, &mut slot, b, n)?;
        slot.ok_or_else(|| candle_core::Error::Msg(format!("layer {layer}: the scan left no state")))
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
        self.head.forward(x)? / self.cfg.logits_scaling
    }

    fn run(&self, ids: &Tensor, pos: usize, kv: &mut Kv, mut capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        let (b, n) = ids.dims2()?;
        let (res, diff) = (self.cfg.residual_multiplier, self.differentiable);
        let mut x = (self.embed.forward(ids)? * self.cfg.embedding_multiplier)?;
        let mut slots = kv.layers().to_vec();
        for (layer, slot) in self.layers.iter().zip(slots.iter_mut()) {
            if let Some(cap) = capture.as_deref_mut() {
                cap.push(x.clone());
            }
            let y = layer.input_norm.forward(&x, diff)?;
            let mixed = match &layer.mixer {
                Mixer::Attention(a) => self.attention(a, &y, pos, slot, b, n)?,
                Mixer::Mamba(m) => self.mamba(m, &y, slot, b, n)?,
            };
            x = (x + (mixed * res)?)?;
            let y = layer.post_norm.forward(&x, diff)?;
            let gu = layer.mlp_in.forward(&y)?;
            let m = self.cfg.mlp_size;
            let mut mlp = layer.mlp_out.forward(&(candle_nn::ops::silu(&gu.narrow(2, 0, m)?)? * gu.narrow(2, m, m)?)?)?;
            if let Some(moe) = &layer.moe {
                mlp = (moe.forward(&y)? + mlp)?;
            }
            x = (x + (mlp * res)?)?;
        }
        *kv = Kv::from_layers(slots);
        self.norm.forward(&x, diff)
    }

    fn attention(&self, a: &Attention, y: &Tensor, pos: usize, slot: &mut Option<(Tensor, Tensor)>, b: usize, n: usize) -> Result<Tensor> {
        let c = &self.cfg;
        let hd = c.head_dim;
        let q = a.q.forward(y)?.reshape((b, n, c.heads, hd))?.transpose(1, 2)?.contiguous()?;
        let mut k = a.k.forward(y)?.reshape((b, n, c.kv_heads, hd))?.transpose(1, 2)?.contiguous()?;
        let mut v = a.v.forward(y)?.reshape((b, n, c.kv_heads, hd))?.transpose(1, 2)?.contiguous()?;
        if let Some((pk, pv)) = slot.as_ref() {
            k = Tensor::cat(&[pk, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[pv, &v], 2)?.contiguous()?;
        }
        *slot = Some((k.clone(), v.clone()));
        let o = crate::attend::attend(&q, k, v, pos, c.attention_multiplier, None, self.differentiable)?;
        a.o.forward(&o)
    }

    fn mamba(&self, m: &Mamba, y: &Tensor, slot: &mut Option<(Tensor, Tensor)>, b: usize, n: usize) -> Result<Tensor> {
        let c = &self.cfg;
        let (inner, ch, mh, p, ns, g) = (c.inner(), c.conv_channels(), c.m_heads, c.m_head_dim, c.d_state, c.groups);
        let proj = m.in_proj.forward(y)?; // [b, n, inner + ch + mh]
        let gate = proj.narrow(2, 0, inner)?;
        let raw = proj.narrow(2, inner, ch)?.transpose(1, 2)?.contiguous()?; // [b, ch, n]
        let dt = proj.narrow(2, inner + ch, mh)?;
        let (prev_c, prev_s) = match slot.as_ref() {
            Some((pc, ps)) => (pc.clone(), Some(ps.clone())),
            None => (Tensor::zeros((b, ch, c.conv - 1), DType::F32, y.device())?, None),
        };
        let padded = Tensor::cat(&[&prev_c, &raw], 2)?; // [b, ch, conv − 1 + n]
        let new_c = padded.narrow(2, n, c.conv - 1)?.contiguous()?;
        // depthwise causal conv: out[t] = bias + Σ_j w[j] · padded[t + j]
        let mut conv = m.conv_bias.reshape((1, ch, 1))?.broadcast_as((b, ch, n))?.contiguous()?;
        for j in 0..c.conv {
            let w = m.conv.narrow(1, j, 1)?.reshape((1, ch, 1))?;
            conv = (conv + padded.narrow(2, j, n)?.broadcast_mul(&w)?)?;
        }
        let xbc = candle_nn::ops::silu(&conv)?.transpose(1, 2)?.contiguous()?; // [b, n, ch]
        let x = xbc.narrow(2, 0, inner)?.reshape((b, n, mh, p))?;
        let per_head = |t: Tensor| -> Result<Tensor> {
            // [b, n, g·ns] → [b, n, mh, ns], each group shared by mh / g heads
            t.reshape((b, n, g, 1, ns))?.repeat((1, 1, 1, mh / g, 1))?.reshape((b, n, mh, ns))
        };
        let bm = per_head(xbc.narrow(2, inner, g * ns)?)?;
        let cm = per_head(xbc.narrow(2, inner + g * ns, g * ns)?)?;
        let dt = softplus(&dt.broadcast_add(&m.dt_bias)?)?; // [b, n, mh]
        let log_decay = dt.broadcast_mul(&m.neg_a)?;
        let state = match prev_s {
            Some(s) => s,
            None => Tensor::zeros((b, mh, p, ns), DType::F32, y.device())?,
        };
        let (out, state) = scan_chunks(&(x.broadcast_mul(&dt.unsqueeze(3)?))?, &bm, &cm, &log_decay, state)?;
        *slot = Some((new_c, state));
        let out = (out + x.broadcast_mul(&m.d.reshape((1, 1, mh, 1))?)?)?.reshape((b, n, inner))?;
        let normed = m.norm.forward(&(out * candle_nn::ops::silu(&gate)?)?, self.differentiable)?;
        m.out.forward(&normed)
    }
}

const CHUNK: usize = 64;

/// The selective scan in chunks: x `[b, n, h, p]` (already × dt), B and C `[b, n, h, ns]`, the log
/// decay dt·A `[b, n, h]`, state `[b, h, p, ns]`. Within a chunk the outputs are one masked matmul,
/// `Y = ((C Bᵀ) ∘ exp(gᵢ − gⱼ)) X`, plus the entering state read through C; the state crosses chunk
/// boundaries once per chunk. Returns the outputs `[b, n, h, p]` (no D term) and the final state.
fn scan_chunks(x: &Tensor, bm: &Tensor, cm: &Tensor, log_decay: &Tensor, state: Tensor) -> Result<(Tensor, Tensor)> {
    let (b, n, h, p) = x.dims4()?;
    let ns = bm.dim(3)?;
    let dev = x.device();
    let q = CHUNK.min(n);
    let chunks = n.div_ceil(q);
    let pad = chunks * q - n;
    // [b·h, chunks, q, d], zero-padded: a padded step has x 0 and decay 1, so it changes nothing
    let blocks = |t: &Tensor| -> Result<Tensor> {
        let t = t.transpose(1, 2)?.contiguous()?;
        let d = t.dim(3)?;
        let t = if pad > 0 { Tensor::cat(&[&t, &Tensor::zeros((b, h, pad, d), DType::F32, dev)?], 2)? } else { t };
        t.reshape((b * h, chunks, q, d))
    };
    let (x, bm, cm) = (blocks(x)?, blocks(bm)?, blocks(cm)?);
    let g = blocks(&log_decay.unsqueeze(3)?)?.squeeze(3)?.cumsum(2)?; // [b·h, chunks, q]
    let lower: Vec<f32> = (0..q * q).map(|i| f32::from(i % q <= i / q)).collect();
    let lower = Tensor::from_vec(lower, (q, q), dev)?;
    let diffs = g.unsqueeze(3)?.broadcast_sub(&g.unsqueeze(2)?)?;
    let decay = diffs.broadcast_mul(&lower)?.exp()?.broadcast_mul(&lower)?;
    let inner = cm.matmul(&bm.transpose(2, 3)?.contiguous()?)?.mul(&decay)?.matmul(&x)?; // [b·h, chunks, q, p]
    // the state as [ns, p], so reading it is C · S and writing it is Bᵀ · X
    let mut st = state.reshape((b * h, p, ns))?.transpose(1, 2)?.contiguous()?;
    let mut outs = Vec::with_capacity(chunks);
    for k in 0..chunks {
        let at = |t: &Tensor| -> Result<Tensor> { t.narrow(1, k, 1)?.squeeze(1)?.contiguous() };
        let (xc, bc, cc, gc) = (at(&x)?, at(&bm)?, at(&cm)?, at(&g)?);
        let from_state = cc.broadcast_mul(&gc.exp()?.unsqueeze(2)?)?.matmul(&st)?;
        outs.push((from_state + at(&inner)?)?);
        let last = gc.narrow(1, q - 1, 1)?; // [b·h, 1]
        let b_to_end = bc.broadcast_mul(&last.broadcast_sub(&gc)?.exp()?.unsqueeze(2)?)?;
        st = (st.broadcast_mul(&last.exp()?.unsqueeze(2)?)? + b_to_end.transpose(1, 2)?.contiguous()?.matmul(&xc)?)?;
    }
    let o = Tensor::cat(&outs, 1)?.narrow(1, 0, n)?.reshape((b, h, n, p))?.transpose(1, 2)?.contiguous()?;
    Ok((o, st.transpose(1, 2)?.contiguous()?.reshape((b, h, p, ns))?))
}

/// log(1 + eˣ) without overflow: max(x, 0) + log(1 + e^−|x|).
fn softplus(x: &Tensor) -> Result<Tensor> {
    x.relu()? + (x.abs()?.neg()?.exp()? + 1.0)?.log()?
}
