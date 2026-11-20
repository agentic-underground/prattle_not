//! Qwen3.5's hybrid decoder on the GPU — the browser's flagship pair, 0.8B → 2B. The arithmetic is
//! prattle-core's `qwen3_5` module's (transformers' `modeling_qwen3_5.py`), step for step; its weights
//! are f16 on the GPU, its activations and state f32.
//!
//! Every `full_attention_interval`-th layer is gated attention: q and its gate from one projection,
//! RMS(1 + w) q/k norms, RoPE over the first `rot` dims of each head, the output × σ(gate). The rest
//! are Gated DeltaNet: a causal depthwise conv (SiLU) over [q | k | v], L2-normalised q and k, q ÷ √dk,
//! the delta rule per value head, then RMS(o)·w · SiLU(z) and the output projection.
//!
//! STATE: an attention layer's KV cache (positions × kv width, keys rotated); a GDN layer's conv rows
//! (conv − 1 × channels) and its recurrent state (value heads × dk × dv) — what a bridge copies across
//! from a source of the same recurrent shape.

use prattle_core::qwen3_5::Config;

use crate::bridge::{Carry, Kind};
use crate::stream::Store;
use crate::{Gpu, GpuTensor, Weights};

struct Attention {
    /// q and its gate, side by side per head: heads·2hd × hidden.
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    o: GpuTensor,
    q_norm: GpuTensor,
    k_norm: GpuTensor,
    kc: GpuTensor,
    vc: GpuTensor,
}

struct Gdn {
    qkv: GpuTensor,
    z: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    /// channels × conv
    conv: GpuTensor,
    /// −exp(A_log), per value head
    neg_a: GpuTensor,
    dt_bias: GpuTensor,
    norm: GpuTensor,
    out: GpuTensor,
    /// The conv's last conv − 1 input rows (conv − 1 × channels), oldest first.
    c: GpuTensor,
    /// The recurrent state, value heads × dk × dv.
    r: GpuTensor,
}

enum Mixer {
    Attention(Attention),
    Gdn(Gdn),
}

struct Layer {
    mixer: Mixer,
    ln1: GpuTensor,
    ln2: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    down: GpuTensor,
}

pub struct Qwen35 {
    cfg: Config,
    eps: f32,
    max_ctx: usize,
    embed: GpuTensor,
    head: GpuTensor,
    norm: GpuTensor,
    cos: GpuTensor,
    sin: GpuTensor,
    layers: Vec<Layer>,
    /// How many positions the state holds.
    pub pos: usize,
}

/// The text decoder's own name for a tensor of the file: `model.language_model.*` in the multimodal
/// checkpoint, `model.*` in a text-only one; `lm_head.weight` as it is. Anything else (the vision tower,
/// the multi-token-prediction layer) is not the decoder's.
fn local(name: &str) -> Option<&str> {
    if name == "lm_head.weight" {
        return Some(name);
    }
    name.strip_prefix("model.language_model.").or_else(|| name.strip_prefix("model.").filter(|n| !n.starts_with("visual.")))
}

/// The decoder's tensors: matrices at f16, the small ones (norms, the conv, the decay's parameters) as
/// host values the loader adjusts before upload. The vision tower and the MTP layer are skipped.
pub fn rule(name: &str, shape: &[usize]) -> Option<Store> {
    let n = local(name)?;
    if n.starts_with("mtp.") {
        return None;
    }
    let small = n.ends_with("norm.weight") || n.ends_with("A_log") || n.ends_with("dt_bias") || n.ends_with("conv1d.weight");
    Some(if small || shape.len() < 2 { Store::Small } else { Store::Half })
}

/// Whether a config.json is Qwen3.5's.
pub fn is(config: &[u8]) -> bool {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(config) else { return false };
    [&v["model_type"], &v["text_config"]["model_type"]].iter().any(|t| t.as_str().is_some_and(|k| k.starts_with("qwen3_5")))
}

impl Qwen35 {
    /// From config.json and its weights, streamed by `rule`, with `max_ctx` positions of KV cache.
    pub fn from_weights(gpu: &Gpu, config: &[u8], mut w: Weights, max_ctx: usize) -> Result<Qwen35, String> {
        let cfg = Config::from_json(config)?;
        let eps = cfg.eps as f32;
        let prefix = if w.has("model.language_model.embed_tokens.weight") { "model.language_model." } else { "model." };
        let mut take = |n: &str| w.take(&format!("{prefix}{n}"));
        let embed = take("embed_tokens.weight")?;
        let head = if w.has("lm_head.weight") { w.take("lm_head.weight")? } else { embed.clone() };
        // RMS(1 + w) for the zero-centred norms, RMS(w) for the gated one
        let mut norm = |n: &str, offset: f32| -> Result<GpuTensor, String> {
            let (shape, v) = w.host(&format!("{prefix}{n}"))?;
            Ok(gpu.upload(&v.iter().map(|x| x + offset).collect::<Vec<_>>(), &shape))
        };
        let final_norm = norm("norm.weight", 1.0)?;
        let (ch, kvw) = (2 * cfg.k_heads * cfg.d_k + cfg.v_heads * cfg.d_v, cfg.kv_heads * cfg.head_dim);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("{prefix}layers.{i}.{s}");
            let mixer = if cfg.attention[i] {
                Mixer::Attention(Attention {
                    q: w.take(&p("self_attn.q_proj.weight"))?,
                    k: w.take(&p("self_attn.k_proj.weight"))?,
                    v: w.take(&p("self_attn.v_proj.weight"))?,
                    o: w.take(&p("self_attn.o_proj.weight"))?,
                    q_norm: norm_at(gpu, &mut w, &p("self_attn.q_norm.weight"), 1.0)?,
                    k_norm: norm_at(gpu, &mut w, &p("self_attn.k_norm.weight"), 1.0)?,
                    kc: gpu.zeros(&[max_ctx, kvw]),
                    vc: gpu.zeros(&[max_ctx, kvw]),
                })
            } else {
                let (_, conv) = w.host(&p("linear_attn.conv1d.weight"))?;
                let (_, a_log) = w.host(&p("linear_attn.A_log"))?;
                let (_, dt) = w.host(&p("linear_attn.dt_bias"))?;
                Mixer::Gdn(Gdn {
                    qkv: w.take(&p("linear_attn.in_proj_qkv.weight"))?,
                    z: w.take(&p("linear_attn.in_proj_z.weight"))?,
                    a: w.take(&p("linear_attn.in_proj_a.weight"))?,
                    b: w.take(&p("linear_attn.in_proj_b.weight"))?,
                    conv: gpu.upload(&conv, &[ch, cfg.conv]),
                    neg_a: gpu.upload(&a_log.iter().map(|x| -x.exp()).collect::<Vec<_>>(), &[cfg.v_heads]),
                    dt_bias: gpu.upload(&dt, &[cfg.v_heads]),
                    norm: norm_at(gpu, &mut w, &p("linear_attn.norm.weight"), 0.0)?,
                    out: w.take(&p("linear_attn.out_proj.weight"))?,
                    c: gpu.zeros(&[cfg.conv - 1, ch]),
                    r: gpu.zeros(&[cfg.v_heads, cfg.d_k, cfg.d_v]),
                })
            };
            layers.push(Layer {
                mixer,
                ln1: norm_at(gpu, &mut w, &p("input_layernorm.weight"), 1.0)?,
                ln2: norm_at(gpu, &mut w, &p("post_attention_layernorm.weight"), 1.0)?,
                gate: w.take(&p("mlp.gate_proj.weight"))?,
                up: w.take(&p("mlp.up_proj.weight"))?,
                down: w.take(&p("mlp.down_proj.weight"))?,
            });
        }
        // The rotation tables as core builds them: f32 frequencies over the first `rot` dims.
        let inv: Vec<f32> = (0..cfg.rot).step_by(2).map(|i| 1.0 / cfg.rope_theta.powf(i as f32 / cfg.rot as f32)).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(max_ctx * inv.len()), Vec::with_capacity(max_ctx * inv.len()));
        for t in 0..max_ctx {
            for f in &inv {
                let a = t as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok(Qwen35 {
            eps,
            max_ctx,
            embed,
            head,
            norm: final_norm,
            cos: gpu.upload(&cos, &[max_ctx, inv.len()]),
            sin: gpu.upload(&sin, &[max_ctx, inv.len()]),
            layers,
            pos: 0,
            cfg,
        })
    }

    /// Read `ids` from the current position, extending the state; the last position's logits.
    pub async fn forward(&mut self, gpu: &Gpu, ids: &[u32]) -> Vec<f32> {
        let n = ids.len();
        let x = self.layers_over(gpu, ids, None);
        let f = gpu.rms_norm(&x, &self.norm, self.eps);
        let logits = gpu.matmul_rows(&f, n - 1, 1, &self.head);
        self.pos += n;
        gpu.download(&logits).await
    }

    /// Read `ids`, returning each layer's INPUT residual stream (n × hidden): what a bridge maps from.
    pub fn prefill_capture(&mut self, gpu: &Gpu, ids: &[u32]) -> Vec<GpuTensor> {
        let mut capture = Vec::with_capacity(self.layers.len());
        self.layers_over(gpu, ids, Some(&mut capture));
        self.pos += ids.len();
        capture
    }

    /// Read `ids` into the state, with no logits.
    pub fn prefill(&mut self, gpu: &Gpu, ids: &[u32]) {
        self.layers_over(gpu, ids, None);
        self.pos += ids.len();
    }

    /// Install a bridged history of `n` positions: translated (keys, values) for each attention layer,
    /// and for each Gated DeltaNet layer the source's own conv rows and recurrent state.
    pub fn install(&mut self, gpu: &Gpu, kvs: &[Carry], n: usize, source: &Qwen35) -> Result<(), String> {
        if kvs.len() != self.layers.len() || source.layers.len() != self.layers.len() {
            return Err(format!("{} translated layers and a source of {} for a target of {}", kvs.len(), source.layers.len(), self.layers.len()));
        }
        for (l, ((mine, theirs), kv)) in self.layers.iter().zip(&source.layers).zip(kvs).enumerate() {
            match (&mine.mixer, &theirs.mixer, kv) {
                (Mixer::Attention(a), _, Carry::Kv(k, v)) => {
                    gpu.copy_into(k, &a.kc, 0);
                    gpu.copy_into(v, &a.vc, 0);
                }
                (Mixer::Gdn(g), Mixer::Gdn(s), Carry::Source) => {
                    if g.c.len() != s.c.len() || g.r.len() != s.r.len() {
                        return Err(format!("layer {l}: the source's recurrent state is {:?}, the target's {:?}", s.r.shape, g.r.shape));
                    }
                    gpu.copy_into(&s.c, &g.c, 0);
                    gpu.copy_into(&s.r, &g.r, 0);
                }
                _ => return Err(format!("layer {l}: the bridge and the two models disagree on whether it is attention")),
            }
        }
        self.pos = n;
        Ok(())
    }

    fn layers_over(&self, gpu: &Gpu, ids: &[u32], mut capture: Option<&mut Vec<GpuTensor>>) -> GpuTensor {
        let n = ids.len();
        assert!(self.pos + n <= self.max_ctx, "{} positions in a cache of {}", self.pos + n, self.max_ctx);
        let x = gpu.embed(&gpu.upload_ids(ids), &self.embed);
        for l in &self.layers {
            if let Some(c) = capture.as_deref_mut() {
                c.push(gpu.copy(&x));
            }
            let y = gpu.rms_norm(&x, &l.ln1, self.eps);
            let mixed = match &l.mixer {
                Mixer::Attention(a) => self.attention(gpu, a, &y),
                Mixer::Gdn(g) => self.gdn(gpu, g, &y),
            };
            gpu.add(&x, &mixed);
            let b = gpu.rms_norm(&x, &l.ln2, self.eps);
            let s = gpu.silu_mul(&gpu.matmul(&b, &l.gate), &gpu.matmul(&b, &l.up));
            gpu.add(&x, &gpu.matmul(&s, &l.down));
        }
        x
    }

    fn attention(&self, gpu: &Gpu, a: &Attention, y: &GpuTensor) -> GpuTensor {
        let c = &self.cfg;
        let (n, hd) = (y.shape[0], c.head_dim);
        let (q, gate) = gpu.split_halves(&gpu.matmul(y, &a.q), c.heads);
        let q = gpu.rms_norm(&q.view(&[n * c.heads, hd]), &a.q_norm, self.eps).view(&[n, c.heads * hd]);
        let k = gpu.rms_norm(&gpu.matmul(y, &a.k).view(&[n * c.kv_heads, hd]), &a.k_norm, self.eps).view(&[n, c.kv_heads * hd]);
        let v = gpu.matmul(y, &a.v);
        gpu.rope_part(&q, c.heads, c.rot, &self.cos, &self.sin, self.pos);
        gpu.rope_part(&k, c.kv_heads, c.rot, &self.cos, &self.sin, self.pos);
        let width = c.kv_heads * hd;
        gpu.copy_into(&k, &a.kc, self.pos * width);
        gpu.copy_into(&v, &a.vc, self.pos * width);
        let att = gpu.attention(&q, &a.kc, &a.vc, c.heads, c.kv_heads, self.pos);
        gpu.mul_sigmoid(&att, &gate);
        gpu.matmul(&att, &a.o)
    }

    fn gdn(&self, gpu: &Gpu, g: &Gdn, y: &GpuTensor) -> GpuTensor {
        let c = &self.cfg;
        let n = y.shape[0];
        let mixed = gpu.conv_silu(&gpu.matmul(y, &g.qkv), &g.conv, &g.c);
        gpu.l2norm_heads(&mixed, 2 * c.k_heads, c.d_k, c.k_heads, 1.0 / (c.d_k as f32).sqrt());
        let (decay, beta) = (gpu.matmul(y, &g.a), gpu.matmul(y, &g.b));
        gpu.gdn_gates(&decay, &beta, &g.neg_a, &g.dt_bias);
        let o = gpu.delta(&mixed, &decay, &beta, &g.r, c.k_heads, c.v_heads, c.d_k, c.d_v);
        let o = gpu.rms_norm(&o.view(&[n * c.v_heads, c.d_v]), &g.norm, self.eps).view(&[n, c.v_heads * c.d_v]);
        gpu.matmul(&gpu.silu_mul(&gpu.matmul(y, &g.z), &o), &g.out)
    }

    /// Rotate keys (n × kv width) in place to positions pos..pos+n, as the attention layers do.
    pub fn rotate(&self, gpu: &Gpu, k: &GpuTensor, pos: usize) {
        gpu.rope_part(k, self.cfg.kv_heads, self.cfg.rot, &self.cos, &self.sin, pos);
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn kv_width(&self) -> usize {
        self.cfg.kv_heads * self.cfg.head_dim
    }

    /// What layer `l` takes from a bridge: keys and values (attention), or the source's own state.
    pub fn kind(&self, l: usize) -> Kind {
        if self.cfg.attention[l] { Kind::Kv } else { Kind::Source }
    }

    /// Forget every position: the recurrent state zeroed, the next forward at 0.
    pub fn reset(&mut self, gpu: &Gpu) {
        for l in &self.layers {
            if let Mixer::Gdn(g) = &l.mixer {
                gpu.clear(&g.c);
                gpu.clear(&g.r);
            }
        }
        self.pos = 0;
    }
}

/// A small weight as an RMS norm's scale: 1 + w for the zero-centred norms, w for the gated one.
fn norm_at(gpu: &Gpu, w: &mut Weights, name: &str, offset: f32) -> Result<GpuTensor, String> {
    let (shape, v) = w.host(name)?;
    Ok(gpu.upload(&v.iter().map(|x| x + offset).collect::<Vec<_>>(), &shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decoder_keeps_its_own_tensors_matrices_at_half_and_skips_vision_and_mtp() {
        let m = &[2048, 1024][..];
        assert_eq!(rule("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", m), Some(Store::Half));
        assert_eq!(rule("model.language_model.embed_tokens.weight", m), Some(Store::Half));
        assert_eq!(rule("model.layers.3.self_attn.q_proj.weight", m), Some(Store::Half));
        assert_eq!(rule("lm_head.weight", m), Some(Store::Half));
        assert_eq!(rule("model.language_model.layers.0.input_layernorm.weight", &[1024]), Some(Store::Small));
        assert_eq!(rule("model.language_model.layers.0.linear_attn.conv1d.weight", &[6144, 1, 4]), Some(Store::Small));
        assert_eq!(rule("model.language_model.layers.0.linear_attn.A_log", &[16]), Some(Store::Small));
        assert_eq!(rule("model.visual.blocks.0.attn.qkv.weight", m), None);
        assert_eq!(rule("mtp.layers.0.mlp.gate_proj.weight", m), None);
        assert_eq!(rule("model.language_model.mtp.fc.weight", m), None);
    }

    #[test]
    fn a_qwen35_config_is_recognised_by_its_model_type_and_a_llama_one_is_not() {
        assert!(is(br#"{"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}"#));
        assert!(is(br#"{"model_type": "qwen3_5_text"}"#));
        assert!(!is(br#"{"model_type": "llama"}"#));
    }
}
