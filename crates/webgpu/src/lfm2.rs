//! Liquid AI's LFM2 on the GPU — the browser's LFM2-350M → LFM2-1.2B pair. The arithmetic is
//! prattle-core's `lfm2` module's (transformers' `modeling_lfm2.py`), step for step; its matrices are
//! f16 on the GPU, its activations and state f32.
//!
//! Most layers are short convolutions: one projection to [B | C | x], a causal depthwise conv (no
//! activation) over B·x, then C · conv and the output projection. The rest (`full_attn_idxs`) are
//! attention with RMS q/k norms per head and full rotate-half RoPE. The MLP is SwiGLU (w1 gate, w3 up,
//! w2 down); norms scale by w; the final norm is `embedding_norm`; the head is tied.
//!
//! STATE: an attention layer's KV cache (positions × kv width, keys rotated); a conv layer's last
//! conv − 1 rows of B·x (conv − 1 × hidden), oldest first — what a bridge writes from the source's
//! residual streams, as the two sizes differ in width.

use prattle_core::lfm2::Config;

use crate::bridge::{Carry, Kind};
use crate::stream::Store;
use crate::{Gpu, GpuTensor, Weights};

struct Attention {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    o: GpuTensor,
    q_norm: GpuTensor,
    k_norm: GpuTensor,
    kc: GpuTensor,
    vc: GpuTensor,
}

struct ShortConv {
    /// [B | C | x] × hidden: 3·hidden × hidden.
    in_proj: GpuTensor,
    /// hidden × conv
    conv: GpuTensor,
    out: GpuTensor,
    /// The last conv − 1 rows of B·x (conv − 1 × hidden), oldest first.
    rows: GpuTensor,
}

// one per layer, sixteen in all: their sizes do not matter
#[allow(clippy::large_enum_variant)]
enum Mixer {
    Attention(Attention),
    Conv(ShortConv),
}

struct Layer {
    mixer: Mixer,
    operator_norm: GpuTensor,
    ffn_norm: GpuTensor,
    w1: GpuTensor,
    w2: GpuTensor,
    w3: GpuTensor,
}

pub struct Lfm2 {
    cfg: Config,
    eps: f32,
    max_ctx: usize,
    embed: GpuTensor,
    head: GpuTensor,
    norm: GpuTensor,
    cos: GpuTensor,
    sin: GpuTensor,
    layers: Vec<Layer>,
    pub pos: usize,
}

/// Matrices at f16; norms and the conv's taps at f32, which the conv kernel reads.
pub fn rule(name: &str, shape: &[usize]) -> Option<Store> {
    Some(if shape.len() >= 2 && !name.ends_with("conv.conv.weight") { Store::Half } else { Store::Full })
}

/// Whether a config.json is LFM2's.
pub fn is(config: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(config).is_ok_and(|v| v["model_type"] == "lfm2")
}

impl Lfm2 {
    /// From config.json and its weights, streamed by `rule`, with `max_ctx` positions of KV cache.
    pub fn from_weights(gpu: &Gpu, config: &[u8], mut w: Weights, max_ctx: usize) -> Result<Lfm2, String> {
        let cfg = Config::from_json(config)?;
        let (h, hd, kvw) = (cfg.hidden_size, cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let embed = w.take("model.embed_tokens.weight")?;
        let head = if w.has("lm_head.weight") { w.take("lm_head.weight")? } else { embed.clone() };
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{i}.{s}.weight");
            let mixer = if cfg.attention[i] {
                Mixer::Attention(Attention {
                    q: w.take(&p("self_attn.q_proj"))?,
                    k: w.take(&p("self_attn.k_proj"))?,
                    v: w.take(&p("self_attn.v_proj"))?,
                    o: w.take(&p("self_attn.out_proj"))?,
                    q_norm: w.take(&p("self_attn.q_layernorm"))?,
                    k_norm: w.take(&p("self_attn.k_layernorm"))?,
                    kc: gpu.zeros(&[max_ctx, kvw]),
                    vc: gpu.zeros(&[max_ctx, kvw]),
                })
            } else {
                Mixer::Conv(ShortConv {
                    in_proj: w.take(&p("conv.in_proj"))?,
                    conv: w.take(&p("conv.conv"))?.view(&[h, cfg.conv]),
                    out: w.take(&p("conv.out_proj"))?,
                    rows: gpu.zeros(&[cfg.conv - 1, h]),
                })
            };
            layers.push(Layer {
                mixer,
                operator_norm: w.take(&p("operator_norm"))?,
                ffn_norm: w.take(&p("ffn_norm"))?,
                w1: w.take(&p("feed_forward.w1"))?,
                w2: w.take(&p("feed_forward.w2"))?,
                w3: w.take(&p("feed_forward.w3"))?,
            });
        }
        // The rotation tables as core builds them: f32 frequencies over the whole head.
        let inv: Vec<f32> = (0..hd).step_by(2).map(|i| 1.0 / cfg.rope_theta.powf(i as f32 / hd as f32)).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(max_ctx * inv.len()), Vec::with_capacity(max_ctx * inv.len()));
        for t in 0..max_ctx {
            for f in &inv {
                let a = t as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok(Lfm2 {
            eps: cfg.eps as f32,
            max_ctx,
            embed,
            head,
            norm: w.take("model.embedding_norm.weight")?,
            cos: gpu.upload(&cos, &[max_ctx, inv.len()]),
            sin: gpu.upload(&sin, &[max_ctx, inv.len()]),
            layers,
            pos: 0,
            cfg,
        })
    }

    pub async fn forward(&mut self, gpu: &Gpu, ids: &[u32]) -> Vec<f32> {
        let n = ids.len();
        let x = self.layers_over(gpu, ids, None);
        let f = gpu.rms_norm(&x, &self.norm, self.eps);
        let logits = gpu.matmul_rows(&f, n - 1, 1, &self.head);
        self.pos += n;
        gpu.download(&logits).await
    }

    pub fn prefill_capture(&mut self, gpu: &Gpu, ids: &[u32]) -> Vec<GpuTensor> {
        let mut capture = Vec::with_capacity(self.layers.len());
        self.layers_over(gpu, ids, Some(&mut capture));
        self.pos += ids.len();
        capture
    }

    pub fn prefill(&mut self, gpu: &Gpu, ids: &[u32]) {
        self.layers_over(gpu, ids, None);
        self.pos += ids.len();
    }

    /// Install a bridged history of `n` positions: keys and values for each attention layer, B·x rows
    /// for each short conv.
    pub fn install(&mut self, gpu: &Gpu, carries: &[Carry], n: usize) -> Result<(), String> {
        if carries.len() != self.layers.len() {
            return Err(format!("{} translated layers for a target of {}", carries.len(), self.layers.len()));
        }
        for (l, (layer, carry)) in self.layers.iter().zip(carries).enumerate() {
            match (&layer.mixer, carry) {
                (Mixer::Attention(a), Carry::Kv(k, v)) => {
                    gpu.copy_into(k, &a.kc, 0);
                    gpu.copy_into(v, &a.vc, 0);
                }
                (Mixer::Conv(c), Carry::Rows(r)) if r.len() == c.rows.len() => gpu.copy_into(r, &c.rows, 0),
                _ => return Err(format!("layer {l}: the bridge's translation does not fit this layer")),
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
            let y = gpu.rms_norm(&x, &l.operator_norm, self.eps);
            let mixed = match &l.mixer {
                Mixer::Attention(a) => self.attention(gpu, a, &y),
                Mixer::Conv(c) => self.short_conv(gpu, c, &y),
            };
            gpu.add(&x, &mixed);
            let b = gpu.rms_norm(&x, &l.ffn_norm, self.eps);
            let s = gpu.silu_mul(&gpu.matmul(&b, &l.w1), &gpu.matmul(&b, &l.w3));
            gpu.add(&x, &gpu.matmul(&s, &l.w2));
        }
        x
    }

    fn attention(&self, gpu: &Gpu, a: &Attention, y: &GpuTensor) -> GpuTensor {
        let c = &self.cfg;
        let (n, hd) = (y.shape[0], c.head_dim);
        let q = gpu.rms_norm(&gpu.matmul(y, &a.q).view(&[n * c.heads, hd]), &a.q_norm, self.eps).view(&[n, c.heads * hd]);
        let k = gpu.rms_norm(&gpu.matmul(y, &a.k).view(&[n * c.kv_heads, hd]), &a.k_norm, self.eps).view(&[n, c.kv_heads * hd]);
        let v = gpu.matmul(y, &a.v);
        gpu.rope(&q, c.heads, &self.cos, &self.sin, self.pos);
        gpu.rope(&k, c.kv_heads, &self.cos, &self.sin, self.pos);
        let width = c.kv_heads * hd;
        gpu.copy_into(&k, &a.kc, self.pos * width);
        gpu.copy_into(&v, &a.vc, self.pos * width);
        let att = gpu.attention(&q, &a.kc, &a.vc, c.heads, c.kv_heads, self.pos);
        gpu.matmul(&att, &a.o)
    }

    fn short_conv(&self, gpu: &Gpu, s: &ShortConv, y: &GpuTensor) -> GpuTensor {
        let h = self.cfg.hidden_size;
        let bcx = gpu.matmul(y, &s.in_proj);
        let bx = gpu.mul_cols(&bcx, 0, &bcx, 2 * h, h);
        let conv = gpu.conv(&bx, &s.conv, &s.rows, false);
        gpu.matmul(&gpu.mul_cols(&bcx, h, &conv, 0, h), &s.out)
    }

    pub fn rotate(&self, gpu: &Gpu, k: &GpuTensor, pos: usize) {
        gpu.rope(k, self.cfg.kv_heads, &self.cos, &self.sin, pos);
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn kv_width(&self) -> usize {
        self.cfg.kv_heads * self.cfg.head_dim
    }

    pub fn kind(&self, l: usize) -> Kind {
        if self.cfg.attention[l] { Kind::Kv } else { Kind::Rows }
    }

    pub fn conv_rows(&self) -> usize {
        self.cfg.conv - 1
    }

    pub fn reset(&mut self, gpu: &Gpu) {
        for l in &self.layers {
            if let Mixer::Conv(c) = &l.mixer {
                gpu.clear(&c.rows);
            }
        }
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfm2_keeps_its_matrices_at_half_and_its_conv_taps_and_norms_at_f32() {
        assert_eq!(rule("model.layers.0.conv.in_proj.weight", &[3072, 1024]), Some(Store::Half));
        assert_eq!(rule("model.embed_tokens.weight", &[65536, 1024]), Some(Store::Half));
        assert_eq!(rule("model.layers.0.conv.conv.weight", &[1024, 1, 3]), Some(Store::Full));
        assert_eq!(rule("model.layers.2.self_attn.q_layernorm.weight", &[64]), Some(Store::Full));
        assert!(is(br#"{"model_type": "lfm2"}"#) && !is(br#"{"model_type": "llama"}"#));
    }
}
