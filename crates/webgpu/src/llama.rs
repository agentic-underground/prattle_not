//! A Llama-family decoder (SmolLM2) on the GPU, its KV cache resident there. The arithmetic is
//! prattle-core's `llama` module's, step for step, so the two can be compared token for token.

use crate::{Gpu, GpuTensor};

struct Layer {
    ln1: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    o: GpuTensor,
    ln2: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    down: GpuTensor,
    kc: GpuTensor,
    vc: GpuTensor,
}

pub struct Llama {
    heads: usize,
    kvh: usize,
    hd: usize,
    eps: f32,
    max_ctx: usize,
    embed: GpuTensor,
    head: GpuTensor,
    norm: GpuTensor,
    cos: GpuTensor,
    sin: GpuTensor,
    layers: Vec<Layer>,
    /// How many positions the cache holds.
    pub pos: usize,
}

/// A tensor's values as f32, whatever its stored precision.
fn floats(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>, String> {
    let b = view.data();
    Ok(match view.dtype() {
        safetensors::Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        safetensors::Dtype::BF16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        safetensors::Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => return Err(format!("weights in {other:?} are not read")),
    })
}

impl Llama {
    /// From a model's config.json and its safetensors bytes, with `max_ctx` positions of KV cache.
    pub fn load(gpu: &Gpu, config: &[u8], weights: &[u8], max_ctx: usize) -> Result<Llama, String> {
        let c: serde_json::Value = serde_json::from_slice(config).map_err(|e| format!("config.json: {e}"))?;
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("config.json has no {k}"));
        let (h, heads, kvh, layers_n) = (u("hidden_size")?, u("num_attention_heads")?, u("num_key_value_heads")?, u("num_hidden_layers")?);
        let hd = c["head_dim"].as_u64().map_or(h / heads, |d| d as usize);
        let eps = c["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let theta = c["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
        if !c["rope_scaling"].is_null() {
            return Err("rope_scaling is not read yet: only plain RoPE (SmolLM2)".into());
        }
        let st = safetensors::SafeTensors::deserialize(weights).map_err(|e| format!("safetensors: {e}"))?;
        let get = |name: &str| -> Result<GpuTensor, String> {
            let view = st.tensor(name).map_err(|_| format!("the weights have no {name}"))?;
            Ok(gpu.upload(&floats(&view)?, view.shape()))
        };
        let embed = get("model.embed_tokens.weight")?;
        let tied = c["tie_word_embeddings"].as_bool().unwrap_or(false) || st.tensor("lm_head.weight").is_err();
        let head = if tied { get("model.embed_tokens.weight")? } else { get("lm_head.weight")? };
        let mut layers = Vec::with_capacity(layers_n);
        for l in 0..layers_n {
            let p = |s: &str| format!("model.layers.{l}.{s}.weight");
            layers.push(Layer {
                ln1: get(&p("input_layernorm"))?,
                q: get(&p("self_attn.q_proj"))?,
                k: get(&p("self_attn.k_proj"))?,
                v: get(&p("self_attn.v_proj"))?,
                o: get(&p("self_attn.o_proj"))?,
                ln2: get(&p("post_attention_layernorm"))?,
                gate: get(&p("mlp.gate_proj"))?,
                up: get(&p("mlp.up_proj"))?,
                down: get(&p("mlp.down_proj"))?,
                kc: gpu.zeros(&[max_ctx, kvh * hd]),
                vc: gpu.zeros(&[max_ctx, kvh * hd]),
            });
        }
        // The rotation tables exactly as candle builds them: f32 frequencies, f32 cos and sin.
        let inv: Vec<f32> = (0..hd).step_by(2).map(|i| 1f32 / theta.powf(i as f32 / hd as f32)).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(max_ctx * hd / 2), Vec::with_capacity(max_ctx * hd / 2));
        for t in 0..max_ctx {
            for f in &inv {
                let a = t as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok(Llama {
            heads,
            kvh,
            hd,
            eps,
            max_ctx,
            embed,
            head,
            norm: get("model.norm.weight")?,
            cos: gpu.upload(&cos, &[max_ctx, hd / 2]),
            sin: gpu.upload(&sin, &[max_ctx, hd / 2]),
            layers,
            pos: 0,
        })
    }

    /// Read `ids` from the current position, extending the cache; the last position's logits.
    pub async fn forward(&mut self, gpu: &Gpu, ids: &[u32]) -> Vec<f32> {
        let n = ids.len();
        let x = self.layers_over(gpu, ids, None);
        let f = gpu.rms_norm(&x, &self.norm, self.eps);
        let logits = gpu.matmul_rows(&f, n - 1, 1, &self.head);
        self.pos += n;
        gpu.download(&logits).await
    }

    /// Read `ids` from the current position, returning each layer's INPUT residual stream
    /// (n × hidden): the features a bridge maps from. The cache is extended as by `forward`.
    pub fn prefill_capture(&mut self, gpu: &Gpu, ids: &[u32]) -> Vec<GpuTensor> {
        let mut capture = Vec::with_capacity(self.layers.len());
        self.layers_over(gpu, ids, Some(&mut capture));
        self.pos += ids.len();
        capture
    }

    /// Read `ids` from the current position into the cache, with no logits: a history another step
    /// will continue from.
    pub fn prefill(&mut self, gpu: &Gpu, ids: &[u32]) {
        self.layers_over(gpu, ids, None);
        self.pos += ids.len();
    }

    /// Install translated (keys, values) per layer as positions 0..n, and continue from n.
    pub fn install(&mut self, gpu: &Gpu, kvs: &[(GpuTensor, GpuTensor)], n: usize) {
        assert_eq!(kvs.len(), self.layers.len(), "{} translated layers for {}", kvs.len(), self.layers.len());
        for (l, (k, v)) in self.layers.iter().zip(kvs) {
            assert_eq!(k.len(), n * self.kvh * self.hd, "translated keys are {:?}", k.shape);
            gpu.copy_into(k, &l.kc, 0);
            gpu.copy_into(v, &l.vc, 0);
        }
        self.pos = n;
    }

    /// The decoder layers over `ids` from the current position (which is not advanced); the final
    /// residual stream, and each layer's input pushed to `capture`.
    fn layers_over(&self, gpu: &Gpu, ids: &[u32], mut capture: Option<&mut Vec<GpuTensor>>) -> GpuTensor {
        let n = ids.len();
        assert!(self.pos + n <= self.max_ctx, "{} positions in a cache of {}", self.pos + n, self.max_ctx);
        let x = gpu.embed(&gpu.upload_ids(ids), &self.embed);
        let width = self.kvh * self.hd;
        for l in &self.layers {
            if let Some(c) = capture.as_deref_mut() {
                c.push(gpu.copy(&x));
            }
            let a = gpu.rms_norm(&x, &l.ln1, self.eps);
            let (q, k, v) = (gpu.matmul(&a, &l.q), gpu.matmul(&a, &l.k), gpu.matmul(&a, &l.v));
            gpu.rope(&q, self.heads, &self.cos, &self.sin, self.pos);
            gpu.rope(&k, self.kvh, &self.cos, &self.sin, self.pos);
            gpu.copy_into(&k, &l.kc, self.pos * width);
            gpu.copy_into(&v, &l.vc, self.pos * width);
            let att = gpu.attention(&q, &l.kc, &l.vc, self.heads, self.kvh, self.pos);
            gpu.add(&x, &gpu.matmul(&att, &l.o));
            let b = gpu.rms_norm(&x, &l.ln2, self.eps);
            let s = gpu.silu_mul(&gpu.matmul(&b, &l.gate), &gpu.matmul(&b, &l.up));
            gpu.add(&x, &gpu.matmul(&s, &l.down));
        }
        x
    }

    /// Rotate keys (n × kv-width) in place to positions pos..pos+n, as this model's attention does.
    pub fn rotate(&self, gpu: &Gpu, k: &GpuTensor, pos: usize) {
        gpu.rope(k, self.kvh, &self.cos, &self.sin, pos);
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The width of one layer's keys (and values) per position: KV heads × head width.
    pub fn kv_width(&self) -> usize {
        self.kvh * self.hd
    }

    /// Forget every position: the next forward starts at 0.
    pub fn reset(&mut self) {
        self.pos = 0;
    }
}
