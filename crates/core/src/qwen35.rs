//! The Qwen3.5 hybrid in Rust — the 4B and the 27B, its shape read from the GGUF — docs/BRIDGE.md slice 11e, checked field by
//! field against the oracle. CPU, f32, weights dequantized as each layer runs: an instrument for
//! checking the port, not yet a fast runtime.
//!
//! Layer `il` is Gated DeltaNet unless (il+1) % 4 == 0, which is gated attention. Each layer: RMS norm
//! → mixer → residual → post-attention RMS norm → SwiGLU → residual. H is the last layer's output at
//! the last position, before the final norm.

use std::collections::BTreeMap;

/// Every canonical field the oracle exports: `R.i`, `C.i` (GDN layers), `K.i`, `V.i` (attention
/// layers, `[kv_heads, tokens, head_dim]`, keys rotated), and `H`.
pub type Fields = BTreeMap<String, Vec<f32>>;

use crate::gdn::{self, Geometry, matmul_t, rms_rows};
use candle_core::Device;
use candle_core::quantized::gguf_file;

/// The model's shape, from the GGUF's own `qwen35.*` metadata — the 4B and the 27B are one
/// architecture at two sizes.
struct Config {
    hidden: usize,
    /// Main layers only: a multi-token-prediction layer, when present, is a separate head.
    layers: usize,
    ffn: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Partial RoPE: the first `rot` dims of each head, NEOX pairing (i, i + rot/2).
    rot: usize,
    rope_base: f32,
    eps: f32,
    /// Layer `il` is gated attention when (il+1) % interval == 0, Gated DeltaNet otherwise.
    interval: usize,
    gdn: Geometry,
}

impl Config {
    fn read(content: &gguf_file::Content) -> Result<Config, String> {
        let get = |k: &str| content.metadata.get(&format!("qwen35.{k}")).ok_or(format!("GGUF has no qwen35.{k}"));
        let u = |k: &str| get(k).and_then(|v| v.to_u32().map(|n| n as usize).map_err(|e| format!("qwen35.{k}: {e}")));
        let f = |k: &str| get(k).and_then(|v| v.to_f32().map_err(|e| format!("qwen35.{k}: {e}")));
        let mtp = if get("nextn_predict_layers").is_ok() { u("nextn_predict_layers")? } else { 0 };
        Ok(Config {
            hidden: u("embedding_length")?,
            layers: u("block_count")? - mtp,
            ffn: u("feed_forward_length")?,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rot: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            interval: u("full_attention_interval")?,
            gdn: Geometry {
                key_heads: u("ssm.group_count")?,
                value_heads: u("ssm.time_step_rank")?,
                dim: u("ssm.state_size")?,
                conv: u("ssm.conv_kernel")?,
            },
        })
    }
}

struct Gguf {
    content: gguf_file::Content,
    file: std::fs::File,
}

impl Gguf {
    fn get(&mut self, name: &str) -> Result<Vec<f32>, String> {
        self.content
            .tensor(&mut self.file, name, &Device::Cpu)
            .and_then(|t| t.dequantize(&Device::Cpu))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("{name}: {e}"))
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// NEOX RoPE on the first `rot` dims of each head row of `x` (`[tokens, heads, head_dim]`).
fn rope(c: &Config, x: &mut [f32], heads: usize) {
    let half = c.rot / 2;
    for (row, chunk) in x.chunks_mut(heads * c.head_dim).enumerate() {
        for head in chunk.chunks_mut(c.head_dim) {
            for i in 0..half {
                let theta = row as f32 * c.rope_base.powf(-2.0 * i as f32 / c.rot as f32);
                let (sin, cos) = theta.sin_cos();
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * cos - b * sin;
                head[i + half] = a * sin + b * cos;
            }
        }
    }
}

/// Prefill the model in `gguf` over `ids` from an empty state.
pub fn prefill(gguf: &str, ids: &[u32]) -> Result<Fields, String> {
    let mut file = std::fs::File::open(gguf).map_err(|e| format!("{gguf}: {e}"))?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;
    let c = Config::read(&content)?;
    let mut g = Gguf { content, file };
    let t = ids.len();
    let embd = g.get("token_embd.weight")?;
    let mut x: Vec<f32> = ids.iter().flat_map(|&i| embd[i as usize * c.hidden..(i as usize + 1) * c.hidden].to_vec()).collect();
    drop(embd);
    let mut fields = Fields::new();
    for il in 0..c.layers {
        let p = |n: &str| format!("blk.{il}.{n}");
        let mixed = if (il + 1) % c.interval != 0 {
            let w = gdn::Weights {
                norm: g.get(&p("attn_norm.weight"))?,
                qkv: g.get(&p("attn_qkv.weight"))?,
                alpha: g.get(&p("ssm_alpha.weight"))?,
                beta: g.get(&p("ssm_beta.weight"))?,
                a: g.get(&p("ssm_a"))?,
                dt_bias: g.get(&p("ssm_dt.bias"))?,
                conv: g.get(&p("ssm_conv1d.weight"))?,
                eps: c.eps,
            };
            let (state, o) = gdn::prefill_with_output(&c.gdn, &w, &x, c.hidden);
            fields.insert(format!("R.{il}"), state.r);
            fields.insert(format!("C.{il}"), state.c);
            // gated norm per head: rms(o_h)·ssm_norm·silu(z_h), then the output projection
            let xn = rms_rows(&x, &w.norm, c.eps);
            let z = matmul_t(&xn, t, &g.get(&p("attn_gate.weight"))?, c.gdn.value_heads * c.gdn.dim);
            let normed = rms_rows(&o, &g.get(&p("ssm_norm.weight"))?, c.eps);
            let gated: Vec<f32> = normed.iter().zip(&z).map(|(a, zz)| a * silu(*zz)).collect();
            matmul_t(&gated, t, &g.get(&p("ssm_out.weight"))?, c.hidden)
        } else {
            let xn = rms_rows(&x, &g.get(&p("attn_norm.weight"))?, c.eps);
            let qg = matmul_t(&xn, t, &g.get(&p("attn_q.weight"))?, c.heads * 2 * c.head_dim);
            let mut q = vec![0f32; t * c.heads * c.head_dim];
            let mut gate = vec![0f32; t * c.heads * c.head_dim];
            for (row, chunk) in qg.chunks(c.heads * 2 * c.head_dim).enumerate() {
                for h in 0..c.heads {
                    let base = (row * c.heads + h) * c.head_dim;
                    q[base..base + c.head_dim].copy_from_slice(&chunk[h * 2 * c.head_dim..h * 2 * c.head_dim + c.head_dim]);
                    gate[base..base + c.head_dim].copy_from_slice(&chunk[h * 2 * c.head_dim + c.head_dim..(h + 1) * 2 * c.head_dim]);
                }
            }
            let mut q = rms_rows(&q, &g.get(&p("attn_q_norm.weight"))?, c.eps);
            let k = matmul_t(&xn, t, &g.get(&p("attn_k.weight"))?, c.kv_heads * c.head_dim);
            let mut k = rms_rows(&k, &g.get(&p("attn_k_norm.weight"))?, c.eps);
            let v = matmul_t(&xn, t, &g.get(&p("attn_v.weight"))?, c.kv_heads * c.head_dim);
            rope(&c, &mut q, c.heads);
            rope(&c, &mut k, c.kv_heads);
            // causal attention; query head h reads KV head h / (c.heads / c.kv_heads) (mul_mat broadcast)
            let scale = 1.0 / (c.head_dim as f32).sqrt();
            let mut att = vec![0f32; t * c.heads * c.head_dim];
            for h in 0..c.heads {
                let kvh = h / (c.heads / c.kv_heads);
                for i in 0..t {
                    let qi = &q[(i * c.heads + h) * c.head_dim..(i * c.heads + h + 1) * c.head_dim];
                    let scores: Vec<f32> = (0..=i)
                        .map(|j| qi.iter().zip(&k[(j * c.kv_heads + kvh) * c.head_dim..]).map(|(a, b)| a * b).sum::<f32>() * scale)
                        .collect();
                    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                    let z: f32 = e.iter().sum();
                    let out = &mut att[(i * c.heads + h) * c.head_dim..(i * c.heads + h + 1) * c.head_dim];
                    for (j, ej) in e.iter().enumerate() {
                        let vj = &v[(j * c.kv_heads + kvh) * c.head_dim..(j * c.kv_heads + kvh + 1) * c.head_dim];
                        out.iter_mut().zip(vj).for_each(|(o, vv)| *o += ej / z * vv);
                    }
                }
            }
            let gated: Vec<f32> = att.iter().zip(&gate).map(|(a, gg)| a / (1.0 + (-gg).exp())).collect();
            // canonical K and V: [kv_heads, tokens, head_dim], keys rotated
            let canon = |src: &[f32]| -> Vec<f32> {
                (0..c.kv_heads).flat_map(|h| (0..t).flat_map(move |i| src[(i * c.kv_heads + h) * c.head_dim..(i * c.kv_heads + h + 1) * c.head_dim].to_vec())).collect()
            };
            fields.insert(format!("K.{il}"), canon(&k));
            fields.insert(format!("V.{il}"), canon(&v));
            matmul_t(&gated, t, &g.get(&p("attn_output.weight"))?, c.hidden)
        };
        x.iter_mut().zip(&mixed).for_each(|(a, b)| *a += b);
        let f = rms_rows(&x, &g.get(&p("post_attention_norm.weight"))?, c.eps);
        let gate = matmul_t(&f, t, &g.get(&p("ffn_gate.weight"))?, c.ffn);
        let up = matmul_t(&f, t, &g.get(&p("ffn_up.weight"))?, c.ffn);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| silu(*a) * b).collect();
        let down = matmul_t(&act, t, &g.get(&p("ffn_down.weight"))?, c.hidden);
        x.iter_mut().zip(&down).for_each(|(a, b)| *a += b);
    }
    fields.insert("H".into(), x[(t - 1) * c.hidden..].to_vec());
    Ok(fields)
}

/// The first token from a final hidden state: argmax of the output head over the final-normed `h`.
/// The head (vocabulary × width) multiplies quantized, as stored, rather than dequantized to f32.
pub fn first_token(gguf: &str, h: &[f32]) -> Result<u32, String> {
    use candle_core::quantized::QMatMul;
    use candle_core::{Module, Tensor};
    let mut file = std::fs::File::open(gguf).map_err(|e| format!("{gguf}: {e}"))?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;
    let c = Config::read(&content)?;
    let mut g = Gguf { content, file };
    let hn = rms_rows(h, &g.get("output_norm.weight")?, c.eps);
    let head = if g.content.tensor_infos.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" };
    let q = g.content.tensor(&mut g.file, head, &Device::Cpu).map_err(|e| format!("{head}: {e}"))?;
    let logits = QMatMul::from_qtensor(q)
        .and_then(|m| m.forward(&Tensor::from_vec(hn, (1, c.hidden), &Device::Cpu)?))
        .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
        .map_err(|e| format!("{head}: {e}"))?;
    Ok(logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, v)| if *v > best.1 { (i, *v) } else { best }).0 as u32)
}
