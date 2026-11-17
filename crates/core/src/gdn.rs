//! The Qwen3.5 hybrid's Gated DeltaNet layer — docs/BRIDGE.md slice 11e, checked against the oracle.
//!
//! Computed in the layout the pinned GGUF implies (llama.cpp: value heads TILED over key heads, S
//! stored [head, value, key]) and converted to the canonical layout upstream's ABI exports (value
//! heads GROUPED, R as [head, key, value]) by `canonical_r` / `canonical_c`, ported from upstream
//! `native/layout.h`.

/// The geometry of one GDN layer.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub key_heads: usize,
    pub value_heads: usize,
    pub dim: usize,
    pub conv: usize,
}

impl Geometry {
    pub fn channels(&self) -> usize {
        2 * self.key_heads * self.dim + self.value_heads * self.dim
    }
}

/// One GDN layer's weights, dequantized to f32. `qkv` is `[channels, hidden]` row-major, `alpha` and
/// `beta` are `[value_heads, hidden]`, `conv` is `[channels, conv]`.
pub struct Weights {
    pub norm: Vec<f32>,
    pub qkv: Vec<f32>,
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
    pub a: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub conv: Vec<f32>,
    pub eps: f32,
}

/// The persistent state a prefill leaves, in the CANONICAL layout: `r` is
/// `[value_heads, dim(key), dim(value)]`, `c` is `[channels, conv]` (raw projections of the last
/// `conv` tokens, oldest first).
pub struct State {
    pub r: Vec<f32>,
    pub c: Vec<f32>,
}

// ---- canonical layout, from upstream native/layout.h ---------------------------------------------

/// The GGUF tiles value heads over key heads; canonical groups them.
pub fn v_head(h: usize, key_heads: usize, value_heads: usize) -> usize {
    let ratio = value_heads / key_heads;
    (h % ratio) * key_heads + h / ratio
}

/// The native channel holding canonical channel `c` (q and k channels are unchanged).
pub fn c_channel(c: usize, g: &Geometry) -> usize {
    let base = 2 * g.key_heads * g.dim;
    if c < base { c } else { base + v_head((c - base) / g.dim, g.key_heads, g.value_heads) * g.dim + (c - base) % g.dim }
}

/// The native index of canonical recurrent element `i` (native S is [head, value, key]).
pub fn r_index(i: usize, g: &Geometry) -> usize {
    let d = g.dim;
    v_head(i / (d * d), g.key_heads, g.value_heads) * d * d + (i % d) * d + (i / d) % d
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn softplus(v: f32) -> f32 {
    if v > 20.0 { v } else { v.exp().ln_1p() }
}

/// ggml_l2_norm: x / max(‖x‖, eps), per head.
fn l2_normalize(v: &mut [f32], eps: f32) {
    let n = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(eps);
    v.iter_mut().for_each(|a| *a /= n);
}

/// `[rows, cols] · [out, cols]ᵀ` → `[rows, out]`, through candle's matmul.
pub fn matmul_t(x: &[f32], rows: usize, w: &[f32], out: usize) -> Vec<f32> {
    let cols = x.len() / rows;
    let xt = candle_core::Tensor::from_slice(x, (rows, cols), &candle_core::Device::Cpu).unwrap();
    let wt = candle_core::Tensor::from_slice(w, (out, cols), &candle_core::Device::Cpu).unwrap();
    xt.matmul(&wt.t().unwrap()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// RMS-normalise each `width`-row of `x` and scale by `w`.
pub fn rms_rows(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let width = w.len();
    x.chunks(width)
        .flat_map(|row| {
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + eps).sqrt();
            row.iter().zip(w).map(move |(v, n)| v / rms * n).collect::<Vec<_>>()
        })
        .collect()
}

/// Prefill the layer over `x` (`[tokens, hidden]` — the layer's INPUT, before its norm), from a
/// zero state, one token at a time (the chunked kernel the oracle uses is the same recurrence).
pub fn prefill(g: &Geometry, w: &Weights, x: &[f32], hidden: usize) -> State {
    prefill_with_output(g, w, x, hidden).0
}

/// As `prefill`, also returning the delta rule's output per token, `[tokens, value_heads, dim]` in
/// the NATIVE head order (before the gated norm and the output projection).
pub fn prefill_with_output(g: &Geometry, w: &Weights, x: &[f32], hidden: usize) -> (State, Vec<f32>) {
    let (d, ch, kh, vh) = (g.dim, g.channels(), g.key_heads, g.value_heads);
    let tokens = x.len() / hidden;
    let xn = rms_rows(x, &w.norm, w.eps);
    let raw_all = matmul_t(&xn, tokens, &w.qkv, ch); // pre-conv projections, native channels
    let alpha_all = matmul_t(&xn, tokens, &w.alpha, vh);
    let beta_all = matmul_t(&xn, tokens, &w.beta, vh);
    let raw: Vec<&[f32]> = raw_all.chunks(ch).collect();
    let scale = 1.0 / (d as f32).sqrt();
    let mut s = vec![0f32; vh * d * d]; // native [head, value, key]
    let mut out = vec![0f32; tokens * vh * d];
    for t in 0..tokens {
        let (alpha, beta) = (&alpha_all[t * vh..(t + 1) * vh], &beta_all[t * vh..(t + 1) * vh]);
        // causal depthwise conv over the last `conv` raw rows (zeros before the start), then SiLU
        let mut mixed = vec![0f32; ch];
        for (c, m) in mixed.iter_mut().enumerate() {
            let mut acc = 0f32;
            for j in 0..g.conv {
                let back = g.conv - 1 - j; // j = conv-1 is the current token
                if t >= back {
                    acc += w.conv[c * g.conv + j] * raw[t - back][c];
                }
            }
            *m = silu(acc);
        }
        let (q_all, rest) = mixed.split_at_mut(kh * d);
        let (k_all, v_all) = rest.split_at_mut(kh * d);
        q_all.chunks_mut(d).for_each(|q| l2_normalize(q, w.eps));
        k_all.chunks_mut(d).for_each(|k| l2_normalize(k, w.eps));

        for h in 0..vh {
            let k = &k_all[(h % kh) * d..(h % kh + 1) * d]; // ggml broadcasts key heads modularly
            let q = &q_all[(h % kh) * d..(h % kh + 1) * d];
            let v = &v_all[h * d..(h + 1) * d];
            let decay = (softplus(alpha[h] + w.dt_bias[h]) * w.a[h]).exp();
            let b = 1.0 / (1.0 + (-beta[h]).exp());
            let sh = &mut s[h * d * d..(h + 1) * d * d];
            sh.iter_mut().for_each(|e| *e *= decay);
            for j in 0..d {
                let srow = &mut sh[j * d..(j + 1) * d];
                let sk: f32 = srow.iter().zip(k).map(|(a, b)| a * b).sum();
                let delta = b * (v[j] - sk);
                srow.iter_mut().zip(k).for_each(|(e, ki)| *e += ki * delta);
                out[(t * vh + h) * d + j] = srow.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
        }
    }
    let r = (0..vh * d * d).map(|i| s[r_index(i, g)]).collect();
    let mut c = vec![0f32; ch * g.conv];
    for canon in 0..ch {
        let n = c_channel(canon, g);
        for j in 0..g.conv {
            let back = g.conv - 1 - j;
            if tokens > back {
                c[canon * g.conv + j] = raw[tokens - 1 - back][n];
            }
        }
    }
    (State { r, c }, out)
}
