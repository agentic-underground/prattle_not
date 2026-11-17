//! Upstream's 4B → 27B state translator (its "final16 dual-branch" bridge) in Rust — docs/BRIDGE.md
//! slice 11e, ported from upstream `_reference/architecture.py` and `_reference/bridge.py` and checked
//! stage by stage against the oracle's own run (tests/translator_oracle.rs).
//!
//! Two branches share one frozen base. The PARENT reads the 4B's state and writes the 27B's recurrent
//! state (C, R) and final hidden state (H); the K/V BRANCH — the same network with 36 tensors
//! overridden — writes the 27B's attention keys and values:
//!
//!   persistent encoder   the 4B's C, R and H → 49 tokens (`memory`) and their mean `g`
//!   chunk encoder        the 4B's de-rotated K, V per position → `content`, modulated by `g`
//!   position router      `content` at 4B positions → features at 27B positions (the alignment)
//!   K/V head             features → the 27B's K, V per position          (K/V branch)
//!   query decoder        97 learned queries attend to `memory` → C, R, H  (parent)
//!
//! The source state is rounded to bf16 on entry and the outputs on exit, where upstream casts. Measured
//! against the oracle: every stage within 6.5e-7, every one of the 129 output fields within 1.2e-4.

use std::collections::{BTreeMap, HashMap};

use candle_core::Device;
use candle_core::quantized::gguf_file;

use crate::gdn::matmul_t;

const EPS: f32 = 1e-6;
/// Width of the chunk encoder and the router's content; the persistent width; its heads.
const KV_WIDTH: usize = 2560;
const P_WIDTH: usize = 1152;
const P_HEADS: usize = 12;
/// architecture_v2.json `position_router`.
const R_WIDTH: usize = 256;
const SOURCE_RADIUS: i64 = 8;
const TARGET_RADIUS: i64 = 16;
const POOLS: [usize; 2] = [16, 64];
const OFFSET_LIMIT: f32 = 4.0;
const PRIOR_SIGMA: f32 = 1.5;
/// geometry.json — both models: 4 KV heads of 256, RoPE over the first 64 dims.
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const ROT: usize = 64;
/// The 27B's recurrent state: 48 heads of 128 × 128, built at rank 96.
const T_REC_HEADS: usize = 48;
const REC: usize = 128;
const RANK: usize = 96;

fn bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// RMS-normalise each `width`-row, then scale by `w` (or by 1).
fn rms(x: &[f32], width: usize, w: Option<&[f32]>) -> Vec<f32> {
    x.chunks(width)
        .flat_map(|row| {
            let r = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + EPS).sqrt();
            row.iter().enumerate().map(move |(i, v)| v * r * w.map_or(1.0, |w| w[i])).collect::<Vec<_>>()
        })
        .collect()
}

/// The 4B's canonical state and the numeric alignment the translator reads.
pub struct Source {
    pub fields: BTreeMap<String, Vec<f32>>,
    pub source_length: usize,
    pub target_length: usize,
    /// One source-position center per 27B position, monotonic.
    pub centers: Vec<f32>,
}

impl Source {
    /// A field as the translator reads it: rounded to bf16, as upstream's `historical_source` does.
    fn field(&self, name: &str) -> Vec<f32> {
        self.fields.get(name).unwrap_or_else(|| panic!("the source state has no {name}")).iter().map(|v| bf16(*v)).collect()
    }
    fn layers(&self, prefix: &str) -> Vec<usize> {
        let mut v: Vec<usize> = self.fields.keys().filter_map(|k| k.strip_prefix(prefix)?.parse().ok()).collect();
        v.sort();
        v
    }
}

/// The persistent encoder's output: `memory` `[49, 1152]` and its mean `g` `[1152]`.
pub struct Persistent {
    pub memory: Vec<f32>,
    pub g: Vec<f32>,
}

pub struct Translator {
    w: HashMap<String, Vec<f32>>,
}

impl Translator {
    /// Every tensor of upstream's bridge GGUF, as f32 (all 262 are stored F32).
    pub fn load(gguf: &str) -> Result<Translator, String> {
        let mut file = std::fs::File::open(gguf).map_err(|e| format!("{gguf}: {e}"))?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;
        let mut w = HashMap::new();
        for name in content.tensor_infos.keys() {
            let v = content
                .tensor(&mut file, name, &Device::Cpu)
                .and_then(|t| t.dequantize(&Device::Cpu))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| format!("{name}: {e}"))?;
            w.insert(name.clone(), v);
        }
        Ok(Translator { w })
    }

    fn get(&self, name: &str) -> &[f32] {
        self.w.get(name).unwrap_or_else(|| panic!("the bridge has no {name}"))
    }

    /// A base tensor for `branch`: the K/V branch's override where it has one, else the shared one.
    fn base(&self, branch: &str, name: &str) -> &[f32] {
        let own = format!("{branch}.base.{name}");
        self.w.get(&own).map(|v| v.as_slice()).unwrap_or_else(|| self.get(&format!("parent.base.{name}")))
    }

    /// `x` `[rows, in]` through the linear map `name` of `branch` (`[out, in]`).
    fn lin(&self, branch: &str, name: &str, x: &[f32], rows: usize) -> Vec<f32> {
        let w = self.base(branch, &format!("{name}.weight"));
        let inp = x.len() / rows;
        matmul_t(x, rows, w, w.len() / inp)
    }

    fn swiglu(&self, branch: &str, prefix: &str, x: &[f32], rows: usize) -> Vec<f32> {
        let gate = self.lin(branch, &format!("{prefix}.gate"), x, rows);
        let up = self.lin(branch, &format!("{prefix}.up"), x, rows);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| silu(*a) * b).collect();
        self.lin(branch, &format!("{prefix}.down"), &act, rows)
    }

    /// Multi-head attention (no mask): `query` `[n, width]` attends to `memory` `[m, width]`.
    fn attention(&self, prefix: &str, query: &[f32], memory: &[f32]) -> Vec<f32> {
        let (n, m, hd) = (query.len() / P_WIDTH, memory.len() / P_WIDTH, P_WIDTH / P_HEADS);
        let q = self.lin("parent", &format!("{prefix}.q"), query, n);
        let k = self.lin("parent", &format!("{prefix}.k"), memory, m);
        let v = self.lin("parent", &format!("{prefix}.v"), memory, m);
        let mut out = vec![0f32; n * P_WIDTH];
        let scale = 1.0 / (hd as f32).sqrt();
        for h in 0..P_HEADS {
            for i in 0..n {
                let qi = &q[i * P_WIDTH + h * hd..i * P_WIDTH + (h + 1) * hd];
                let s: Vec<f32> =
                    (0..m).map(|j| qi.iter().zip(&k[j * P_WIDTH + h * hd..]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                let top = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = s.iter().map(|x| (x - top).exp()).collect();
                let z: f32 = e.iter().sum();
                for (j, ej) in e.iter().enumerate() {
                    for d in 0..hd {
                        out[i * P_WIDTH + h * hd + d] += ej / z * v[j * P_WIDTH + h * hd + d];
                    }
                }
            }
        }
        self.lin("parent", &format!("{prefix}.o"), &out, n)
    }

    /// (x − mean) / std for source `layer` and family (0: K or C, 1: V or R).
    fn normalize(&self, x: &mut [f32], layer: usize, family: usize) {
        let (mean, std) = (self.get("parent.base.source_mean")[layer * 2 + family], self.get("parent.base.source_std")[layer * 2 + family]);
        x.iter_mut().for_each(|v| *v = (*v - mean) / std.max(1e-8));
    }

    /// x · gain · std + mean for target `layer` and family.
    fn denormalize(&self, x: &mut [f32], layer: usize, family: usize) {
        let i = layer * 2 + family;
        let (gain, std, mean) =
            (self.get("parent.base.output_gain")[i], self.get("parent.base.target_std")[i], self.get("parent.base.target_mean")[i]);
        x.iter_mut().for_each(|v| *v = *v * gain * std + mean);
    }

    pub fn persistent(&self, s: &Source) -> Persistent {
        let mut tokens = Vec::new();
        for i in s.layers("R.") {
            let mut conv = s.field(&format!("C.{i}"));
            self.normalize(&mut conv, i, 0);
            let basis = self.lin("parent", "conv_input_basis", &conv, 1);
            tokens.extend(self.lin("parent", "conv_input", &basis, 1));
            let mut rec = s.field(&format!("R.{i}"));
            self.normalize(&mut rec, i, 1);
            let heads = rec.len() / (REC * REC);
            let z = self.lin("parent", "rec_input_basis", &rec, heads);
            tokens.extend(self.lin("parent", "rec_input", &z, 1));
        }
        let ts = self.get("parent.base.terminal_stats");
        let h: Vec<f32> = s.field("H").iter().map(|v| (v - ts[0]) / ts[1].max(1e-8)).collect();
        tokens.extend(self.lin("parent", "terminal_input", &h, 1));
        let mut x: Vec<f32> = tokens.iter().zip(self.get("parent.base.source_identity")).map(|(a, b)| a + b).collect();
        let n = x.len() / P_WIDTH;
        for b in 0..2 {
            let p = |s: &str| format!("state_encoder.{b}.{s}");
            let y = rms(&x, P_WIDTH, Some(self.base("parent", &p("norm1.weight"))));
            let a = self.attention(&format!("state_encoder.{b}.attn"), &y, &y);
            x.iter_mut().zip(&a).for_each(|(x, a)| *x += 0.5 * a);
            let y = rms(&x, P_WIDTH, Some(self.base("parent", &p("norm2.weight"))));
            let f = self.swiglu("parent", &p("ffn"), &y, n);
            x.iter_mut().zip(&f).for_each(|(x, f)| *x += 0.5 * f);
        }
        let memory = rms(&x, P_WIDTH, Some(self.base("parent", "state_norm.weight")));
        let g = (0..P_WIDTH).map(|d| (0..n).map(|t| memory[t * P_WIDTH + d]).sum::<f32>() / n as f32).collect();
        Persistent { memory, g }
    }

    /// The chunk encoder: each source position's de-rotated, normalised K and V from every
    /// attention layer → `[source_length, 2560]`.
    pub fn content(&self, s: &Source, p: &Persistent, branch: &str) -> Vec<f32> {
        let n = s.source_length;
        let modulation = self.lin("parent", "modulation", &p.g, 1);
        let (scale, shift) = modulation.split_at(KV_WIDTH);
        let inv = self.get("parent.base.source_rope.inv_freq");
        let layers = s.layers("K.");
        let width = layers.len() * 2 * KV_HEADS * HEAD_DIM;
        let mut flat = vec![0f32; n * width];
        for (l, &i) in layers.iter().enumerate() {
            let (k, v) = (s.field(&format!("K.{i}")), s.field(&format!("V.{i}")));
            for t in 0..n {
                for h in 0..KV_HEADS {
                    let src = (h * n + t) * HEAD_DIM;
                    let mut kk = derotate(&k[src..src + HEAD_DIM], t, inv);
                    self.normalize(&mut kk, i, 0);
                    let mut vv = v[src..src + HEAD_DIM].to_vec();
                    self.normalize(&mut vv, i, 1);
                    let base = t * width + l * 2 * KV_HEADS * HEAD_DIM;
                    flat[base + h * HEAD_DIM..base + (h + 1) * HEAD_DIM].copy_from_slice(&kk);
                    flat[base + (KV_HEADS + h) * HEAD_DIM..base + (KV_HEADS + h + 1) * HEAD_DIM].copy_from_slice(&vv);
                }
            }
        }
        let mut x = self.lin(branch, "kv_input", &flat, n);
        for b in 0..4 {
            let y = rms(&x, KV_WIDTH, Some(self.base(branch, &format!("core.{b}.norm.weight"))));
            let y: Vec<f32> = y
                .iter()
                .enumerate()
                .map(|(i, v)| v * (1.0 + 0.1 * scale[i % KV_WIDTH].tanh()) + 0.1 * shift[i % KV_WIDTH])
                .collect();
            let f = self.swiglu(branch, &format!("core.{b}.ffn"), &y, n);
            x.iter_mut().zip(&f).for_each(|(x, f)| *x += 0.25 * f);
        }
        rms(&x, KV_WIDTH, Some(self.base(branch, "kv_norm.weight")))
    }

    /// The router's bank: every position, ragged means over 16 and 64 positions, and one global
    /// token. Returns the bank and where each pool starts.
    fn pool_bank(&self, branch: &str, z: &[f32], g: &[f32]) -> (Vec<f32>, [usize; 2], usize) {
        let n = z.len() / R_WIDTH;
        let mut bank = z.to_vec();
        let mut starts = [0; 2];
        let mut cursor = n;
        for (k, &scale) in POOLS.iter().enumerate() {
            starts[k] = cursor;
            for grp in 0..n.div_ceil(scale) {
                let (a, b) = (grp * scale, ((grp + 1) * scale).min(n));
                bank.extend((0..R_WIDTH).map(|d| (a..b).map(|t| z[t * R_WIDTH + d]).sum::<f32>() / (b - a) as f32));
                cursor += 1;
            }
        }
        let global = self.lin(branch, "position_router.global_projection", g, 1);
        bank.extend((0..R_WIDTH).map(|d| global[d] + (0..n).map(|t| z[t * R_WIDTH + d]).sum::<f32>() / n as f32));
        (bank, starts, cursor)
    }

    /// Gathered attention for one query over a window of the bank plus the pools and global token.
    #[allow(clippy::too_many_arguments)]
    fn gathered(q: &[f32], keys: &[f32], values: &[f32], center: f32, point: f32, radius: i64, n: usize, bank: ([usize; 2], usize, &[f32])) -> Vec<f32> {
        let (starts, global, bias_pool) = bank;
        let base = center.floor() as i64;
        let mut idx = Vec::new();
        let mut bias = Vec::new();
        for o in -radius..=radius {
            let i = base + o;
            if i >= 0 && (i as usize) < n {
                idx.push(i as usize);
                bias.push(-((i as f32 - point).powi(2)) / (2.0 * PRIOR_SIGMA * PRIOR_SIGMA));
            }
        }
        for (k, &scale) in POOLS.iter().enumerate() {
            idx.push(base as usize / scale + starts[k]);
            bias.push(bias_pool[k]);
        }
        idx.push(global);
        bias.push(bias_pool[2]);
        let r = (R_WIDTH as f32).sqrt();
        let logits: Vec<f32> =
            idx.iter().zip(&bias).map(|(&i, b)| q.iter().zip(&keys[i * R_WIDTH..]).map(|(a, k)| a * k).sum::<f32>() / r + b).collect();
        let top = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|l| (l - top).exp()).collect();
        let z: f32 = e.iter().sum();
        let mut out = vec![0f32; R_WIDTH];
        for (&i, ei) in idx.iter().zip(&e) {
            out.iter_mut().zip(&values[i * R_WIDTH..]).for_each(|(o, v)| *o += ei / z * v);
        }
        out
    }

    /// The position router: `content` at 4B positions → features at 27B positions `[target_length, 2560]`.
    pub fn route(&self, s: &Source, p: &Persistent, content: &[f32], branch: &str) -> Vec<f32> {
        let (n, m) = (s.source_length, s.target_length);
        let pb = self.base(branch, "position_router.pool_bias");
        let lin = |name: &str, x: &[f32], rows: usize| self.lin(branch, &format!("position_router.{name}"), x, rows);
        let norm = |name: &str, x: &[f32]| rms(x, R_WIDTH, Some(self.base(branch, &format!("position_router.{name}.weight"))));
        let z = norm("source_norm", &lin("down", content, n));
        // mixer: each source position attends to its neighbours and the pools
        let (bank, starts, global) = self.pool_bank(branch, &z, &p.g);
        let rows = bank.len() / R_WIDTH;
        let (mk, mv) = (lin("mixer.k", &bank, rows), lin("mixer.v", &bank, rows));
        let mq = lin("mixer.q", &z, n);
        let mut mixed_v = Vec::with_capacity(n * R_WIDTH);
        for t in 0..n {
            mixed_v.extend(Self::gathered(&mq[t * R_WIDTH..(t + 1) * R_WIDTH], &mk, &mv, t as f32, t as f32, SOURCE_RADIUS, n, (starts, global, pb)));
        }
        let mo = lin("mixer.o", &mixed_v, n);
        let mixed: Vec<f32> = z.iter().zip(&mo).map(|(a, b)| a + 0.25 * b).collect();
        let z = norm("mixed_norm", &mixed);
        // route: each target position queries the source around its aligned center
        let (bank, starts, global) = self.pool_bank(branch, &z, &p.g);
        let rows = bank.len() / R_WIDTH;
        let (rk, rv) = (lin("route.k", &bank, rows), lin("route.v", &bank, rows));
        let anchor: Vec<usize> = s.centers.iter().map(|c| c.round_ties_even() as usize).collect();
        let feats: Vec<f32> = (0..m)
            .flat_map(|j| {
                let pp = j as f32;
                [
                    s.centers[j] / (n.max(2) - 1) as f32,
                    pp / (m.max(2) - 1) as f32,
                    pp.sin(),
                    pp.cos(),
                    (pp / 16.0).sin(),
                    (pp / 16.0).cos(),
                    (pp / 256.0).sin(),
                    (pp / 256.0).cos(),
                    (n as f32).ln_1p() / 16.0,
                    (m as f32).ln_1p() / 16.0,
                ]
            })
            .collect();
        let pos = lin("position", &feats, m);
        let glob = lin("global_projection", &p.g, 1);
        let query_in: Vec<f32> = (0..m * R_WIDTH).map(|i| z[anchor[i / R_WIDTH] * R_WIDTH + i % R_WIDTH] + pos[i] + glob[i % R_WIDTH]).collect();
        let query = norm("query_norm", &query_in);
        let off = lin("offset", &query, m);
        let mut corrected = Vec::with_capacity(m);
        let mut running = f32::NEG_INFINITY;
        for j in 0..m {
            running = running.max(s.centers[j] + OFFSET_LIMIT * off[j].tanh());
            corrected.push(running.clamp(0.0, (n - 1) as f32));
        }
        let rq = lin("route.q", &query, m);
        let mut routed = Vec::with_capacity(m * R_WIDTH);
        for j in 0..m {
            routed.extend(Self::gathered(&rq[j * R_WIDTH..(j + 1) * R_WIDTH], &rk, &rv, s.centers[j], corrected[j], TARGET_RADIUS, n, (starts, global, pb)));
        }
        let ro = lin("route.o", &routed, m);
        let diff: Vec<f32> = (0..m * R_WIDTH).map(|i| ro[i] - z[anchor[i / R_WIDTH] * R_WIDTH + i % R_WIDTH]).collect();
        let up = lin("up", &diff, m);
        (0..m * KV_WIDTH).map(|i| content[anchor[i / KV_WIDTH] * KV_WIDTH + i % KV_WIDTH] + 0.25 * up[i]).collect()
    }

    /// The parent's query decoder → `[97, 1152]`: 48 (C, R) query pairs and one for H.
    pub fn queries(&self, p: &Persistent, routed: &[f32], target_length: usize) -> Vec<f32> {
        let last = &routed[(target_length - 1) * KV_WIDTH..target_length * KV_WIDTH];
        let fin = self.lin("parent", "final_kv_condition", last, 1);
        let memory: Vec<f32> = p.memory.iter().enumerate().map(|(i, v)| v + fin[i % P_WIDTH]).collect();
        let tq = self.get("parent.base.target_queries");
        let queries: Vec<f32> = tq.iter().enumerate().map(|(i, v)| v + p.g[i % P_WIDTH] + fin[i % P_WIDTH]).collect();
        let nq = queries.len() / P_WIDTH;
        let qn = rms(&queries, P_WIDTH, Some(self.get("parent.base.query_norm.weight")));
        let mn = rms(&memory, P_WIDTH, Some(self.get("parent.base.memory_norm.weight")));
        let cross = self.attention("cross", &qn, &mn);
        let mut q: Vec<f32> = queries.iter().zip(&cross).map(|(a, b)| a + b).collect();
        let f = self.swiglu("parent", "query_ffn", &rms(&q, P_WIDTH, Some(self.get("parent.base.query_ffn_norm.weight"))), nq);
        q.iter_mut().zip(&f).for_each(|(a, b)| *a += 0.5 * b);
        let mut q = rms(&q, P_WIDTH, Some(self.get("parent.base.decoder_norm.weight")));
        // upstream's layer-identity hook on the decoder norm: the 96 (C, R) queries are re-centred
        // on their own learned identities
        let n = 2 * T_REC_HEADS;
        let identity = rms(&tq[..n * P_WIDTH], P_WIDTH, None);
        let mixed: Vec<f32> = q[..n * P_WIDTH].iter().zip(&identity).map(|(a, b)| a + b).collect();
        q[..n * P_WIDTH].copy_from_slice(&rms(&mixed, P_WIDTH, None));
        q
    }

    /// The whole translation: the 27B's canonical state, field by field.
    pub fn translate(&self, s: &Source) -> BTreeMap<String, Vec<f32>> {
        let mut out = BTreeMap::new();
        let p = self.persistent(s);
        let m = s.target_length;
        // the parent: C, R and H
        let routed = self.route(s, &p, &self.content(s, &p, "parent"), "parent");
        let q = self.queries(&p, &routed, m);
        let row = |r: usize| &q[r * P_WIDTH..(r + 1) * P_WIDTH];
        let target_linear: Vec<usize> = (0..64).filter(|i| (i + 1) % 4 != 0).collect();
        let (hb_g, hb_b, hb_u, hb_v, hb_scale) = (
            self.get("parent.head_basis.g.weight"),
            self.get("parent.head_basis.g.bias"),
            self.get("parent.head_basis.U"),
            self.get("parent.head_basis.V"),
            self.get("parent.head_basis.scale"),
        );
        for (j, &i) in target_linear.iter().enumerate() {
            let mut conv = self.lin("parent", "conv_output_basis", &self.lin("parent", "conv_output", row(2 * j), 1), 1);
            self.denormalize(&mut conv, i, 0);
            let tmean = self.get("parent.base.target_mean");
            let pm = self.get(&format!("population_means.C.{i}"));
            let conv: Vec<f32> = conv.iter().zip(pm).map(|(c, pm)| bf16(bf16(*c) - tmean[i * 2] + pm)).collect();
            out.insert(format!("C.{i}"), conv);

            let z = self.lin("parent", "rec_output", row(2 * j + 1), 1); // [48, 128]
            let left = self.lin("parent", "rec_output_basis.left", &z, T_REC_HEADS); // [48, 128·96]
            let right = self.lin("parent", "rec_output_basis.right", &z, T_REC_HEADS); // [48, 96·128]
            let factor = 3.0 / (RANK as f32).sqrt();
            let coeff: Vec<f32> = matmul_t(&z, T_REC_HEADS, hb_g, REC).iter().enumerate().map(|(k, v)| 1.0 + 0.1 * (v + hb_b[k % REC]).tanh()).collect();
            let mut rec = vec![0f32; T_REC_HEADS * REC * REC];
            for h in 0..T_REC_HEADS {
                let (a, b) = (&left[h * REC * RANK..(h + 1) * REC * RANK], &right[h * RANK * REC..(h + 1) * RANK * REC]);
                let r = &mut rec[h * REC * REC..(h + 1) * REC * REC];
                for x in 0..REC {
                    for k in 0..RANK {
                        let av = a[x * RANK + k] * factor;
                        r[x * REC..(x + 1) * REC].iter_mut().zip(&b[k * REC..(k + 1) * REC]).for_each(|(o, bv)| *o += av * bv);
                    }
                }
            }
            self.denormalize(&mut rec, i, 1);
            let pm = self.get(&format!("population_means.R.{i}"));
            rec.iter_mut().zip(pm).for_each(|(r, pm)| *r = *r - tmean[i * 2 + 1] + pm);
            // the head-basis correction: (U · diag(coefficients)) V, per layer and head, scaled
            for h in 0..T_REC_HEADS {
                let base = (j * T_REC_HEADS + h) * REC * REC;
                let (u, v) = (&hb_u[base..base + REC * REC], &hb_v[base..base + REC * REC]);
                let c = &coeff[h * REC..(h + 1) * REC];
                let r = &mut rec[h * REC * REC..(h + 1) * REC * REC];
                for x in 0..REC {
                    for k in 0..REC {
                        let uc = u[x * REC + k] * c[k] * hb_scale[j];
                        r[x * REC..(x + 1) * REC].iter_mut().zip(&v[k * REC..(k + 1) * REC]).for_each(|(o, vv)| *o += uc * vv);
                    }
                }
            }
            out.insert(format!("R.{i}"), rec);
        }
        let ts = self.get("parent.base.terminal_stats");
        let gain = self.get("parent.base.terminal_gain")[0];
        let t = self.lin("parent", "terminal_output", row(2 * T_REC_HEADS), 1);
        let h: Vec<f32> = t.iter().map(|v| bf16(v * gain * ts[3] + ts[2])).collect();
        let normed = rms(&h, h.len(), None);
        let down = matmul_t(&normed, 1, self.get("h.down.weight"), 16);
        let up = matmul_t(&down, 1, self.get("h.up.weight"), h.len());
        out.insert("H".into(), h.iter().zip(&up).map(|(a, b)| bf16(a + b)).collect());

        // the K/V branch: K and V at every 27B position
        let features = self.route(s, &p, &self.content(s, &p, "kv"), "kv");
        let joined = self.lin("kv", "kv_output", &features, m); // [m, 8 layers, 2, 4 heads, 256]
        let inv = self.get("parent.base.target_rope.inv_freq");
        let target_attention: Vec<usize> = (0..64).filter(|i| (i + 1) % 4 == 0).collect();
        let per = 2 * KV_HEADS * HEAD_DIM;
        for (j, &i) in target_attention.iter().enumerate() {
            let (mut k, mut v) = (vec![0f32; KV_HEADS * m * HEAD_DIM], vec![0f32; KV_HEADS * m * HEAD_DIM]);
            for t in 0..m {
                for h in 0..KV_HEADS {
                    let src = t * target_attention.len() * per + j * per;
                    let dst = (h * m + t) * HEAD_DIM;
                    let mut kk = joined[src + h * HEAD_DIM..src + (h + 1) * HEAD_DIM].to_vec();
                    let mut vv = joined[src + (KV_HEADS + h) * HEAD_DIM..src + (KV_HEADS + h + 1) * HEAD_DIM].to_vec();
                    self.denormalize(&mut kk, i, 0);
                    self.denormalize(&mut vv, i, 1);
                    let kk: Vec<f32> = kk.iter().map(|x| bf16(*x)).collect();
                    k[dst..dst + HEAD_DIM].copy_from_slice(&rotate_bf16(&kk, t, inv));
                    v[dst..dst + HEAD_DIM].copy_from_slice(&vv.iter().map(|x| bf16(*x)).collect::<Vec<_>>());
                }
            }
            out.insert(format!("K.{i}"), k);
            out.insert(format!("V.{i}"), v);
        }
        out
    }
}

/// cos and sin of RoPE angle `j` at position `t`, rounded to bf16 as upstream casts them.
fn angle(t: usize, j: usize, inv: &[f32]) -> (f32, f32) {
    let a = t as f32 * inv[j % (ROT / 2)];
    (bf16(a.cos()), bf16(a.sin()))
}

/// Upstream `Rope.forward(inverse=True)` on one bf16 head row: exact division by c² + s².
fn derotate(x: &[f32], t: usize, inv: &[f32]) -> Vec<f32> {
    let half = ROT / 2;
    let mut y = x.to_vec();
    for i in 0..ROT {
        let rot = if i < half { -x[i + half] } else { x[i - half] };
        let (c, s) = angle(t, i, inv);
        y[i] = (x[i] * c - rot * s) / (c * c + s * s);
    }
    y
}

/// Upstream `Rope.forward` on one bf16 head row, in bf16 arithmetic: each product and the sum round.
fn rotate_bf16(x: &[f32], t: usize, inv: &[f32]) -> Vec<f32> {
    let half = ROT / 2;
    let mut y = x.to_vec();
    for i in 0..ROT {
        let rot = if i < half { -x[i + half] } else { x[i - half] };
        let (c, s) = angle(t, i, inv);
        y[i] = bf16(bf16(x[i] * c) + bf16(rot * s));
    }
    y
}
