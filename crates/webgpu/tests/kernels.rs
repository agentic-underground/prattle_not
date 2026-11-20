//! Each WGSL kernel against candle's CPU arithmetic on the same inputs: the engine is only as right
//! as the reference it matches. Shapes are SmolLM2's, plus odd sizes no tile divides.

use candle_core::{Device, Tensor};
use prattle_webgpu::Gpu;

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// One GPU test at a time in this binary: `cargo test` runs tests on parallel threads, and Vulkan
/// devices torn down at once deadlock inside the NVIDIA driver (vkDestroyDevice). Take the turn
/// BEFORE making the device, so the device is dropped first and its teardown happens inside the turn.
fn turn() -> std::sync::MutexGuard<'static, ()> {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn gpu() -> Gpu {
    pollster::block_on(Gpu::new()).expect("no GPU adapter: the WebGPU engine needs one (Vulkan, Metal, DX12, or llvmpipe)")
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    assert!(worst / scale < 1e-5, "{what}: worst difference {worst} against a largest value {scale}");
}

/// y = x Wᵀ, the shape of every linear layer: x is m × k, W is n × k.
#[test]
fn matmul_equals_candle_on_the_models_shapes_and_odd_ones() {
    let _turn = turn();
    let g = gpu();
    for (m, k, n) in [(7, 576, 1536), (1, 960, 2560), (3, 37, 29), (17, 64, 1)] {
        let (x, w) = (values(m * k, 1), values(n * k, 2));
        let want = Tensor::from_slice(&x, (m, k), &Device::Cpu).unwrap()
            .matmul(&Tensor::from_slice(&w, (n, k), &Device::Cpu).unwrap().t().unwrap())
            .unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (gx, gw) = (g.upload(&x, &[m, k]), g.upload(&w, &[n, k]));
        let got = pollster::block_on(g.download(&g.matmul(&gx, &gw)));
        close(&got, &want, &format!("matmul {m}×{k} · ({n}×{k})ᵀ"));
    }
}

/// RMSNorm over each row of x (m × d), scaled by the weight g (d).
#[test]
fn rms_norm_equals_candle_on_the_models_widths_and_odd_ones() {
    let _turn = turn();
    let g = gpu();
    for (m, d) in [(5, 576), (2, 960), (3, 37), (1, 2048)] {
        let (x, w) = (values(m * d, 3), values(d, 4));
        let want = candle_nn::ops::rms_norm(
            &Tensor::from_slice(&x, (m, d), &Device::Cpu).unwrap(),
            &Tensor::from_slice(&w, d, &Device::Cpu).unwrap(),
            1e-5,
        )
        .unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (gx, gw) = (g.upload(&x, &[m, d]), g.upload(&w, &[d]));
        let got = pollster::block_on(g.download(&g.rms_norm(&gx, &gw, 1e-5)));
        close(&got, &want, &format!("rms_norm {m}×{d}"));
    }
}

// ---- Qwen3.5's kernels, each against the arithmetic of prattle-core's `qwen3_5`, written as loops ----

fn f16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
}

/// An f16 weight multiplies as its f16 values do: Qwen3.5's matrices, and an odd count of rows.
#[test]
fn matmul_with_an_f16_weight_equals_its_f16_values_at_f32() {
    let _turn = turn();
    let g = gpu();
    for (m, k, n) in [(3, 1024, 2048), (1, 64, 7), (17, 6, 5)] {
        let (x, w) = (values(m * k, 5), values(n * k, 6));
        let (gx, half, full) = (g.upload(&x, &[m, k]), g.upload_half(&w, &[n, k]), g.upload(&f16(&w), &[n, k]));
        let got = pollster::block_on(g.download(&g.matmul(&gx, &half)));
        let want = pollster::block_on(g.download(&g.matmul(&gx, &full)));
        close(&got, &want, &format!("f16 matmul {m}×{k} · ({n}×{k})ᵀ"));
        let rows = pollster::block_on(g.download(&g.embed(&g.upload_ids(&[2, 0, (n - 1) as u32]), &half)));
        let want_rows: Vec<f32> = [2, 0, n - 1].iter().flat_map(|&r| f16(&w[r * k..(r + 1) * k])).collect();
        close(&rows, &want_rows, &format!("f16 embed of {n}×{k}"));
    }
}

/// RoPE over each head's first `rot` dims, rotate-half within them; the rest untouched.
#[test]
fn rope_turns_only_the_first_rot_dims_of_each_head() {
    let _turn = turn();
    let g = gpu();
    let (n, heads, hd, rot, pos, theta) = (3, 2, 16, 4, 5, 10_000_000f32);
    let x = values(n * heads * hd, 7);
    let inv: Vec<f32> = (0..rot).step_by(2).map(|i| 1.0 / theta.powf(i as f32 / rot as f32)).collect();
    let (mut cos, mut sin) = (vec![], vec![]);
    for t in 0..pos + n {
        for f in &inv {
            cos.push((t as f32 * f).cos());
            sin.push((t as f32 * f).sin());
        }
    }
    let mut want = x.clone();
    for i in 0..n {
        for h in 0..heads {
            let b = (i * heads + h) * hd;
            for j in 0..rot / 2 {
                let (c, s) = (cos[(pos + i) * rot / 2 + j], sin[(pos + i) * rot / 2 + j]);
                let (x1, x2) = (x[b + j], x[b + j + rot / 2]);
                want[b + j] = x1 * c - x2 * s;
                want[b + j + rot / 2] = x2 * c + x1 * s;
            }
        }
    }
    let gx = g.upload(&x, &[n, heads * hd]);
    g.rope_part(&gx, heads, rot, &g.upload(&cos, &[pos + n, rot / 2]), &g.upload(&sin, &[pos + n, rot / 2]), pos);
    close(&pollster::block_on(g.download(&gx)), &want, "partial rope");
}

/// The query and its gate apart, then the attention output × σ(gate).
#[test]
fn a_heads_halves_split_and_the_gate_multiplies_by_its_sigmoid() {
    let _turn = turn();
    let g = gpu();
    let (n, heads, hd) = (3, 2, 5);
    let x = values(n * heads * 2 * hd, 8);
    let (a, b) = g.split_halves(&g.upload(&x, &[n, heads * 2 * hd]), heads);
    let (mut wa, mut wb) = (vec![], vec![]);
    for r in 0..n * heads {
        wa.extend_from_slice(&x[r * 2 * hd..r * 2 * hd + hd]);
        wb.extend_from_slice(&x[r * 2 * hd + hd..(r + 1) * 2 * hd]);
    }
    g.mul_sigmoid(&a, &b);
    let want: Vec<f32> = wa.iter().zip(&wb).map(|(q, z)| q / (1.0 + (-z).exp())).collect();
    close(&pollster::block_on(g.download(&a)), &want, "gate");
    close(&pollster::block_on(g.download(&b)), &wb, "the gate half");
}

/// L2 over the q and k heads of each row, q's then scaled; v's columns untouched. Then the gates.
#[test]
fn queries_and_keys_are_l2_normalised_and_the_gates_are_decay_and_beta() {
    let _turn = turn();
    let g = gpu();
    let (n, kk, dk, vw) = (3, 2, 8, 5);
    let width = 2 * kk * dk + vw;
    let x = values(n * width, 9);
    let gx = g.upload(&x, &[n, width]);
    let scale = 1.0 / (dk as f32).sqrt();
    g.l2norm_heads(&gx, 2 * kk, dk, kk, scale);
    let mut want = x.clone();
    for r in 0..n {
        for h in 0..2 * kk {
            let b = r * width + h * dk;
            let norm = (x[b..b + dk].iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            for i in 0..dk {
                want[b + i] = x[b + i] / norm * if h < kk { scale } else { 1.0 };
            }
        }
    }
    close(&pollster::block_on(g.download(&gx)), &want, "l2 heads");

    let vh = 4;
    let (a, b) = (values(n * vh, 10).iter().map(|v| v * 30.0).collect::<Vec<_>>(), values(n * vh, 11));
    let (a_log, dt) = (values(vh, 12), values(vh, 13));
    let (ga, gb) = (g.upload(&a, &[n, vh]), g.upload(&b, &[n, vh]));
    g.gdn_gates(&ga, &gb, &g.upload(&a_log.iter().map(|x| -x.exp()).collect::<Vec<_>>(), &[vh]), &g.upload(&dt, &[vh]));
    let softplus = |x: f32| x.max(0.0) + (-x.abs()).exp().ln_1p();
    let decay: Vec<f32> = (0..n * vh).map(|i| (-a_log[i % vh].exp() * softplus(a[i] + dt[i % vh])).exp()).collect();
    let beta: Vec<f32> = b.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    close(&pollster::block_on(g.download(&ga)), &decay, "decay");
    close(&pollster::block_on(g.download(&gb)), &beta, "beta");
}

/// The conv and the delta rule over 5 tokens, then 1, then 1, each call continuing the state the last
/// left — against one loop over all 7: what a prompt then a reply does.
#[test]
fn the_conv_and_the_delta_rule_carry_their_state_across_calls() {
    let _turn = turn();
    let g = gpu();
    let (kk, vh, dk, dv, k) = (2, 4, 8, 6, 4);
    let ch = 2 * kk * dk + vh * dv;
    let total = 7;
    let raw = values(total * ch, 14);
    let w = values(ch * k, 15);
    let decay: Vec<f32> = values(total * vh, 16).iter().map(|v| 0.8 + 0.15 * v).collect();
    let beta: Vec<f32> = values(total * vh, 17).iter().map(|v| 0.5 + 0.4 * v).collect();

    // the reference: conv over zero-padded history, then the delta rule from a zero state
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let at = |t: isize, c: usize| if t < 0 { 0.0 } else { raw[t as usize * ch + c] };
    let mixed: Vec<f32> = (0..total * ch).map(|i| {
        let (t, c) = (i / ch, i % ch);
        silu((0..k).map(|j| w[c * k + j] * at(t as isize + j as isize - (k as isize - 1), c)).sum())
    }).collect();
    let mut s = vec![0f32; vh * dk * dv];
    let mut want_o = vec![0f32; total * vh * dv];
    for t in 0..total {
        for h in 0..vh {
            let kh = h / (vh / kk);
            let q = &mixed[t * ch + kh * dk..][..dk];
            let kv = &mixed[t * ch + kk * dk + kh * dk..][..dk];
            let st = &mut s[h * dk * dv..][..dk * dv];
            let (gt, bt) = (decay[t * vh + h], beta[t * vh + h]);
            for j in 0..dv {
                let mut mem = 0.0;
                for i in 0..dk {
                    st[i * dv + j] *= gt;
                    mem += st[i * dv + j] * kv[i];
                }
                let d = (mixed[t * ch + 2 * kk * dk + h * dv + j] - mem) * bt;
                let mut o = 0.0;
                for i in 0..dk {
                    st[i * dv + j] += kv[i] * d;
                    o += st[i * dv + j] * q[i];
                }
                want_o[t * vh * dv + h * dv + j] = o;
            }
        }
    }

    let (gw, cstate, rstate) = (g.upload(&w, &[ch, k]), g.zeros(&[k - 1, ch]), g.zeros(&[vh, dk, dv]));
    let (mut got_mixed, mut got_o) = (vec![], vec![]);
    let mut from = 0;
    for n in [5, 1, 1] {
        let m = g.conv_silu(&g.upload(&raw[from * ch..(from + n) * ch], &[n, ch]), &gw, &cstate);
        let o = g.delta(&m, &g.upload(&decay[from * vh..(from + n) * vh], &[n, vh]), &g.upload(&beta[from * vh..(from + n) * vh], &[n, vh]), &rstate, kk, vh, dk, dv);
        got_mixed.extend(pollster::block_on(g.download(&m)));
        got_o.extend(pollster::block_on(g.download(&o)));
        from += n;
    }
    close(&got_mixed, &mixed, "conv + silu, 5 then 1 then 1");
    close(&got_o, &want_o, "delta rule, 5 then 1 then 1");
    close(&pollster::block_on(g.download(&rstate)), &s, "the recurrent state after 7");
}

/// LFM2's short conv: B·x from one projection's row, the conv with no activation across two calls
/// (its state carried), then C · conv.
#[test]
fn the_short_conv_gates_its_input_and_output_by_column_products_and_carries_its_rows() {
    let _turn = turn();
    let g = gpu();
    let (h, k, total) = (6, 3, 5);
    let bcx = values(total * 3 * h, 18);
    let w = values(h * k, 19);
    let col = |t: usize, part: usize, c: usize| bcx[t * 3 * h + part * h + c];
    let bx = |t: isize, c: usize| if t < 0 { 0.0 } else { col(t as usize, 0, c) * col(t as usize, 2, c) };
    let want: Vec<f32> = (0..total * h)
        .map(|i| {
            let (t, c) = (i / h, i % h);
            let conv: f32 = (0..k).map(|j| w[c * k + j] * bx(t as isize + j as isize - (k as isize - 1), c)).sum();
            col(t, 1, c) * conv
        })
        .collect();
    let (gw, rows) = (g.upload(&w, &[h, k]), g.zeros(&[k - 1, h]));
    let mut got = vec![];
    let mut from = 0;
    for n in [3, 2] {
        let gbcx = g.upload(&bcx[from * 3 * h..(from + n) * 3 * h], &[n, 3 * h]);
        let gbx = g.mul_cols(&gbcx, 0, &gbcx, 2 * h, h);
        let conv = g.conv(&gbx, &gw, &rows, false);
        got.extend(pollster::block_on(g.download(&g.mul_cols(&gbcx, h, &conv, 0, h))));
        from += n;
    }
    close(&got, &want, "short conv, 3 then 2");
}
