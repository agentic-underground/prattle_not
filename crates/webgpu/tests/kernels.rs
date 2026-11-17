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
