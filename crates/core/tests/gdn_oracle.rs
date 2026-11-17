//! Layer 0 of the 4B in Rust against the oracle's canonical state after the same prefill.
//! INTEGRATION: needs `just oracle-models` and `just oracle-state`.

use candle_core::quantized::gguf_file;
use candle_core::Device;
use prattle_core::gdn::{self, Geometry, Weights};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");

fn f32s(c: &gguf_file::Content, f: &mut std::fs::File, name: &str) -> Vec<f32> {
    c.tensor(f, name, &Device::Cpu).unwrap_or_else(|e| panic!("{name}: {e}"))
        .dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// max |a−b| relative to max |b|
fn relative_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
#[ignore = "integration"]
fn layer_zero_matches_the_oracles_recurrent_and_conv_state() {
    let gguf = format!("{ROOT}models/latentport-4-27b/models/Qwen3.5-4B-Q4_K_M.gguf");
    let mut f = std::fs::File::open(&gguf).expect("run `just oracle-models`");
    let c = gguf_file::Content::read(&mut f).unwrap();
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{ROOT}target/oracle/state-4b.json")).expect("run `just oracle-state`"),
    )
    .unwrap();
    let ids: Vec<usize> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let bytes = std::fs::read(format!("{ROOT}target/oracle/state-4b.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let field = |n: &str| -> Vec<f32> {
        st.tensor(n).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    };

    let hidden = 2560;
    let embd = f32s(&c, &mut f, "token_embd.weight");
    let x: Vec<f32> = ids.iter().flat_map(|&t| embd[t * hidden..(t + 1) * hidden].to_vec()).collect();
    let g = Geometry { key_heads: 16, value_heads: 32, dim: 128, conv: 4 };
    let w = Weights {
        norm: f32s(&c, &mut f, "blk.0.attn_norm.weight"),
        qkv: f32s(&c, &mut f, "blk.0.attn_qkv.weight"),
        alpha: f32s(&c, &mut f, "blk.0.ssm_alpha.weight"),
        beta: f32s(&c, &mut f, "blk.0.ssm_beta.weight"),
        a: f32s(&c, &mut f, "blk.0.ssm_a"),
        dt_bias: f32s(&c, &mut f, "blk.0.ssm_dt.bias"),
        conv: f32s(&c, &mut f, "blk.0.ssm_conv1d.weight"),
        eps: 1e-6,
    };
    let state = gdn::prefill(&g, &w, &x, hidden);
    let (er, ec) = (relative_error(&state.r, &field("R.0")), relative_error(&state.c, &field("C.0")));
    println!("layer 0 over {} tokens: R relative error {er:.2e}, C relative error {ec:.2e}", ids.len());
    assert!(ec < 2e-2, "C.0 differs from the oracle by {ec:.2e} of its range");
    assert!(er < 2e-2, "R.0 differs from the oracle by {er:.2e} of its range");
}
