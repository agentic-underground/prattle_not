//! `unembed` is the arithmetic `forward_capture` ends with — the final norm, the head, the family's own
//! softcap or scaling — so a logit lens reads any layer's residual as the model reads its last.
//!
//! The witness is a second model, not the code under test: a two-layer model's residual entering layer 1
//! is exactly the last residual of the same weights loaded with one layer, whose `forward_capture`
//! logits `unembed` must give.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use prattle_core::decoder::Decoder;

/// The family's config at `layers` layers; `types` are its layer kinds, the first `layers` used.
fn config(base: &str, types: &[&str], layers: usize) -> String {
    let kinds: Vec<String> = types[..layers].iter().map(|t| format!("\"{t}\"")).collect();
    format!(r#"{{{base},"num_hidden_layers":{layers},"layer_types":[{}]}}"#, kinds.join(","))
}

/// Weights for the two-layer model, made by the family's own builder, as safetensors bytes.
fn weights(name: &str, build: impl FnOnce(VarBuilder)) -> Vec<u8> {
    let map = VarMap::new();
    build(VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu));
    // every weight small and nonzero — a builder that starts a norm at zero would zero the logits, and a
    // lens over zeros agrees with anything
    for (i, var) in map.all_vars().iter().enumerate() {
        let n = var.elem_count();
        let wave: Vec<f32> = (0..n).map(|j| 0.1 * ((j as f32 + 1.0) * 0.7071 + i as f32).sin()).collect();
        var.set(&Tensor::from_vec(wave, var.shape(), &Device::Cpu).unwrap()).unwrap();
    }
    let path = std::env::temp_dir().join(format!("prattle-unembed-{name}-{}.safetensors", std::process::id()));
    map.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    bytes
}

fn check(name: &str, base: &str, types: &[&str], build: impl FnOnce(&[u8], VarBuilder)) {
    let cpu = Device::Cpu;
    let two = config(base, types, 2);
    let one = config(base, types, 1);
    let bytes = weights(name, |vb| build(two.as_bytes(), vb));
    let load = |c: &str| -> Decoder { prattle_core::model::load_decoder(c.as_bytes(), bytes.clone(), &cpu).unwrap() };
    let (whole, cut) = (load(&two), load(&one));
    let ids = Tensor::new(&[[1u32, 5, 9, 2, 7, 3]], &cpu).unwrap();

    let mut capture = Vec::new();
    whole.forward_capture(&ids, 0, &mut whole.empty_state(), Some(&mut capture)).unwrap();
    let last = capture[1].narrow(1, 5, 1).unwrap();
    let lens = cut.unembed(&last).unwrap().flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let logits = cut.forward_capture(&ids, 0, &mut cut.empty_state(), None).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(lens.len(), logits.len(), "{name}: unembed gave {} values, forward_capture {}", lens.len(), logits.len());
    let worst = lens.iter().zip(&logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let scale = logits.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1.0);
    assert!(worst <= 1e-5 * scale, "{name}: unembed differs from forward_capture by {worst:e}");
    assert!(logits.iter().any(|&v| v != logits[0]), "{name}: degenerate logits, every value {}", logits[0]);
}

#[test]
fn llama_unembeds_as_forward_capture_ends() {
    let base = r#""model_type":"llama","hidden_size":256,"intermediate_size":256,"vocab_size":32,"num_attention_heads":2,
        "num_key_value_heads":1,"rms_norm_eps":1e-5,"rope_theta":10000.0,"max_position_embeddings":64,"tie_word_embeddings":false"#;
    check("llama", base, &["full_attention", "full_attention"], |c, vb| {
        let raw: candle_transformers::models::llama::LlamaConfig = serde_json::from_slice(c).unwrap();
        prattle_core::llama::Llama::load(vb, &raw.into_config(false)).unwrap();
    });
}

#[test]
fn qwen35_unembeds_as_forward_capture_ends() {
    let base = r#""model_type":"qwen3_5_text","hidden_size":256,"intermediate_size":256,"vocab_size":32,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":128,"linear_num_key_heads":2,"linear_num_value_heads":2,"linear_key_head_dim":128,
        "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rms_norm_eps":1e-6,"tie_word_embeddings":true"#;
    check("qwen3_5", base, &["linear_attention", "full_attention"], |c, vb| {
        prattle_core::qwen3_5::Qwen35::from_builder(c, vb).unwrap();
    });
}

#[test]
fn lfm2_unembeds_as_forward_capture_ends() {
    let base = r#""model_type":"lfm2","hidden_size":256,"block_ff_dim":256,"vocab_size":32,"num_attention_heads":2,
        "num_key_value_heads":1,"conv_L_cache":3,"norm_eps":1e-5"#;
    check("lfm2", base, &["conv", "full_attention"], |c, vb| {
        prattle_core::lfm2::Lfm2::from_builder(c, vb).unwrap();
    });
}

#[test]
fn gemma3_unembeds_as_forward_capture_ends() {
    let base = r#""model_type":"gemma3_text","hidden_size":256,"intermediate_size":256,"vocab_size":32,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":128,"query_pre_attn_scalar":128,"sliding_window":4"#;
    check("gemma3", base, &["sliding_attention", "full_attention"], |c, vb| {
        prattle_core::gemma3::Gemma3::from_builder(c, vb).unwrap();
    });
}

#[test]
fn gemma4_unembeds_as_forward_capture_ends() {
    let base = r#""model_type":"gemma4_text","hidden_size":256,"intermediate_size":256,"vocab_size":32,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":128,"global_head_dim":128,"sliding_window":4,"final_logit_softcapping":30.0"#;
    check("gemma4", base, &["sliding_attention", "full_attention"], |c, vb| {
        prattle_core::gemma4::Gemma4::from_builder(c, vb).unwrap();
    });
}

#[test]
fn granite_unembeds_as_forward_capture_ends() {
    let base = r#""model_type":"granitemoehybrid","hidden_size":256,"shared_intermediate_size":256,"vocab_size":32,
        "num_attention_heads":2,"num_key_value_heads":1,"mamba_d_conv":4,"mamba_d_state":16,"mamba_expand":2,"mamba_n_groups":1,
        "mamba_n_heads":8,"mamba_d_head":64,"attention_multiplier":0.1,"embedding_multiplier":2.0,"logits_scaling":4.0,
        "residual_multiplier":0.5,"position_embedding_type":"nope","rms_norm_eps":1e-5"#;
    check("granite", base, &["mamba", "attention"], |c, vb| {
        prattle_core::granite::Granite::from_builder(c, vb).unwrap();
    });
}
