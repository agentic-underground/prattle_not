//! The Rust decoder against a model family's own reference: Hugging Face transformers' logits for the
//! same tokens (scripts/hf_reference.py → target/reference/<model>.safetensors). INTEGRATION: needs
//! the weights (`just models-qwen3`) and the reference run.

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
/// f32 against f32 from the same bf16 weights: only summation order differs.
const BOUND: f32 = 1e-3;

fn matches_transformers(model: &str) {
    let bytes = std::fs::read(format!("{ROOT}target/reference/{model}.safetensors")).expect("run scripts/hf_reference.py");
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let get = |n: &str| -> Vec<f32> { st.tensor(n).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let ids: Vec<u32> = get("ids").iter().map(|v| *v as u32).collect();
    let oracle = get("logits");
    let dir = format!("{ROOT}models/{model}");
    let config = std::fs::read(format!("{dir}/config.json")).unwrap();
    let weights = prattle_core::weights::read_dir(&dir).unwrap();
    let device = candle_core::Device::Cpu;
    let llama = prattle_core::model::load(&config, weights, &device).unwrap();
    let mut kv = llama.empty_state();
    let input = candle_core::Tensor::new(ids.as_slice(), &device).unwrap().unsqueeze(0).unwrap();
    let ours: Vec<f32> = llama.forward(&input, 0, &mut kv).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(ours.len(), oracle.len(), "{} logits, transformers gives {}", ours.len(), oracle.len());
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let num: f64 = ours.iter().zip(&oracle).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = oracle.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt() as f32;
    println!("{model}: logits within {rel:.2e} of transformers; argmax {} (transformers {})", argmax(&ours), argmax(&oracle));
    assert_eq!(argmax(&ours), argmax(&oracle), "{model}: a different next token");
    assert!(rel <= BOUND, "{model}: logits {rel:.2e} from transformers, over {BOUND:.0e}");
}

#[test]
#[ignore = "integration"]
fn qwen3_0_6b_matches_transformers() {
    matches_transformers("Qwen3-0.6B");
}

#[test]
#[ignore = "integration"]
fn qwen3_1_7b_matches_transformers() {
    matches_transformers("Qwen3-1.7B");
}

/// Llama 3's RoPE scaling (`rope_type: llama3`) — BROADEN_SUPPORT's first loader.
#[test]
#[ignore = "integration"]
fn llama_3_2_1b_matches_transformers() {
    matches_transformers("Llama-3.2-1B-Instruct");
}

#[test]
#[ignore = "integration"]
fn llama_3_2_3b_matches_transformers() {
    matches_transformers("Llama-3.2-3B-Instruct");
}

/// Gemma 3's own decoder (`prattle_core::gemma3`) against transformers — BROADEN_SUPPORT's second loader.
fn gemma3_matches_transformers(model: &str) {
    let bytes = std::fs::read(format!("{ROOT}target/reference/{model}.safetensors")).expect("run scripts/hf_reference.py");
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let get = |n: &str| -> Vec<f32> { st.tensor(n).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let ids: Vec<u32> = get("ids").iter().map(|v| *v as u32).collect();
    let oracle = get("logits");
    let dir = format!("{ROOT}models/{model}");
    let device = candle_core::Device::Cpu;
    let g = prattle_core::gemma3::Gemma3::load(&std::fs::read(format!("{dir}/config.json")).unwrap(), prattle_core::weights::read_dir(&dir).unwrap(), &device).unwrap();
    let mut kv = g.empty_state();
    let input = candle_core::Tensor::new(ids.as_slice(), &device).unwrap().unsqueeze(0).unwrap();
    let ours: Vec<f32> = g.forward(&input, 0, &mut kv).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(ours.len(), oracle.len(), "{} logits, transformers gives {}", ours.len(), oracle.len());
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let num: f64 = ours.iter().zip(&oracle).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = oracle.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt() as f32;
    println!("{model}: logits within {rel:.2e} of transformers; argmax {} (transformers {})", argmax(&ours), argmax(&oracle));
    assert_eq!(argmax(&ours), argmax(&oracle), "{model}: a different next token");
    assert!(rel <= BOUND, "{model}: logits {rel:.2e} from transformers, over {BOUND:.0e}");
}

#[test]
#[ignore = "integration"]
fn gemma_3_270m_matches_transformers() {
    gemma3_matches_transformers("gemma-3-270m-it");
}

/// A multimodal checkpoint (`language_model.model.*`, transformers' defaults for what its text_config
/// omits, linear RoPE scaling ×8 on the global layers) — the shape TranslateGemma also has.
#[test]
#[ignore = "integration"]
fn gemma_3_4b_matches_transformers() {
    gemma3_matches_transformers("gemma-3-4b-it");
}

#[test]
#[ignore = "integration"]
fn translategemma_4b_matches_transformers() {
    gemma3_matches_transformers("translategemma-4b-it");
}

/// The bridge's decoder interface on Gemma 3: every-position logits, the captured residual streams,
/// each layer's own RoPE, and the differentiable mode all agree with the plain forward.
#[test]
#[ignore = "integration"]
fn gemma_3_serves_the_bridges_decoder_interface() {
    use candle_core::{Device, Tensor};
    let dir = format!("{ROOT}models/gemma-3-270m-it");
    let device = Device::Cpu;
    let mut g = prattle_core::gemma3::Gemma3::load(&std::fs::read(format!("{dir}/config.json")).unwrap(), prattle_core::weights::read_dir(&dir).unwrap(), &device).unwrap();
    let ids = Tensor::new(&[2u32, 818, 5279, 529, 7001, 563][..], &device).unwrap().unsqueeze(0).unwrap();
    let rel = |a: &Tensor, b: &Tensor| -> f32 {
        let (a, b): (Vec<f32>, Vec<f32>) = (a.flatten_all().unwrap().to_vec1().unwrap(), b.flatten_all().unwrap().to_vec1().unwrap());
        (a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / b.iter().map(|y| y * y).sum::<f32>()).sqrt()
    };
    let last = g.forward(&ids, 0, &mut g.empty_state()).unwrap();
    let all = g.forward_all(&ids, 0, &mut g.empty_state()).unwrap();
    assert!(rel(&all.narrow(1, 5, 1).unwrap().squeeze(1).unwrap(), &last) < 1e-6, "forward_all's last row is not forward");
    let mut caps = Vec::new();
    g.forward_capture(&ids, 0, &mut g.empty_state(), Some(&mut caps)).unwrap();
    assert_eq!(caps.len(), g.geometry().layers, "one captured stream per layer");
    let k = Tensor::randn(0f32, 1.0, (1, 1, 6, 256), &device).unwrap();
    for layer in [0, 5] {
        let back = g.rotate_keys(layer, &g.rotate_keys(layer, &k, 3, false).unwrap(), 3, true).unwrap();
        assert!(rel(&back, &k) < 1e-5, "layer {layer}: de-rotation does not invert rotation");
    }
    g.set_differentiable(true);
    let diff = g.forward(&ids, 0, &mut g.empty_state()).unwrap();
    assert!(rel(&diff, &last) < 1e-4, "the differentiable path disagrees: {}", rel(&diff, &last));
}

/// The whole Model on Gemma 3 — its loader, template and stop token — answering a question.
#[test]
#[ignore = "integration"]
fn a_gemma_3_model_answers_a_question() {
    let dir = format!("{ROOT}models/gemma-3-270m-it");
    let read = |f: &str| std::fs::read(format!("{dir}/{f}")).unwrap();
    let mut m = prattle_core::Model::from_bytes(&read("config.json"), &read("tokenizer.json"), prattle_core::weights::read_dir(&dir).unwrap()).unwrap();
    m.start_arm("What is the capital of France?", prattle_core::Arm::Native).unwrap();
    let mut reply = String::new();
    for _ in 0..24 {
        match m.step().unwrap() {
            prattle_core::Step::Text(t) => reply.push_str(&t),
            prattle_core::Step::Done => break,
        }
    }
    println!("gemma-3-270m-it: {reply:?}");
    assert!(reply.contains("Paris"), "no Paris in {reply:?}");
}

/// Gemma 4 E2B answers through `Model` — its turn, and its reply ending on `<turn|>`.
#[test]
#[ignore = "integration"]
fn a_gemma_4_model_answers_a_question() {
    let dir = format!("{ROOT}models/gemma-4-E2B-it");
    let read = |f: &str| std::fs::read(format!("{dir}/{f}")).unwrap();
    let mut m = prattle_core::Model::from_bytes(&read("config.json"), &read("tokenizer.json"), prattle_core::weights::read_dir(&dir).unwrap()).unwrap();
    m.start_arm("What is the capital of France? Answer in one word.", prattle_core::Arm::Native).unwrap();
    let (mut reply, mut done) = (String::new(), false);
    for _ in 0..24 {
        match m.step().unwrap() {
            prattle_core::Step::Text(t) => reply.push_str(&t),
            prattle_core::Step::Done => { done = true; break; }
        }
    }
    println!("gemma-4-E2B-it: {reply:?}");
    assert!(reply.contains("Paris"), "no Paris in {reply:?}");
    assert!(done, "the reply did not end on its turn's close: {reply:?}");
}

/// Qwen3.5's hybrid decoder from safetensors (`prattle_core::qwen3_5`) — Gated DeltaNet and gated
/// attention — against transformers. BROADEN_SUPPORT's Qwen3.5 loader, for pair C.
#[test]
#[ignore = "integration"]
fn qwen3_5_0_8b_matches_transformers() {
    qwen3_5_matches_transformers("Qwen3.5-0.8B");
}

#[test]
#[ignore = "integration"]
fn qwen3_5_2b_matches_transformers() {
    qwen3_5_matches_transformers("Qwen3.5-2B");
}

/// Pair D's reader, the LatentPort paper's own source model.
#[test]
#[ignore = "integration"]
fn qwen3_5_4b_base_matches_transformers() {
    qwen3_5_matches_transformers("Qwen3.5-4B-Base");
}

/// Pair D's answerer.
#[test]
#[ignore = "integration"]
fn qwen3_5_9b_base_matches_transformers() {
    qwen3_5_matches_transformers("Qwen3.5-9B-Base");
}

/// The same decoder with its weights at half precision (f16 here: candle's CPU has no bf16 matmul) —
/// what lets pair D's 9B fit a card. Same next token as transformers, logits within 3%.
#[test]
#[ignore = "integration"]
fn qwen3_5_0_8b_at_half_precision_matches_transformers() {
    let model = "Qwen3.5-0.8B";
    let bytes = std::fs::read(format!("{ROOT}target/reference/{model}.safetensors")).expect("run scripts/hf_reference.py");
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let get = |n: &str| -> Vec<f32> { st.tensor(n).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let ids: Vec<u32> = get("ids").iter().map(|v| *v as u32).collect();
    let oracle = get("logits");
    let dir = format!("{ROOT}models/{model}");
    let device = candle_core::Device::Cpu;
    let config = std::fs::read(format!("{dir}/config.json")).unwrap();
    let m = prattle_core::model::load_decoder_in(&config, prattle_core::weights::read_dir(&dir).unwrap(), &device, candle_core::DType::F16).unwrap();
    let input = candle_core::Tensor::new(ids.as_slice(), &device).unwrap().unsqueeze(0).unwrap();
    let ours: Vec<f32> = m.forward(&input, 0, &mut m.empty_state()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let num: f64 = ours.iter().zip(&oracle).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = oracle.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt() as f32;
    println!("{model} at f16: logits within {rel:.2e} of transformers; argmax {} (transformers {})", argmax(&ours), argmax(&oracle));
    assert_eq!(argmax(&ours), argmax(&oracle), "{model} at f16: a different next token");
    assert!(rel <= 3e-2, "{model} at f16: logits {rel:.2e} from transformers");
}

fn qwen3_5_matches_transformers(model: &str) {
    let bytes = std::fs::read(format!("{ROOT}target/reference/{model}.safetensors")).expect("run scripts/hf_reference.py");
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let get = |n: &str| -> Vec<f32> { st.tensor(n).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let ids: Vec<u32> = get("ids").iter().map(|v| *v as u32).collect();
    let oracle = get("logits");
    let dir = format!("{ROOT}models/{model}");
    let device = candle_core::Device::Cpu;
    let m = prattle_core::qwen3_5::Qwen35::load(&std::fs::read(format!("{dir}/config.json")).unwrap(), prattle_core::weights::read_dir(&dir).unwrap(), &device).unwrap();
    let input = candle_core::Tensor::new(ids.as_slice(), &device).unwrap().unsqueeze(0).unwrap();
    let ours: Vec<f32> = m.forward(&input, 0, &mut m.empty_state()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(ours.len(), oracle.len(), "{} logits, transformers gives {}", ours.len(), oracle.len());
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let num: f64 = ours.iter().zip(&oracle).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = oracle.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt() as f32;
    println!("{model}: logits within {rel:.2e} of transformers; argmax {} (transformers {})", argmax(&ours), argmax(&oracle));
    assert_eq!(argmax(&ours), argmax(&oracle), "{model}: a different next token");
    assert!(rel <= BOUND, "{model}: logits {rel:.2e} from transformers, over {BOUND:.0e}");
}

/// A hybrid model's own memory — attention K/V AND the Gated DeltaNet's conv and recurrent state —
/// restored into it reproduces its native reply exactly: the state a bridge carries is complete.
#[test]
#[ignore = "integration"]
fn a_qwen3_5_models_restored_memory_reproduces_its_reply() {
    let dir = format!("{ROOT}models/Qwen3.5-0.8B");
    let read = |f: &str| std::fs::read(format!("{dir}/{f}")).unwrap();
    let mut m = prattle_core::Model::from_bytes(&read("config.json"), &read("tokenizer.json"), prattle_core::weights::read_dir(&dir).unwrap()).unwrap();
    let reply = |m: &mut prattle_core::Model, arm: prattle_core::Arm| -> String {
        m.start_arm("What is the capital of France?", arm).unwrap();
        let mut out = String::new();
        for _ in 0..16 {
            match m.step().unwrap() {
                prattle_core::Step::Text(t) => out.push_str(&t),
                prattle_core::Step::Done => break,
            }
        }
        out
    };
    let (native, restored) = (reply(&mut m, prattle_core::Arm::Native), reply(&mut m, prattle_core::Arm::Restore));
    println!("Qwen3.5-0.8B native: {native:?}\n             restored: {restored:?}");
    assert!(native.contains("Paris"), "no Paris in {native:?}");
    assert_eq!(restored, native, "the restored hybrid memory does not reproduce the native reply");
}

/// Qwen3.5's differentiable path (basic-op norms, softmax, RoPE — what the correction trains through)
/// gives the plain forward's logits.
#[test]
#[ignore = "integration"]
fn qwen3_5s_differentiable_path_agrees() {
    use candle_core::{Device, Tensor};
    let dir = format!("{ROOT}models/Qwen3.5-0.8B");
    let device = Device::Cpu;
    let mut m = prattle_core::model::load_decoder(&std::fs::read(format!("{dir}/config.json")).unwrap(), prattle_core::weights::read_dir(&dir).unwrap(), &device).unwrap();
    let ids = Tensor::new(&[3838u32, 374, 279, 6722, 315, 9625, 30][..], &device).unwrap().unsqueeze(0).unwrap();
    let plain: Vec<f32> = m.forward(&ids, 0, &mut m.empty_state()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    m.set_differentiable(true);
    let diff: Vec<f32> = m.forward(&ids, 0, &mut m.empty_state()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let rel = (plain.iter().zip(&diff).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / plain.iter().map(|a| a * a).sum::<f32>()).sqrt();
    println!("Qwen3.5-0.8B differentiable vs plain: {rel:.2e}");
    assert!(rel < 1e-4, "the differentiable path disagrees: {rel:.2e}");
}

/// Any decoder `load_decoder` picks from the config, against transformers.
fn decoder_matches_transformers(model: &str) {
    let bytes = std::fs::read(format!("{ROOT}target/reference/{model}.safetensors")).expect("run scripts/hf_reference.py");
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let get = |n: &str| -> Vec<f32> { st.tensor(n).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let ids: Vec<u32> = get("ids").iter().map(|v| *v as u32).collect();
    let oracle = get("logits");
    let dir = format!("{ROOT}models/{model}");
    let device = candle_core::Device::Cpu;
    let m = prattle_core::model::load_decoder(&std::fs::read(format!("{dir}/config.json")).unwrap(), prattle_core::weights::read_dir(&dir).unwrap(), &device).unwrap();
    let input = candle_core::Tensor::new(ids.as_slice(), &device).unwrap().unsqueeze(0).unwrap();
    let ours: Vec<f32> = m.forward(&input, 0, &mut m.empty_state()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(ours.len(), oracle.len(), "{} logits, transformers gives {}", ours.len(), oracle.len());
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let num: f64 = ours.iter().zip(&oracle).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = oracle.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt() as f32;
    println!("{model}: logits within {rel:.2e} of transformers; argmax {} (transformers {})", argmax(&ours), argmax(&oracle));
    assert_eq!(argmax(&ours), argmax(&oracle), "{model}: a different next token");
    assert!(rel <= BOUND, "{model}: logits {rel:.2e} from transformers, over {BOUND:.0e}");
}

/// Granite 4.0's hybrid — Mamba-2 layers, attention with no position encoding, μP multipliers.
#[test]
#[ignore = "integration"]
fn granite_4_0_h_350m_matches_transformers() {
    decoder_matches_transformers("granite-4.0-h-350m");
}

#[test]
#[ignore = "integration"]
fn granite_4_0_h_1b_matches_transformers() {
    decoder_matches_transformers("granite-4.0-h-1b");
}

/// Granite 4.0's own memory — attention K/V AND each Mamba layer's conv and scan state — restored
/// into it reproduces its native reply exactly: the state a bridge carries is complete.
#[test]
#[ignore = "integration"]
fn a_granite_4_models_restored_memory_reproduces_its_reply() {
    let dir = format!("{ROOT}models/granite-4.0-h-350m");
    let read = |f: &str| std::fs::read(format!("{dir}/{f}")).unwrap();
    let mut m = prattle_core::Model::from_bytes(&read("config.json"), &read("tokenizer.json"), prattle_core::weights::read_dir(&dir).unwrap()).unwrap();
    let reply = |m: &mut prattle_core::Model, arm: prattle_core::Arm| -> String {
        m.start_arm("What is the capital of France?", arm).unwrap();
        let mut out = String::new();
        for _ in 0..16 {
            match m.step().unwrap() {
                prattle_core::Step::Text(t) => out.push_str(&t),
                prattle_core::Step::Done => break,
            }
        }
        out
    };
    let (native, restored) = (reply(&mut m, prattle_core::Arm::Native), reply(&mut m, prattle_core::Arm::Restore));
    println!("granite-4.0-h-350m native: {native:?}\n                   restored: {restored:?}");
    assert!(native.contains("Paris"), "no Paris in {native:?}");
    assert_eq!(restored, native, "the restored hybrid memory does not reproduce the native reply");
}

/// Granite 4.0's mixture of experts — 64 experts, 6 per token, beside the shared MLP.
#[test]
#[ignore = "integration"]
fn granite_4_0_h_tiny_matches_transformers() {
    decoder_matches_transformers("granite-4.0-h-tiny");
}

/// Gemma 4 E2B — per-layer input embeddings, its last 20 layers reading earlier layers' keys and values,
/// 512-wide global heads with a quarter of each rotated, and the vision and audio towers ignored.
#[test]
#[ignore = "integration"]
fn gemma_4_e2b_matches_transformers() {
    decoder_matches_transformers("gemma-4-E2B-it");
}

/// Gemma 4 31B quantized — K = V on its global layers, 4 KV heads 512 wide there and 16 × 256 on the
/// sliding ones — against transformers' bf16 logits (its f32 pass would need ~124 GB of host memory).
/// Measured: Q8 1.08e-1 (Granite's Q8 is 8.3e-2–1.06e-1 against f32), Q4K 6.37e-1, the same next token.
#[test]
#[ignore = "integration"]
fn gemma_4_31b_at_q4k_matches_transformers() {
    quantized_matches_transformers("gemma-4-31B-it", candle_core::quantized::GgmlDType::Q4K, 7e-1);
}

#[test]
#[ignore = "integration"]
fn gemma_4_31b_at_q8_matches_transformers() {
    quantized_matches_transformers("gemma-4-31B-it", candle_core::quantized::GgmlDType::Q8_0, 1.5e-1);
}

/// LFM2's hybrid — short convolutions with attention at `full_attn_idxs`.
#[test]
#[ignore = "integration"]
fn lfm2_350m_matches_transformers() {
    decoder_matches_transformers("LFM2-350M");
}

#[test]
#[ignore = "integration"]
fn lfm2_1_2b_matches_transformers() {
    decoder_matches_transformers("LFM2-1.2B");
}

/// LFM2's own memory — attention K/V AND each short conv's last rows — restored into it reproduces
/// its native reply exactly: the state a bridge carries is complete.
#[test]
#[ignore = "integration"]
fn an_lfm2_models_restored_memory_reproduces_its_reply() {
    let dir = format!("{ROOT}models/LFM2-350M");
    let read = |f: &str| std::fs::read(format!("{dir}/{f}")).unwrap();
    let mut m = prattle_core::Model::from_bytes(&read("config.json"), &read("tokenizer.json"), prattle_core::weights::read_dir(&dir).unwrap()).unwrap();
    let reply = |m: &mut prattle_core::Model, arm: prattle_core::Arm| -> String {
        m.start_arm("What is the capital of France?", arm).unwrap();
        let mut out = String::new();
        for _ in 0..16 {
            match m.step().unwrap() {
                prattle_core::Step::Text(t) => out.push_str(&t),
                prattle_core::Step::Done => break,
            }
        }
        out
    };
    let (native, restored) = (reply(&mut m, prattle_core::Arm::Native), reply(&mut m, prattle_core::Arm::Restore));
    println!("LFM2-350M native: {native:?}\n          restored: {restored:?}");
    assert!(native.contains("Paris"), "no Paris in {native:?}");
    assert_eq!(restored, native, "the restored hybrid memory does not reproduce the native reply");
}

/// Qwen3-1.7B with its linear weights quantized at load (`model::load_decoder_q`) against transformers'
/// f32 logits: close, the same next token — and measurably NOT the f32 path, or it was never quantized.
fn quantized_matches_transformers(model: &str, quant: candle_core::quantized::GgmlDType, bound: f32) {
    let bytes = std::fs::read(format!("{ROOT}target/reference/{model}.safetensors")).expect("run scripts/hf_reference.py");
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let get = |n: &str| -> Vec<f32> { st.tensor(n).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() };
    let ids: Vec<u32> = get("ids").iter().map(|v| *v as u32).collect();
    let oracle = get("logits");
    let dir = format!("{ROOT}models/{model}");
    let device = candle_core::Device::Cpu;
    let m = prattle_core::model::load_decoder_q(&std::fs::read(format!("{dir}/config.json")).unwrap(), prattle_core::weights::read_dir(&dir).unwrap(), &device, candle_core::DType::F32, Some(quant)).unwrap();
    let input = candle_core::Tensor::new(ids.as_slice(), &device).unwrap().unsqueeze(0).unwrap();
    let ours: Vec<f32> = m.forward(&input, 0, &mut m.empty_state()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let num: f64 = ours.iter().zip(&oracle).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = oracle.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt() as f32;
    println!("{model} at {quant:?}: logits within {rel:.2e} of transformers; argmax {} (transformers {})", argmax(&ours), argmax(&oracle));
    assert_eq!(argmax(&ours), argmax(&oracle), "{model} at {quant:?}: a different next token");
    assert!(rel <= bound, "{model} at {quant:?}: logits {rel:.2e} from transformers, over {bound:.0e}");
    assert!(rel > BOUND, "{model} at {quant:?}: logits {rel:.2e} — as close as f32, so the weights were not quantized");
}

#[test]
#[ignore = "integration"]
fn qwen3_1_7b_at_q8_matches_transformers() {
    quantized_matches_transformers("Qwen3-1.7B", candle_core::quantized::GgmlDType::Q8_0, 5e-2);
}

#[test]
#[ignore = "integration"]
fn qwen3_1_7b_at_q4k_matches_transformers() {
    quantized_matches_transformers("Qwen3-1.7B", candle_core::quantized::GgmlDType::Q4K, 4e-1);
}

/// Granite 4.0 builds a Mamba-2 layer's state from that layer's normed inputs (`state_from_inputs`) —
/// the path a bridge takes where the scan state cannot be mapped — and from its own true inputs it
/// rebuilds exactly the state its forward pass keeps.
#[test]
#[ignore = "integration"]
fn granite_rebuilds_a_mamba_layers_state_from_its_own_inputs() {
    use prattle_core::decoder::LayerKeys;
    let dir = format!("{ROOT}models/granite-4.0-h-350m");
    let device = candle_core::Device::Cpu;
    let m = prattle_core::model::load_decoder(&std::fs::read(format!("{dir}/config.json")).unwrap(), prattle_core::weights::read_dir(&dir).unwrap(), &device).unwrap();
    let ids = candle_core::Tensor::new(&[100264u32, 882, 1120, 374, 279, 6864, 315, 9822, 30, 100265][..], &device).unwrap().unsqueeze(0).unwrap();
    let (mut kv, mut residuals) = (m.empty_state(), Vec::new());
    m.forward_capture(&ids, 0, &mut kv, Some(&mut residuals)).unwrap();
    let layer = 0; // a Mamba layer in h-350m
    let inputs = m.fitted_rows(layer, &residuals[layer]).unwrap().expect("a Mamba layer has input rows").unsqueeze(0).unwrap();
    let (c, s) = m.state_from_inputs(layer, &inputs).unwrap();
    let (nc, ns) = kv.layers()[layer].as_ref().unwrap();
    let diff = |a: &candle_core::Tensor, b: &candle_core::Tensor| (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    println!("rebuilt state: conv rows within {:.2e}, scan state within {:.2e}", diff(&c, nc), diff(&s, ns));
    assert!(diff(&c, nc) < 1e-5 && diff(&s, ns) < 1e-5, "the rebuilt state differs from the forward pass's");
}

/// Granite 4.0 h-tiny with its linear layers AND its 64 experts quantized at load — the path a 32B
/// Granite (h-small, mostly experts) needs to fit one card.
#[test]
#[ignore = "integration"]
fn granite_4_0_h_tiny_at_q8_matches_transformers() {
    quantized_matches_transformers("granite-4.0-h-tiny", candle_core::quantized::GgmlDType::Q8_0, 1.5e-1);
}

#[test]
#[ignore = "integration"]
fn granite_4_0_h_350m_at_q8_matches_transformers() {
    quantized_matches_transformers("granite-4.0-h-350m", candle_core::quantized::GgmlDType::Q8_0, 1.5e-1);
}
