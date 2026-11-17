//! The native arm of upstream's TTFT experiment, through the oracle: the target model reads the
//! whole rendered prompt itself, and its first token is timed. INTEGRATION — needs `just oracle`
//! and `just oracle-models`.

use oracle::{Oracle, Side};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/latentport-4-27b/");

fn context_tokens() -> usize {
    std::env::var("PRATTLE_CONTEXT_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(16384)
}

#[test]
#[ignore = "integration"]
fn the_target_reads_the_whole_prompt_and_its_first_token_is_timed() {
    let gguf = std::env::var("PRATTLE_NATIVE_GGUF")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("{DIR}models/Qwen3.8-27B-MTP-Q4_K_M.gguf"));
    let side = if gguf.contains("27B") { Side::Target } else { Side::Source };
    let template_side = if side == Side::Target { "target" } else { "source" };
    let kwargs = if side == Side::Target {
        serde_json::json!({"preserve_thinking": false, "reasoning_effort": "medium"})
    } else {
        serde_json::json!({})
    };
    let oracle = Oracle::load_default().unwrap();
    let (body, counted) = oracle::prompt::incident_log(
        &format!("{DIR}tokenizers/source/tokenizer.json"),
        context_tokens(),
        oracle::prompt::DEFAULT_QUESTION,
    )
    .unwrap();
    assert_eq!(counted, context_tokens(), "the log was not truncated to the asked length");
    let template = std::fs::read_to_string(format!("{DIR}tokenizers/{template_side}/chat_template.jinja")).unwrap();
    let mut model = oracle
        .open(&gguf, side, context_tokens() + 2048)
        .unwrap()
        .with_tokenizer(&format!("{DIR}tokenizers/{template_side}/tokenizer.json"))
        .unwrap();
    let run = model.native_first_token(&template, &kwargs, &body).unwrap();
    println!(
        "native {}: {} prompt tokens, first token {} after {:.3} s (prefill counter +{})",
        gguf.rsplit('/').next().unwrap(),
        run.prompt_tokens,
        run.first_token,
        run.seconds,
        run.prefill_delta
    );
    let tok = tokenizers::Tokenizer::from_file(format!("{DIR}tokenizers/{template_side}/tokenizer.json")).unwrap();
    let first = tok.decode(&[run.first_token as u32], false).unwrap();
    println!("first token decodes to {first:?}");
    assert!(!first.trim().is_empty() || first.contains('\n'), "the first token decodes to nothing: {first:?}");
    assert_eq!(run.prefill_delta, run.prompt_tokens as u64, "the target did not read the whole prompt itself");
    assert!(run.seconds > 0.0);
}
