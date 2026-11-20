//! The handoff on real weights, on the host. Needs `just models`; INTEGRATION, because the weights
//! are not in the repository.

use prattle_core::{Arm, Model, Role, Step, Turns};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");

fn read(name: &str, file: &str) -> Vec<u8> {
    std::fs::read(format!("{DIR}{name}/{file}")).unwrap_or_else(|e| panic!("{name}/{file}: {e} — run `just models`"))
}

fn instruct() -> Model {
    let m = "SmolLM2-135M-Instruct";
    Model::from_bytes(&read(m, "config.json"), &read(m, "tokenizer.json"), read(m, "model.safetensors")).unwrap()
}

fn reply(model: &mut Model, q: &str, bridge: bool, max: usize) -> String {
    model.start(q, bridge).unwrap();
    let mut out = String::new();
    for _ in 0..max {
        match model.step().unwrap() {
            Step::Text(t) => out.push_str(&t),
            Step::Done => break,
        }
    }
    out
}

const Q: &str = "What is the capital of France?";

#[test]
#[ignore = "integration"]
fn restoring_a_models_own_state_reproduces_its_native_reply() {
    let mut m = instruct();
    let native = reply(&mut m, Q, false, 12);
    m.attach_target(&read("SmolLM2-135M-Instruct", "config.json"), read("SmolLM2-135M-Instruct", "model.safetensors")).unwrap();
    let restored = reply(&mut m, Q, true, 12);
    assert!(!native.trim().is_empty());
    assert_eq!(restored, native, "self-handoff changed the reply");
}

#[test]
#[ignore = "integration"]
fn the_base_model_answers_from_the_instruct_models_reading() {
    // On its own the base model cannot follow the chat template; from the Instruct model's state
    // it must at least produce a non-degenerate reply that differs from what it does alone.
    let base = "SmolLM2-135M";
    let mut alone = Model::from_bytes(&read(base, "config.json"), &read("SmolLM2-135M-Instruct", "tokenizer.json"), read(base, "model.safetensors")).unwrap();
    let native = reply(&mut alone, Q, false, 24);
    let mut m = instruct();
    m.attach_target(&read(base, "config.json"), read(base, "model.safetensors")).unwrap();
    let bridged = reply(&mut m, Q, true, 24);
    println!("base alone: {native:?}\nbase from the bridge: {bridged:?}");
    assert!(bridged.split_whitespace().count() >= 3, "degenerate bridged reply: {bridged:?}");
    assert_ne!(bridged, native);
}

/// A model bridged into itself copies its state exactly, so every way of reading a conversation's later
/// turns gives the model-alone replies, turn for turn — a prefilled exchange in the history included.
#[test]
#[ignore = "integration"]
fn a_self_handoff_converses_as_the_model_alone_in_every_mode() {
    let m = "SmolLM2-135M-Instruct";
    let turns = [
        (Role::User, "My name is Ada. Remember it."),
        (Role::Assistant, "Hello Ada, I will remember your name."),
        (Role::User, "What is the capital of France?"),
        (Role::User, "And what is my name?"),
    ];
    let mut model = instruct();
    model.attach_target(&read(m, "config.json"), read(m, "model.safetensors")).unwrap();
    model.set_bridge_tokens(3);
    let alone = model.converse(Some("You are terse."), &turns, Arm::Target, Turns::Native, 16).unwrap();
    assert_eq!(alone.len(), 2, "one reply per user turn after the last assistant turn");
    assert!(alone.iter().all(|r| !r.trim().is_empty()), "an empty reply: {alone:?}");
    for mode in [Turns::Native, Turns::Reread, Turns::Append] {
        let bridged = model.converse(Some("You are terse."), &turns, Arm::Bridge, mode, 16).unwrap();
        assert_eq!(bridged, alone, "{mode:?}: the self-handoff's conversation differs from the model alone's");
    }
}

/// Forced logits over a fixed continuation: one row per continuation token, the target reading the prompt
/// predicts the first token as the model alone does, and a self-handoff's bridged rows are the model's own.
#[test]
#[ignore = "integration"]
fn forced_logits_give_a_row_per_token_and_a_self_handoff_matches_the_model() {
    let m = "SmolLM2-135M-Instruct";
    let mut model = instruct();
    model.attach_target(&read(m, "config.json"), read(m, "model.safetensors")).unwrap();
    model.set_bridge_tokens(3);
    let continuation = [504u32, 3575, 282, 4649, 314];
    let first = model.forced_logits(Q, Arm::Target, &continuation).unwrap();
    assert_eq!(first.len(), continuation.len(), "one row per continuation token");
    let argmax = |r: &[f32]| r.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
    let mut alone = instruct();
    let greedy = alone.forced_logits(Q, Arm::Native, &continuation[..1]).unwrap();
    assert_eq!(argmax(&first[0]), argmax(&greedy[0]), "the arms disagree on the first token");
    let bridged = model.forced_logits(Q, Arm::Bridge, &continuation).unwrap();
    for (i, (a, b)) in first.iter().zip(&bridged).enumerate() {
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-3, "row {i}: the self-handoff's logits differ from the model's by {worst}");
    }
}

/// The index of the largest logit.
fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
}

/// The reply in progress, as text, up to `max` tokens.
fn text(m: &mut Model, max: usize) -> String {
    let mut out = String::new();
    for _ in 0..max {
        match m.step().unwrap() {
            Step::Text(t) => out.push_str(&t),
            Step::Done => break,
        }
    }
    out
}

/// A caller that renders its own prompt (a leaderboard harness) passes ids, and Prattle adds no template:
/// the template's own ids give the templated path's logits, and raw ids are read exactly as given.
#[test]
#[ignore = "integration"]
fn raw_ids_are_read_as_given_and_the_templates_ids_match_the_templated_path() {
    let mut m = instruct();
    let c = m.tokenizer().encode(" Paris", false).unwrap().get_ids().to_vec();
    let templated = m.forced_logits(Q, Arm::Native, &c).unwrap();
    let ids = m.prompt_ids(Q).unwrap();
    let given = m.forced_logits_ids(&ids, Arm::Native, &c).unwrap();
    assert!(given == templated, "the template's ids gave other logits ({} rows against {})", given.len(), templated.len());
    let raw = m.tokenizer().encode("The capital of France is", false).unwrap().get_ids().to_vec();
    m.start_arm_ids(&raw, Arm::Native).unwrap();
    assert_eq!(m.counters().prompt_tokens, raw.len(), "a template was added to raw ids");
}

/// With no repeat penalty, the reply is the chain of plain argmaxes of the model's own logits.
#[test]
#[ignore = "integration"]
fn without_a_repeat_penalty_each_token_is_the_plain_argmax() {
    let mut m = instruct();
    let raw = m.tokenizer().encode("A B C D E F G", false).unwrap().get_ids().to_vec();
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| m.tokenizer().token_to_id(t)).collect();
    let mut chain: Vec<u32> = Vec::new();
    while chain.len() < 48 {
        let probe: Vec<u32> = chain.iter().copied().chain([0]).collect();
        let next = argmax(m.forced_logits_ids(&raw, Arm::Native, &probe).unwrap().last().expect("a row per token"));
        if stop.contains(&next) {
            break;
        }
        chain.push(next);
    }
    assert!(chain.len() >= 16, "too short a chain to test: {chain:?}");
    let plain = m.tokenizer().decode(&chain, true).unwrap();
    // the prompt is one the default penalty changes, so this test can tell the two apart
    m.start_arm_ids(&raw, Arm::Native).unwrap();
    assert_ne!(text(&mut m, 48), plain, "the default penalty left this prompt's reply unchanged");
    m.set_repeat_penalty(None);
    m.start_arm_ids(&raw, Arm::Native).unwrap();
    assert_eq!(text(&mut m, 48), plain);
}

/// A reply may run past the default cap when the caller raises it (IFEval allows 1,280 tokens).
#[test]
#[ignore = "integration"]
fn a_raised_cap_lets_a_reply_run_past_the_default() {
    let mut m = instruct();
    // without the penalty, this reply runs to 400 tokens on SmolLM2-135M-Instruct
    let raw = m.tokenizer().encode("def f():\n    return", false).unwrap().get_ids().to_vec();
    m.set_repeat_penalty(None);
    m.set_max_new_tokens(prattle_core::model::MAX_NEW_TOKENS + 64);
    m.start_arm_ids(&raw, Arm::Native).unwrap();
    let mut steps = 0;
    while let Step::Text(_) = m.step().unwrap() {
        steps += 1;
    }
    assert!(steps > prattle_core::model::MAX_NEW_TOKENS, "stopped at {steps}");
}
