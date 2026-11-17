//! The handoff on real weights, on the host. Needs `just models`; INTEGRATION, because the weights
//! are not in the repository.

use prattle_core::{Model, Step};

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
