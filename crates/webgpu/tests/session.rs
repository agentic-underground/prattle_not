//! The GPU model and prattle-core's candle Model, given the same weights, tokenizer and bridge, must
//! write the same replies in every arm the page runs: native, bridged, restored and wrong-memory.

use prattle_core::{Arm, Model, Step};
use prattle_webgpu::model::GpuModel;

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");

fn read(path: &str) -> Vec<u8> {
    std::fs::read(format!("{MODELS}{path}")).unwrap_or_else(|e| panic!("{MODELS}{path}: {e}"))
}

const STEPS: usize = 24;

fn cpu_reply(m: &mut Model, q: &str, arm: Arm) -> (String, [usize; 4]) {
    m.start_arm(q, arm).unwrap();
    let c = m.counters();
    let mut text = String::new();
    for _ in 0..STEPS {
        match m.step().unwrap() {
            Step::Text(t) => text.push_str(&t),
            Step::Done => break,
        }
    }
    (text, [c.prompt_tokens, c.reader_tokens, c.generator_history, c.bridge_tokens])
}

fn gpu_reply(m: &mut GpuModel, q: &str, arm: Arm) -> (String, [usize; 4]) {
    m.start_arm(q, arm).unwrap();
    let c = m.counters();
    let mut text = String::new();
    for _ in 0..STEPS {
        match pollster::block_on(m.step()).unwrap() {
            Step::Text(t) => text.push_str(&t),
            Step::Done => break,
        }
    }
    (text, [c.prompt_tokens, c.reader_tokens, c.generator_history, c.bridge_tokens])
}

#[test]
#[ignore = "integration"]
fn every_arm_writes_the_same_reply_as_candle() {
    let (sc, tok, sw) = (read("SmolLM2-135M-Instruct/config.json"), read("SmolLM2-135M-Instruct/tokenizer.json"), read("SmolLM2-135M-Instruct/model.safetensors"));
    let (tc, tw) = (read("SmolLM2-360M-Instruct/config.json"), read("SmolLM2-360M-Instruct/model.safetensors"));
    let bridge = read("bridges/smollm2-135m-to-360m.safetensors");
    let mut cpu = Model::from_bytes(&sc, &tok, sw.clone()).unwrap();
    cpu.attach_target_with(&tc, tw.clone(), Some(&bridge)).unwrap();
    let mut gpu = pollster::block_on(GpuModel::from_bytes(&sc, &tok, &sw)).unwrap();
    gpu.attach_target(&tc, &tw, Some(&bridge)).unwrap();

    let (q, donor) = ("What is the capital of France?", "Name three colours.");
    for (name, arm_cpu, arm_gpu) in [
        ("native", Arm::Native, Arm::Native),
        ("bridge", Arm::Bridge, Arm::Bridge),
        ("restore", Arm::Restore, Arm::Restore),
        ("donor", Arm::WrongDonor(donor.into()), Arm::WrongDonor(donor.into())),
        ("target", Arm::Target, Arm::Target),
    ] {
        let want = cpu_reply(&mut cpu, q, arm_cpu);
        let got = gpu_reply(&mut gpu, q, arm_gpu);
        println!("{name}: {:?} {:?}", got.0, got.1);
        assert_eq!(got.1, want.1, "{name}: the counters differ");
        assert_eq!(got.0, want.0, "{name}: the replies differ");
    }
}
