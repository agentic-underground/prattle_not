//! A fitted bridge's measured SPEED-UP to first token on this host — what the browser's Measure does,
//! on the server: the same `Model` arms, the same token clock (`start_arm` then the first `step`,
//! nothing excluded — the bridge arm's source prefill is inside), E120's design and statistic.
//!
//!   speedup <source dir> <target dir> <bridge.safetensors> [tokens=256] [prompts=8] [reps=3]
//!
//! Three arms per prompt and repetition — Target (reads everything), Bridge, Source (the ceiling's
//! denominator) — in rotated order (`measure::first_arm`), one discarded warm-up each. Every run's
//! counters are checked before anything is computed: a bridged run whose target read prompt history
//! is refused, as `reproduce table` refuses the oracle's.

use std::time::Instant;

use prattle_core::measure::{Pair, first_arm, summarize};
use prattle_core::{Arm, Model};

/// The three arms by index: 0 Target (reads everything), 1 Bridge, 2 Source.
fn arm(i: usize) -> Arm {
    match i {
        0 => Arm::Target,
        1 => Arm::Bridge,
        _ => Arm::Source,
    }
}

fn first_token_ms(m: &mut Model, prompt: &str, arm: Arm) -> f64 {
    let started = Instant::now();
    m.start_arm(prompt, arm).unwrap();
    m.step().unwrap();
    started.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() >= 4, "usage: speedup <source dir> <target dir> <bridge> [tokens] [prompts] [reps]");
    let arg = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (tokens, prompts, reps) = (arg(4, 256), arg(5, 8), arg(6, 3));
    let read = |dir: &str, f: &str| std::fs::read(format!("{dir}/{f}")).unwrap_or_else(|e| panic!("{dir}/{f}: {e}"));
    let weights = |dir: &str| prattle_core::weights::read_dir(dir).unwrap();
    // One `Model` holds both arms, as the page does: the source on the first device, the target and
    // its bridge on the second (`prattle_core::device`; the same one if only one is named). Each
    // step's token reaches the host, so a GPU's queued work is inside the clock.
    let (device, target_device) = prattle_core::device::pair().unwrap();
    let mut m = Model::from_bytes_in(&read(&a[1], "config.json"), &read(&a[1], "tokenizer.json"), weights(&a[1]), device, prattle_core::device::source_dtype().unwrap()).unwrap();
    m.place_target_on(target_device.clone());
    m.target_weights_in(prattle_core::device::target_dtype().unwrap());
    m.target_weights_quantized(prattle_core::device::target_quant().unwrap());
    let bridge = std::fs::read(&a[3]).unwrap_or_else(|e| panic!("{}: {e}", a[3]));
    let target = prattle_core::fast_llm::target_decoder(&a[2], &target_device).unwrap();
    m.attach_target_decoder(target, Some(&bridge)).unwrap();
    m.set_bridge_tokens(1); // the paper's one-token handoff, which the corrected bridges answer with

    let texts: Vec<String> = (0..prompts as u32).map(|v| m.synthetic_prompt(tokens, v).unwrap()).collect();
    for i in 0..3 {
        first_token_ms(&mut m, &texts[0], arm(i)); // one discarded warm-up per arm
    }
    let (mut speed, mut ceiling) = (Vec::new(), Vec::new());
    for rep in 0..reps {
        for (p, text) in texts.iter().enumerate() {
            let mut ms = [0.0; 3];
            for k in 0..3 {
                let i = (first_arm(p, rep, 3) + k) % 3;
                ms[i] = first_token_ms(&mut m, text, arm(i));
                let c = m.counters();
                if i == 1 {
                    assert!(c.generator_history == 0, "prompt {p}: the bridged target read {} prompt tokens itself", c.generator_history);
                } else {
                    assert!(c.reader_tokens == 0 && c.bridge_tokens == 0, "prompt {p}: a reading arm used the bridge");
                }
            }
            speed.push(Pair { prompt: p, native_ms: ms[0], bridge_ms: ms[1] });
            ceiling.push(Pair { prompt: p, native_ms: ms[0], bridge_ms: ms[2] });
            eprintln!("prompt {p} rep {rep}: target {:.0} ms, bridge {:.0} ms, source {:.0} ms", ms[0], ms[1], ms[2]);
        }
    }
    let (s, c) = (summarize(&speed).unwrap(), summarize(&ceiling).unwrap());
    println!(
        "speed-up {:.2}× (95% CI {:.2}–{:.2}, median {:.2}×) of a ceiling {:.2}× — ~{tokens} tokens, {} prompts × {reps}, one bridge token",
        s.speedup, s.ci_low, s.ci_high, s.median_ratio, c.speedup, s.prompts
    );
}
