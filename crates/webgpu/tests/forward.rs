//! SmolLM2-135M run end to end on the GPU engine against prattle-core's candle Llama on the CPU, from
//! the same weights: the last position's logits, then a greedy reply token by token through each
//! engine's KV cache.

use candle_core::{Device, Tensor};
use prattle_webgpu::{Gpu, llama::Llama};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/SmolLM2-135M-Instruct/");

fn read(file: &str) -> Vec<u8> {
    std::fs::read(format!("{DIR}{file}")).unwrap_or_else(|e| panic!("{DIR}{file}: {e} — `just models` fetches it"))
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

/// A prompt of 23 tokens: real SmolLM2 ids, so the attention sees a realistic spread of keys.
const PROMPT: [u32; 23] = [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198, 1, 4093];

#[test]
#[ignore = "integration"]
fn smollm2_135m_logits_and_greedy_reply_equal_candle() {
    let (config, weights) = (read("config.json"), read("model.safetensors"));
    let reference = prattle_core::model::load(&config, weights.clone(), &Device::Cpu).unwrap();
    let mut kv = reference.empty_state();
    let gpu = pollster::block_on(Gpu::new()).expect("a GPU adapter");
    let mut engine = Llama::load(&gpu, &config, &weights, 256).unwrap();

    let ids = Tensor::new(&PROMPT[..], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let want = reference.forward(&ids, 0, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let got = pollster::block_on(engine.forward(&gpu, &PROMPT));
    assert_eq!(got.len(), want.len(), "vocabulary");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    println!("prompt logits: worst difference {worst:.2e} against a largest value {scale:.2}");
    assert!(worst / scale < 1e-3, "prompt logits differ by {worst} against a largest value {scale}");

    let (mut theirs, mut ours) = (vec![argmax(&want)], vec![argmax(&got)]);
    for step in 0..12 {
        let pos = PROMPT.len() + step;
        let next = Tensor::new(&[*theirs.last().unwrap()], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        theirs.push(argmax(&reference.forward(&next, pos, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()));
        ours.push(argmax(&pollster::block_on(engine.forward(&gpu, &[*ours.last().unwrap()]))));
    }
    println!("greedy reply: {ours:?}");
    assert_eq!(ours, theirs, "the greedy replies part");
}
