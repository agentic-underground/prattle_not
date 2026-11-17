//! LatentPort's one-token handoff on the GPU engine against prattle-core's on the CPU, from the same
//! weights and the same fitted bridge: SmolLM2-135M reads all but the last prompt token, the bridge
//! translates its residual streams into SmolLM2-360M's KV cache, and the 360M reads the last token
//! itself. Its logits, then its greedy reply, must agree.

use candle_core::{Device, Tensor};
use prattle_webgpu::{Gpu, bridge::Bridge, llama::Llama};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");

fn read(path: &str) -> Vec<u8> {
    std::fs::read(format!("{MODELS}{path}")).unwrap_or_else(|e| panic!("{MODELS}{path}: {e} — `just models` / `just bridge-fetch`"))
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

const PROMPT: [u32; 23] = [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198, 1, 4093];

#[test]
#[ignore = "integration"]
fn smollm2_handoff_on_the_gpu_equals_candle() {
    let (sc, sw) = (read("SmolLM2-135M-Instruct/config.json"), read("SmolLM2-135M-Instruct/model.safetensors"));
    let (tc, tw) = (read("SmolLM2-360M-Instruct/config.json"), read("SmolLM2-360M-Instruct/model.safetensors"));
    let map_bytes = read("bridges/smollm2-135m-to-360m.safetensors");
    let n = PROMPT.len();

    // candle, the reference
    let source = prattle_core::model::load(&sc, sw.clone(), &Device::Cpu).unwrap();
    let target = prattle_core::model::load(&tc, tw.clone(), &Device::Cpu).unwrap();
    let map = prattle_core::bridge::KvMap::from_bytes(&map_bytes).unwrap();
    let ids = |s: &[u32]| Tensor::new(s, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let mut caps = Vec::new();
    source.forward_capture(&ids(&PROMPT[..n - 1]), 0, &mut source.empty_state(), Some(&mut caps)).unwrap();
    let mut kv = prattle_core::bridge::translate_hidden(&map, &caps, &target).unwrap();
    let want = target.forward(&ids(&PROMPT[n - 1..]), n - 1, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // the GPU engine
    let gpu = pollster::block_on(Gpu::new()).expect("a GPU adapter");
    let mut src = Llama::load(&gpu, &sc, &sw, 256).unwrap();
    let mut tgt = Llama::load(&gpu, &tc, &tw, 256).unwrap();
    let bridge = Bridge::load(&gpu, &map_bytes).unwrap();
    let hidden = src.prefill_capture(&gpu, &PROMPT[..n - 1]);
    let kvs = bridge.translate(&gpu, &hidden, &tgt);
    tgt.install(&gpu, &kvs, n - 1);
    let got = pollster::block_on(tgt.forward(&gpu, &PROMPT[n - 1..]));

    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    println!("bridged logits: worst difference {worst:.2e} against a largest value {scale:.2}");
    assert!(worst / scale < 1e-3, "bridged logits differ by {worst} against a largest value {scale}");

    let (mut theirs, mut ours) = (vec![argmax(&want)], vec![argmax(&got)]);
    for step in 0..8 {
        let pos = n + step;
        theirs.push(argmax(&target.forward(&ids(&[*theirs.last().unwrap()]), pos, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()));
        ours.push(argmax(&pollster::block_on(tgt.forward(&gpu, &[*ours.last().unwrap()]))));
    }
    println!("bridged greedy reply: {ours:?}");
    assert_eq!(ours, theirs, "the bridged greedy replies part");
}
