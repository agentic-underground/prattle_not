//! Qwen3.5 on the GPU engine against prattle-core's candle Qwen3.5 on the CPU, from the same files:
//! the 0.8B's prompt logits and greedy reply, then LatentPort's handoff 0.8B → 2B — the 2B's attention
//! layers from the fitted bridge, its Gated DeltaNet layers from the 0.8B's own state. The GPU holds
//! the weights at f16 and candle at f32, so logits agree to f16's precision, and the greedy tokens
//! exactly.

use candle_core::{Device, Tensor};
use prattle_core::text::Template;
use prattle_webgpu::{Gpu, Weights, bridge::Bridge, decoder::Decoder, qwen35, upload_file};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
const WEIGHTS: &str = "model.safetensors-00001-of-00001.safetensors";

fn read(path: &str) -> Vec<u8> {
    std::fs::read(format!("{MODELS}{path}")).unwrap_or_else(|e| panic!("{MODELS}{path}: {e} — `just models` / `just bridge-fetch`"))
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

fn prompt(dir: &str) -> Vec<u32> {
    let tok = tokenizers::Tokenizer::from_bytes(read(&format!("{dir}/tokenizer.json"))).unwrap();
    let text = Template::from_config(&read(&format!("{dir}/config.json"))).unwrap().prompt("What is the capital of France? Answer in one word.");
    tok.encode(text, false).unwrap().get_ids().to_vec()
}

fn on_gpu(gpu: &Gpu, dir: &str) -> Decoder {
    let config = read(&format!("{dir}/config.json"));
    let mut w = Weights::default();
    upload_file(gpu, &read(&format!("{dir}/{WEIGHTS}")), qwen35::rule, &mut w).unwrap();
    Decoder::build(gpu, &config, w, 512).unwrap()
}

/// Logits within f16's reach of candle's: the worst difference against the largest value.
fn agree(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: vocabulary");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    println!("{what}: worst difference {worst:.2e} against a largest value {scale:.2}");
    assert!(worst / scale < 2e-2, "{what}: logits differ by {worst} against a largest value {scale}");
    assert_eq!(argmax(got), argmax(want), "{what}: the most likely token");
}


/// One GPU test at a time in this binary: `cargo test` runs tests on parallel threads, and Vulkan
/// devices torn down at once deadlock inside the NVIDIA driver (vkDestroyDevice). Take the turn
/// BEFORE making the device, so the device is dropped first and its teardown happens inside the turn.
fn turn() -> std::sync::MutexGuard<'static, ()> {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
#[ignore = "integration"]
fn qwen35_08b_logits_and_greedy_reply_equal_candle() {
    let dir = "Qwen3.5-0.8B";
    let ids = prompt(dir);
    let reference = prattle_core::qwen3_5::Qwen35::load(&read(&format!("{dir}/config.json")), read(&format!("{dir}/{WEIGHTS}")), &Device::Cpu).unwrap();
    let mut kv = reference.empty_state();
    let t = |s: &[u32]| Tensor::new(s, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let want = reference.forward(&t(&ids), 0, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let _turn = turn();
    let gpu = pollster::block_on(Gpu::new()).expect("a GPU adapter");
    let mut engine = on_gpu(&gpu, dir);
    let got = pollster::block_on(engine.forward(&gpu, &ids));
    agree(&got, &want, "0.8B prompt logits");

    let (mut theirs, mut ours) = (vec![argmax(&want)], vec![argmax(&got)]);
    for step in 0..8 {
        let pos = ids.len() + step;
        theirs.push(argmax(&reference.forward(&t(&[*theirs.last().unwrap()]), pos, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()));
        ours.push(argmax(&pollster::block_on(engine.forward(&gpu, &[*ours.last().unwrap()]))));
    }
    println!("0.8B greedy reply: {ours:?}");
    assert_eq!(ours, theirs, "the greedy replies part");
}

#[test]
#[ignore = "integration"]
fn qwen35_handoff_08b_to_2b_on_the_gpu_equals_candle() {
    let (s, tg) = ("Qwen3.5-0.8B", "Qwen3.5-2B");
    let ids = prompt(s);
    let n = ids.len();
    let map_bytes = read("bridges/qwen3.5-0.8b-to-2b.safetensors");

    // candle, the reference: the attention layers translated, the recurrent ones the source's own
    let source = prattle_core::qwen3_5::Qwen35::load(&read(&format!("{s}/config.json")), read(&format!("{s}/{WEIGHTS}")), &Device::Cpu).unwrap();
    let target = prattle_core::qwen3_5::Qwen35::load(&read(&format!("{tg}/config.json")), read(&format!("{tg}/{WEIGHTS}")), &Device::Cpu).unwrap();
    let map = prattle_core::bridge::KvMap::from_bytes(&map_bytes).unwrap();
    let t = |x: &[u32]| Tensor::new(x, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let (mut caps, mut state) = (Vec::new(), source.empty_state());
    source.forward_capture(&t(&ids[..n - 1]), 0, &mut state, Some(&mut caps)).unwrap();
    let translated = prattle_core::bridge::translate_hidden(&map, &caps, &target).unwrap();
    let mut kv = prattle_core::bridge::fill_from_source(translated, &state).unwrap();
    let want = target.forward(&t(&ids[n - 1..]), n - 1, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // the GPU engine
    let _turn = turn();
    let gpu = pollster::block_on(Gpu::new()).expect("a GPU adapter");
    let mut src = on_gpu(&gpu, s);
    let mut tgt = on_gpu(&gpu, tg);
    let bridge = Bridge::load(&gpu, &map_bytes).unwrap();
    bridge.fits(&tgt).unwrap();
    let hidden = src.prefill_capture(&gpu, &ids[..n - 1]);
    let kvs = bridge.translate(&gpu, &hidden, &tgt);
    tgt.install(&gpu, &kvs, n - 1, &src).unwrap();
    let got = pollster::block_on(tgt.forward(&gpu, &ids[n - 1..]));
    agree(&got, &want, "bridged 2B logits");

    let (mut theirs, mut ours) = (vec![argmax(&want)], vec![argmax(&got)]);
    for step in 0..8 {
        let pos = n + step;
        theirs.push(argmax(&target.forward(&t(&[*theirs.last().unwrap()]), pos, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()));
        ours.push(argmax(&pollster::block_on(tgt.forward(&gpu, &[*ours.last().unwrap()]))));
    }
    println!("bridged 2B greedy reply: {ours:?}");
    assert_eq!(ours, theirs, "the bridged greedy replies part");
}
