//! Qwen3 on the GPU engine against prattle-core's candle Llama-family model on the CPU, from the same
//! files: the 0.6B's prompt logits and greedy reply, then the bridged handoff 0.6B → 1.7B. The GPU
//! holds Qwen3's matrices at f16 and candle at f32, so logits agree to f16's precision and the
//! greedy tokens exactly.

use candle_core::{Device, Tensor};
use prattle_core::text::Template;
use prattle_webgpu::{Gpu, Weights, bridge::Bridge, decoder::{self, Decoder}, upload_file};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");

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
    upload_file(gpu, &read(&format!("{dir}/model.safetensors")), decoder::rule(&config), &mut w).unwrap();
    Decoder::build(gpu, &config, w, 512).unwrap()
}

fn agree(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: vocabulary");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    println!("{what}: worst difference {worst:.2e} against a largest value {scale:.2}");
    assert!(worst / scale < 2e-2, "{what}: logits differ by {worst} against a largest value {scale}");
    assert_eq!(argmax(got), argmax(want), "{what}: the most likely token");
}

#[test]
#[ignore = "integration"]
fn qwen3_handoff_06b_to_17b_on_the_gpu_equals_candle() {
    let (s, tg) = ("Qwen3-0.6B", "Qwen3-1.7B");
    let ids = prompt(s);
    let n = ids.len();
    let map_bytes = read("bridges/qwen3-0.6b-to-1.7b.safetensors");
    let t = |x: &[u32]| Tensor::new(x, &Device::Cpu).unwrap().unsqueeze(0).unwrap();

    let source = prattle_core::model::load(&read(&format!("{s}/config.json")), read(&format!("{s}/model.safetensors")), &Device::Cpu).unwrap();
    let target = prattle_core::model::load(&read(&format!("{tg}/config.json")), read(&format!("{tg}/model.safetensors")), &Device::Cpu).unwrap();
    let gpu = pollster::block_on(Gpu::new()).expect("a GPU adapter");
    let mut src = on_gpu(&gpu, s);
    let mut tgt = on_gpu(&gpu, tg);

    // the 0.6B reading the whole prompt itself
    let want = source.forward(&t(&ids), 0, &mut source.empty_state()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let got = pollster::block_on(src.forward(&gpu, &ids));
    agree(&got, &want, "0.6B prompt logits");

    // the handoff
    let map = prattle_core::bridge::KvMap::from_bytes(&map_bytes).unwrap();
    let mut caps = Vec::new();
    source.forward_capture(&t(&ids[..n - 1]), 0, &mut source.empty_state(), Some(&mut caps)).unwrap();
    let mut kv = prattle_core::bridge::translate_hidden(&map, &caps, &target).unwrap();
    let want = target.forward(&t(&ids[n - 1..]), n - 1, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bridge = Bridge::load(&gpu, &map_bytes).unwrap();
    bridge.fits(&tgt).unwrap();
    src.reset(&gpu);
    let hidden = src.prefill_capture(&gpu, &ids[..n - 1]);
    let kvs = bridge.translate(&gpu, &hidden, &tgt);
    tgt.install(&gpu, &kvs, n - 1, &src).unwrap();
    let got = pollster::block_on(tgt.forward(&gpu, &ids[n - 1..]));
    agree(&got, &want, "bridged 1.7B logits");

    let (mut theirs, mut ours) = (vec![argmax(&want)], vec![argmax(&got)]);
    for step in 0..8 {
        let pos = n + step;
        theirs.push(argmax(&target.forward(&t(&[*theirs.last().unwrap()]), pos, &mut kv).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()));
        ours.push(argmax(&pollster::block_on(tgt.forward(&gpu, &[*ours.last().unwrap()]))));
    }
    println!("bridged 1.7B greedy reply: {ours:?}");
    assert_eq!(ours, theirs, "the bridged greedy replies part");
}
