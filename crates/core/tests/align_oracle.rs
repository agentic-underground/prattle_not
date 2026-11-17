//! The numeric alignment between the 4B's and the 27B's tokens in Rust, against the centers upstream's
//! own aligner gave the translator on the fixed prompt (`just oracle-translator`). INTEGRATION.

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");

fn dump(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(format!("{ROOT}target/oracle/translator.safetensors")).expect("run `just oracle-translator`");
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensor(name).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn pieces(side: &str, ids: &[f32]) -> Vec<Vec<u8>> {
    let tok = tokenizers::Tokenizer::from_file(format!("{ROOT}models/latentport-4-27b/tokenizers/{side}/tokenizer.json")).unwrap();
    prattle_core::align::pieces(&tok, &ids.iter().map(|v| *v as u32).collect::<Vec<_>>())
}

#[test]
#[ignore = "integration"]
fn the_centers_are_upstreams() {
    let (source, target) = (pieces("source", &dump("tokens.source")), pieces("target", &dump("tokens.target")));
    let ours = prattle_core::align::centers(&source, &target);
    let oracle = dump("meta.centers");
    assert_eq!(ours.len(), oracle.len(), "{} centers, upstream gave {}", ours.len(), oracle.len());
    let worst = ours.iter().zip(&oracle).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("{} centers; worst difference {worst:.2e}", ours.len());
    assert!(worst < 1e-5, "a center differs by {worst}");
}
