//! The whole 4B, and the whole 27B, in Rust against the oracle's canonical state after the same
//! prefill: every R, C, K, V and H. INTEGRATION: needs `just oracle-models` and `just oracle-state`.

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
/// MEASURED agreement, as a regression guard (relative Frobenius error, ‖ours − oracle‖ / ‖oracle‖):
/// The SAME bounds hold the 4B and the 27B (27B: layer 0 0.6%, median 4.1%, worst 8.2%, H 4.6%).
/// layer 0 — where our arithmetic and the oracle's are the same recurrence — within 1%; H within 5%;
/// every field within 15%. Every field from layer 1 on sits at ~5–7%: the ORACLE'S OWN rounding, not
/// the port. llama.cpp rounds every quantized matmul's activations to 8 bits (blocks of 32 on CUDA,
/// a Q4_K min term against the unrounded block sum); emulating that takes layer 0's output from 4.6%
/// to 0.7% of the oracle's, and the 4B amplifies any rounding difference through its depth — two
/// configurations of the oracle itself disagree by 3.1% at H. Tighter bounds would mean reproducing
/// each kernel's rounding. A structural error shows as >100%, so these bounds still fail loudly on one.
const LAYER_ZERO: f32 = 1e-2;
const H_BOUND: f32 = 5e-2;
const ANY_FIELD: f32 = 1.5e-1;

#[test]
#[ignore = "integration"]
fn every_field_of_the_4b_matches_the_oracle() {
    every_field_matches("4b", "Qwen3.5-4B-Q4_K_M.gguf");
}

#[test]
#[ignore = "integration"]
fn every_field_of_the_27b_matches_the_oracle() {
    every_field_matches("27b", "Qwen3.8-27B-MTP-Q4_K_M.gguf");
}

fn every_field_matches(model: &str, gguf: &str) {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{ROOT}target/oracle/state-{model}.json")).expect("run `just oracle-state`"),
    )
    .unwrap();
    let ids: Vec<u32> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let bytes = std::fs::read(format!("{ROOT}target/oracle/state-{model}.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let ours = prattle_core::qwen35::prefill(&format!("{ROOT}models/latentport-4-27b/models/{gguf}"), &ids).unwrap();
    let mut worst = (String::new(), 0f32);
    let mut failures = vec![];
    for name in st.names() {
        let oracle: Vec<f32> = st.tensor(name).unwrap().data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let Some(mine) = ours.get(&name.to_string()) else {
            failures.push(format!("{name}: missing"));
            continue;
        };
        if mine.len() != oracle.len() {
            failures.push(format!("{name}: {} values, oracle has {}", mine.len(), oracle.len()));
            continue;
        }
        let scale = oracle.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        let err = mine.iter().zip(&oracle).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale; // max-relative, reported
        let frob = (mine.iter().zip(&oracle).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>()
            / oracle.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().max(1e-30))
        .sqrt() as f32;
        if std::env::var("PRATTLE_ALL_ERRORS").is_ok() {
            println!("ERR {name} max {err:.3e} frob {frob:.3e}");
        }
        if frob > worst.1 {
            worst = (name.to_string(), frob);
        }
        let bound = if name.ends_with(".0") { LAYER_ZERO } else if name == "H" { H_BOUND } else { ANY_FIELD };
        if frob > bound {
            failures.push(format!("{name}: Frobenius error {frob:.2e} over {bound:.0e} (max-relative {err:.2e})"));
        }
    }
    println!("{model}: {} fields; worst Frobenius error {} at {:.2e}", st.names().len(), worst.0, worst.1);
    assert!(failures.is_empty(), "{} fields disagree with the oracle:\n{}", failures.len(), failures.join("\n"));
}
