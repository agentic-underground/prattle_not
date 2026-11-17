//! Upstream's 4B → 27B translator in Rust, stage by stage against the oracle's own run of it on one
//! prompt (`just oracle-translator`: its inputs, every stage between, and its outputs).
//! INTEGRATION: needs the bridge GGUF (`just oracle-home`) and the dump.

use prattle_core::translator::{Source, Translator};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
/// f32 stages: the same arithmetic on another device, so only summation order differs. MEASURED
/// 2.6e-7 – 6.5e-7 at every stage.
const STAGE: f32 = 1e-5;
/// Outputs are bf16 where upstream emits bf16; an f32 difference flips the odd rounding. MEASURED
/// worst field 1.2e-4 of 129.
const OUTPUT: f32 = 1e-3;

struct Dump(Vec<u8>);

impl Dump {
    fn open() -> Dump {
        Dump(std::fs::read(format!("{ROOT}target/oracle/translator.safetensors")).expect("run `just oracle-translator`"))
    }
    fn get(&self, name: &str) -> Vec<f32> {
        let st = safetensors::SafeTensors::deserialize(&self.0).unwrap();
        let t = st.tensor(name).unwrap_or_else(|_| panic!("the dump has no {name}"));
        t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
    fn names(&self) -> Vec<String> {
        safetensors::SafeTensors::deserialize(&self.0).unwrap().names().into_iter().map(str::to_string).collect()
    }
    fn source(&self) -> Source {
        let lengths = self.get("meta.lengths");
        let mut fields = std::collections::BTreeMap::new();
        for n in self.names() {
            if let Some(f) = n.strip_prefix("in.") {
                fields.insert(f.to_string(), self.get(&n));
            }
        }
        Source { fields, source_length: lengths[0] as usize, target_length: lengths[1] as usize, centers: self.get("meta.centers") }
    }
}

fn rel(ours: &[f32], oracle: &[f32]) -> f32 {
    assert_eq!(ours.len(), oracle.len(), "{} values, the oracle has {}", ours.len(), oracle.len());
    let num: f64 = ours.iter().zip(oracle).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = oracle.iter().map(|b| (*b as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn check(stage: &str, ours: &[f32], oracle: &[f32], bound: f32) {
    let e = rel(ours, oracle);
    println!("{stage}: relative Frobenius error {e:.2e}");
    assert!(e <= bound, "{stage}: {e:.2e} over {bound:.0e}");
}

fn translator() -> Translator {
    Translator::load(&format!("{ROOT}models/latentport-4-27b/latentport-4-27b-bridge.F32.gguf")).unwrap()
}

#[test]
#[ignore = "integration"]
fn the_persistent_encoder_matches() {
    let (d, t) = (Dump::open(), translator());
    let p = t.persistent(&d.source());
    check("memory", &p.memory, &d.get("parent.memory"), STAGE);
    check("g", &p.g, &d.get("parent.g"), STAGE);
}

#[test]
#[ignore = "integration"]
fn the_chunk_encoder_matches() {
    let (d, t) = (Dump::open(), translator());
    let s = d.source();
    let p = t.persistent(&s);
    check("content", &t.content(&s, &p, "parent"), &d.get("parent.content"), STAGE);
}

#[test]
#[ignore = "integration"]
fn both_position_routers_match() {
    let (d, t) = (Dump::open(), translator());
    let s = d.source();
    let p = t.persistent(&s);
    for branch in ["parent", "kv"] {
        let content = t.content(&s, &p, branch);
        check(&format!("{branch} routed"), &t.route(&s, &p, &content, branch), &d.get(&format!("{branch}.routed")), STAGE);
    }
}

#[test]
#[ignore = "integration"]
fn the_query_decoder_matches() {
    let (d, t) = (Dump::open(), translator());
    let s = d.source();
    let p = t.persistent(&s);
    let routed = t.route(&s, &p, &t.content(&s, &p, "parent"), "parent");
    check("q", &t.queries(&p, &routed, s.target_length), &d.get("parent.q"), STAGE);
}

#[test]
#[ignore = "integration"]
fn every_translated_field_matches() {
    let (d, t) = (Dump::open(), translator());
    let out = t.translate(&d.source());
    let mut failures = vec![];
    let mut worst = (String::new(), 0f32);
    for name in d.names().into_iter().filter(|n| n.starts_with("out.")) {
        let field = &name[4..];
        let Some(ours) = out.get(field) else {
            failures.push(format!("{field}: missing"));
            continue;
        };
        let e = rel(ours, &d.get(&name));
        if e > worst.1 {
            worst = (field.to_string(), e);
        }
        if e > OUTPUT {
            failures.push(format!("{field}: {e:.2e}"));
        }
    }
    println!("{} fields; worst {} at {:.2e}", out.len(), worst.0, worst.1);
    assert!(failures.is_empty(), "{} fields disagree:\n{}", failures.len(), failures.join("\n"));
}

/// The whole handoff in this project's own code: the Rust 4B reads the prompt, the Rust aligner and
/// translator carry its state to the 27B's, and the 27B's output head on the translated H gives the
/// first token — upstream's composite first token for the same prompt.
#[test]
#[ignore = "integration"]
fn the_rust_handoff_gives_upstreams_first_token() {
    let d = Dump::open();
    let ids = |name: &str| d.get(name).iter().map(|v| *v as u32).collect::<Vec<u32>>();
    let (source_ids, target_ids) = (ids("tokens.source"), ids("tokens.target"));
    let models = format!("{ROOT}models/latentport-4-27b");
    let fields = prattle_core::qwen35::prefill(&format!("{models}/models/Qwen3.5-4B-Q4_K_M.gguf"), &source_ids).unwrap();
    let tok = |side: &str| tokenizers::Tokenizer::from_file(format!("{models}/tokenizers/{side}/tokenizer.json")).unwrap();
    let centers = prattle_core::align::centers(
        &prattle_core::align::pieces(&tok("source"), &source_ids),
        &prattle_core::align::pieces(&tok("target"), &target_ids),
    );
    let source = Source { fields, source_length: source_ids.len(), target_length: target_ids.len(), centers };
    let out = translator().translate(&source);
    let first = prattle_core::qwen35::first_token(&format!("{models}/models/Qwen3.8-27B-MTP-Q4_K_M.gguf"), &out["H"]).unwrap();
    let h = rel(&out["H"], &d.get("out.H"));
    println!("translated H vs upstream's: {h:.2e}; first token {first}, upstream's {}", d.get("first.composite")[0]);
    assert_eq!(first, d.get("first.composite")[0] as u32, "the Rust handoff's first token is not upstream's");
}
