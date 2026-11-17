//! E120's statistic over a raw oracle record — `just reproduce`. The oracle (upstream's library)
//! produces the timings; `prattle_core::measure` is the one definition of the statistic, the same
//! one the page's Measure button uses.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct LengthRow {
    pub length: u64,
    pub speedup: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub median_ratio: f64,
    pub prompts: usize,
    pub pairs: usize,
    pub native_median_s: f64,
    pub composite_median_s: f64,
}

/// One row per length. Refuses a record whose counters break the claim: a composite run in which
/// the target read prompt tokens, or a native run in which the source did.
pub fn summarize(record: &Value) -> Result<Vec<LengthRow>, String> {
    let lengths = record["lengths"].as_array().ok_or("the record has no lengths")?;
    lengths.iter().map(summarize_length).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.is_empty() { f64::NAN } else if v.len() % 2 == 1 { v[m] } else { (v[m - 1] + v[m]) / 2.0 }
}

fn summarize_length(entry: &Value) -> Result<LengthRow, String> {
    let length = entry["length"].as_u64().ok_or("a length entry without a length")?;
    let runs = entry["runs"].as_array().ok_or("a length entry without runs")?;
    let mut native = std::collections::BTreeMap::new();
    let mut composite = std::collections::BTreeMap::new();
    for r in runs {
        let key = (r["prompt"].as_u64().ok_or("run without prompt")?, r["rep"].as_u64().ok_or("run without rep")?);
        let s = r["observed_ttft_s"].as_f64().ok_or("run without observed_ttft_s")?;
        let (source, target) = (r["prefill_tokens"]["source"].as_u64(), r["prefill_tokens"]["target"].as_u64());
        match r["arm"].as_str() {
            Some("native") => {
                if source != Some(0) {
                    return Err(format!("{length}: a native run used the source ({source:?} prompt tokens) at {key:?}"));
                }
                native.insert(key, s);
            }
            Some("composite") => {
                if target != Some(0) {
                    return Err(format!("{length}: a composite run had the target read {target:?} prompt tokens at {key:?}"));
                }
                composite.insert(key, s);
            }
            other => return Err(format!("{length}: unknown arm {other:?}")),
        }
    }
    let pairs: Vec<prattle_core::measure::Pair> = native
        .iter()
        .filter_map(|(k, &n)| composite.get(k).map(|&c| prattle_core::measure::Pair { prompt: k.0 as usize, native_ms: n * 1e3, bridge_ms: c * 1e3 }))
        .collect();
    let s = prattle_core::measure::summarize(&pairs).ok_or(format!("{length}: no paired runs"))?;
    Ok(LengthRow {
        length,
        speedup: s.speedup,
        ci_low: s.ci_low,
        ci_high: s.ci_high,
        median_ratio: s.median_ratio,
        prompts: s.prompts,
        pairs: s.pairs,
        native_median_s: median(native.values().copied().collect()),
        composite_median_s: median(composite.values().copied().collect()),
    })
}

/// One fidelity row: a condition's mean loss per byte of reference, its excess over native, and
/// its NCR with a 95% paired-bootstrap interval over documents (10,000 resamples, seed 1).
#[derive(Debug, Clone, PartialEq)]
pub struct FidelityRow {
    pub length: u64,
    pub condition: String,
    pub nats_per_byte: f64,
    pub excess: f64,
    pub ncr: f64,
    pub ncr_low: f64,
    pub ncr_high: f64,
}

pub const FIDELITY_CONDITIONS: [&str; 5] = ["native", "composite", "source", "empty", "wrong_donor"];

pub fn fidelity(record: &Value) -> Result<Vec<FidelityRow>, String> {
    let mut rows = Vec::new();
    for entry in record["lengths"].as_array().ok_or("the record has no lengths")? {
        let length = entry["length"].as_u64().ok_or("a length entry without a length")?;
        let docs = entry["documents"].as_array().ok_or("a length entry without documents")?;
        // per document, per condition: nats per byte of reference
        let per_byte: Vec<Vec<f64>> = docs
            .iter()
            .map(|d| {
                let bytes = d["ref_bytes"].as_f64().filter(|b| *b > 0.0).ok_or("a document without ref_bytes")?;
                FIDELITY_CONDITIONS
                    .iter()
                    .map(|c| d["nll"][c].as_f64().map(|n| n / bytes).ok_or(format!("a document without nll.{c}")))
                    .collect::<Result<Vec<f64>, String>>()
            })
            .collect::<Result<_, String>>()?;
        if per_byte.is_empty() {
            return Err(format!("{length}: no documents"));
        }
        let mean = |k: usize| per_byte.iter().map(|d| d[k]).sum::<f64>() / per_byte.len() as f64;
        let (native, empty) = (mean(0), mean(3));
        for (k, c) in FIDELITY_CONDITIONS.iter().enumerate() {
            let triples: Vec<[f64; 3]> = per_byte.iter().map(|d| [d[3], d[k], d[0]]).collect();
            let (ncr_low, ncr_high) = prattle_core::measure::bootstrap_ncr(&triples, 10_000, 1);
            rows.push(FidelityRow {
                length,
                condition: c.to_string(),
                nats_per_byte: mean(k),
                excess: mean(k) - native,
                ncr: prattle_core::measure::ncr(empty, mean(k), native),
                ncr_low,
                ncr_high,
            });
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(arm: &str, prompt: u64, rep: u64, s: f64, source: u64, target: u64) -> Value {
        json!({"arm": arm, "prompt": prompt, "rep": rep, "observed_ttft_s": s,
               "prefill_tokens": {"source": source, "target": target}})
    }

    fn record(runs: Vec<Value>) -> Value {
        json!({"lengths": [{"length": 2048, "runs": runs}]})
    }

    #[test]
    fn fidelity_rows_are_per_byte_with_ncr_against_empty_and_native() {
        let doc = |p: u64| json!({"prompt": p, "ref_bytes": 100, "nll": {
            "native": 10.0, "composite": 30.0, "source": 50.0, "empty": 90.0, "wrong_donor": 120.0}});
        let rec = json!({"lengths": [{"length": 2048, "documents": [doc(0), doc(1), doc(2)]}]});
        let rows = fidelity(&rec).unwrap();
        let row = |c: &str| rows.iter().find(|r| r.condition == c).unwrap().clone();
        assert!((row("native").nats_per_byte - 0.1).abs() < 1e-12 && row("native").excess.abs() < 1e-12);
        assert!((row("native").ncr - 1.0).abs() < 1e-12 && row("empty").ncr.abs() < 1e-12);
        assert!((row("composite").ncr - 0.75).abs() < 1e-12, "{:?}", row("composite"));
        assert!((row("composite").excess - 0.2).abs() < 1e-12);
        assert!(row("wrong_donor").ncr < 0.0);
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn a_constant_halving_reads_two_x_over_every_pair() {
        let mut runs = vec![];
        for p in 0..2 {
            for r in 0..2 {
                runs.push(run("native", p, r, 2.0, 0, 2080));
                runs.push(run("composite", p, r, 1.0, 2080, 0));
            }
        }
        let rows = summarize(&record(runs)).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!((row.speedup - 2.0).abs() < 1e-12 && (row.median_ratio - 2.0).abs() < 1e-12);
        assert_eq!((row.length, row.prompts, row.pairs), (2048, 2, 4));
        assert!((row.native_median_s - 2.0).abs() < 1e-12 && (row.composite_median_s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_composite_run_where_the_target_read_the_prompt_is_refused() {
        let runs = vec![run("native", 0, 0, 2.0, 0, 2080), run("composite", 0, 0, 1.0, 2080, 7)];
        let e = summarize(&record(runs)).unwrap_err();
        assert!(e.contains("target"), "{e}");
    }

    #[test]
    fn a_native_run_that_used_the_source_is_refused() {
        let runs = vec![run("native", 0, 0, 2.0, 5, 2080), run("composite", 0, 0, 1.0, 2080, 0)];
        let e = summarize(&record(runs)).unwrap_err();
        assert!(e.contains("source"), "{e}");
    }
}
