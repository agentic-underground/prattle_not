//! `just reproduce` helpers.
//!   reproduce prompts <dir> <source tokenizer.json> <count> <min tokens>   write the prompt set
//!   reproduce table <results.json>                                      E120's table, beside upstream's
//!   reproduce fidelity <results.json>                                   §1.7's table: NCR per condition

use prattle_core::measure::{SYNTHETIC_QUESTION, incident_line};

/// Upstream's published E120 rows (docs/BRIDGE.md §3.2): approximate length, ratio, 95% interval.
const UPSTREAM: [(u64, &str); 4] =
    [(128, "1.06x (0.87–1.29)"), (2048, "1.65x (1.61–1.69)"), (8192, "2.01x (1.96–2.05)"), (16384, "2.19x (2.15–2.23)")];

fn prompts(dir: &str, tokenizer: &str, count: u32, min_tokens: usize) -> Result<(), String> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    for v in 0..count {
        let mut text = String::from("SERVICE INCIDENT LOG (synthetic demonstration data)\n");
        let mut i = 0;
        while tok.encode(text.as_str(), false).map_err(|e| e.to_string())?.len() < min_tokens {
            for _ in 0..200 {
                text.push_str(&incident_line(v, i));
                i += 1;
            }
        }
        let path = format!("{dir}/prompt-{v}.txt");
        std::fs::write(&path, &text).map_err(|e| e.to_string())?;
        println!("{path}: {} lines", i);
    }
    std::fs::write(format!("{dir}/question.txt"), SYNTHETIC_QUESTION.trim()).map_err(|e| e.to_string())?;
    println!("question: {}", SYNTHETIC_QUESTION.trim());
    Ok(())
}

fn table(path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let record: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let rows = oracle::reproduce::summarize(&record)?;
    println!("| length | native median | composite median | speed-up (95% CI) | median ratio | prompts × pairs | upstream E120 |");
    println!("|---|---|---|---|---|---|---|");
    for r in rows {
        let up = UPSTREAM.iter().find(|(l, _)| *l == r.length).map(|(_, s)| *s).unwrap_or("—");
        println!(
            "| {} | {:.3} s | {:.3} s | **{:.2}x** ({:.2}–{:.2}) | {:.2}x | {} × {} | {} |",
            r.length, r.native_median_s, r.composite_median_s, r.speedup, r.ci_low, r.ci_high, r.median_ratio, r.prompts, r.pairs, up
        );
    }
    Ok(())
}

fn fidelity(path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let record: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    println!("| length | condition | nats / byte | excess over native | NCR (95% CI) |");
    println!("|---|---|---|---|---|");
    for r in oracle::reproduce::fidelity(&record)? {
        println!(
            "| {} | {} | {:.4} | {:+.4} | **{:.3}** ({:.3}–{:.3}) |",
            r.length, r.condition, r.nats_per_byte, r.excess, r.ncr, r.ncr_low, r.ncr_high
        );
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("prompts") if args.len() == 6 => {
            prompts(&args[2], &args[3], args[4].parse().unwrap_or(8), args[5].parse().unwrap_or(17000))
        }
        Some("table") if args.len() == 3 => table(&args[2]),
        Some("fidelity") if args.len() == 3 => fidelity(&args[2]),
        _ => Err("usage: reproduce prompts <dir> <tokenizer.json> <count> <min tokens> | reproduce table <results.json>".into()),
    };
    if let Err(e) = result {
        eprintln!("⛔ {e}");
        std::process::exit(1);
    }
}
