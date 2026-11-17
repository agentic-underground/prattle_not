//! The paired time-to-first-token statistic and the run schedule — docs/BRIDGE.md §3.4.
//!
//! STATISTIC (upstream's E120): for each prompt, the mean over its repetitions of
//! `ln(native / bridge)`; the speed-up is `exp` of the mean over prompts, with a 95% Student
//! interval over prompts. The median of all paired ratios is reported beside it.

/// One paired repetition: the same prompt, timed in both arms.
#[derive(Debug, Clone, Copy)]
pub struct Pair {
    pub prompt: usize,
    pub native_ms: f64,
    pub bridge_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// native / bridge: above 1 means the bridge is faster.
    pub speedup: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub median_ratio: f64,
    pub prompts: usize,
    pub pairs: usize,
}

/// Two-sided 95% Student t quantile for `df` degrees of freedom.
pub fn t95(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912, 2.364624, 2.306004, 2.262157, 2.228139,
        2.200985, 2.178813, 2.160369, 2.144787, 2.131450, 2.119905, 2.109816, 2.100922, 2.093024, 2.085963,
        2.079614, 2.073873, 2.068658, 2.063899, 2.059539, 2.055529, 2.051831, 2.048407, 2.045230, 2.042272,
    ];
    match df {
        0 => f64::NAN,
        1..=30 => TABLE[df - 1],
        _ => {
            // Cornish-Fisher expansion about the normal quantile; well inside 1e-4 beyond df = 30.
            let (z, n) = (1.959_963_985_f64, df as f64);
            z + (z.powi(3) + z) / (4.0 * n) + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * n * n)
        }
    }
}

pub fn summarize(pairs: &[Pair]) -> Option<Summary> {
    if pairs.is_empty() {
        return None;
    }
    let mut prompts: Vec<usize> = pairs.iter().map(|p| p.prompt).collect();
    prompts.sort_unstable();
    prompts.dedup();
    let logs: Vec<f64> = prompts
        .iter()
        .map(|&id| {
            let own: Vec<f64> =
                pairs.iter().filter(|p| p.prompt == id).map(|p| (p.native_ms / p.bridge_ms).ln()).collect();
            own.iter().sum::<f64>() / own.len() as f64
        })
        .collect();
    let n = logs.len() as f64;
    let mean = logs.iter().sum::<f64>() / n;
    let (ci_low, ci_high) = if logs.len() < 2 {
        (f64::NAN, f64::NAN)
    } else {
        let var = logs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let half = t95(logs.len() - 1) * (var / n).sqrt();
        ((mean - half).exp(), (mean + half).exp())
    };
    let mut ratios: Vec<f64> = pairs.iter().map(|p| p.native_ms / p.bridge_ms).collect();
    ratios.sort_by(f64::total_cmp);
    let mid = ratios.len() / 2;
    let median_ratio = if ratios.len() % 2 == 1 { ratios[mid] } else { (ratios[mid - 1] + ratios[mid]) / 2.0 };
    Some(Summary { speedup: mean.exp(), ci_low, ci_high, median_ratio, prompts: logs.len(), pairs: pairs.len() })
}

/// Which of `arms` runs first in repetition `rep` of prompt `prompt` — the rest follow in order,
/// wrapping. Rotating, so no arm always inherits another's warm caches: measured on the host, a
/// FIXED order made two identical-size models look 20% apart.
pub fn first_arm(prompt: usize, rep: usize, arms: usize) -> usize {
    (prompt + rep) % arms
}

// ---- fidelity — docs/BRIDGE.md §1.7 ----------------------------------------------------------

/// Native-closure ratio (paper Eq. 8): how much of the gap between an empty target and the native
/// target a condition closes. 1 = native, 0 = no better than empty, negative = worse than empty.
pub fn ncr(empty: f64, condition: f64, native: f64) -> f64 {
    (empty - condition) / (empty - native)
}

/// 95% percentile interval for the NCR of a condition, by paired bootstrap over documents: each
/// resample draws documents with replacement and recomputes NCR from the resampled means. `rows`
/// are per-document `[empty, condition, native]` losses. Deterministic for a given `seed`.
pub fn bootstrap_ncr(rows: &[[f64; 3]], resamples: usize, seed: u64) -> (f64, f64) {
    if rows.is_empty() || resamples == 0 {
        return (f64::NAN, f64::NAN);
    }
    // xorshift64*: small, deterministic, and enough for resampling indices.
    let mut state = seed.max(1);
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let n = rows.len();
    let mut values: Vec<f64> = (0..resamples)
        .map(|_| {
            let mut sum = [0.0; 3];
            for _ in 0..n {
                let r = &rows[(next() % n as u64) as usize];
                for k in 0..3 {
                    sum[k] += r[k];
                }
            }
            ncr(sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64)
        })
        .collect();
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((q * (resamples - 1) as f64).round() as usize).min(resamples - 1)];
    (at(0.025), at(0.975))
}

/// What every synthetic prompt asks after its log.
pub const SYNTHETIC_QUESTION: &str = "\nWhich service failed first, and why? Answer in one sentence.";

/// Line `i` of synthetic incident log `v`: deterministic, varied, and log-shaped.
pub fn incident_line(v: u32, i: u32) -> String {
    const SERVICES: [&str; 7] = ["auth", "billing", "search", "queue", "cache", "gateway", "storage"];
    const EVENTS: [&str; 6] = ["latency spike", "retry storm", "connection reset", "disk pressure", "timeout", "recovered"];
    let x = (v.wrapping_mul(2_654_435_761) ^ i.wrapping_mul(40_503)).wrapping_mul(2_246_822_519);
    let (h, m, sec) = (9 + i / 3600 % 12, i / 60 % 60, i % 60);
    format!(
        "{h:02}:{m:02}:{sec:02} node-{} {} {} {}ms\n",
        x % 13,
        SERVICES[(x >> 4) as usize % SERVICES.len()],
        EVENTS[(x >> 8) as usize % EVENTS.len()],
        20 + (x >> 12) % 980
    )
}

/// FNV-1a, 64-bit, hex: names a prompt in a published record without publishing its text twice.
pub fn prompt_hash(text: &str) -> String {
    let h = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * a.abs().max(1.0)
    }

    #[test]
    fn t_quantiles_match_the_table() {
        assert!((t95(7) - 2.364624).abs() < 1e-5);
        assert!((t95(1) - 12.706205).abs() < 1e-5);
        assert!((t95(30) - 2.042272).abs() < 1e-5);
        assert!((t95(1000) - 1.962339).abs() < 1e-4, "beyond the table, the expansion holds");
        assert!((t95(60) - 2.000298).abs() < 1e-4);
    }

    #[test]
    fn a_constant_ratio_is_reported_exactly_with_a_zero_width_interval() {
        let pairs: Vec<Pair> =
            (0..8).flat_map(|p| (0..3).map(move |_| Pair { prompt: p, native_ms: 200.0, bridge_ms: 100.0 })).collect();
        let s = summarize(&pairs).unwrap();
        assert!(close(s.speedup, 2.0) && close(s.ci_low, 2.0) && close(s.ci_high, 2.0) && close(s.median_ratio, 2.0));
        assert_eq!((s.prompts, s.pairs), (8, 24));
    }

    #[test]
    fn the_estimate_is_geometric_over_per_prompt_means_of_log_ratios() {
        // prompt 0: ratios 2 and 8 -> mean log = ln 4; prompt 1: ratio 1 -> ln 1. exp(mean) = 2.
        let pairs = [
            Pair { prompt: 0, native_ms: 2.0, bridge_ms: 1.0 },
            Pair { prompt: 0, native_ms: 8.0, bridge_ms: 1.0 },
            Pair { prompt: 1, native_ms: 5.0, bridge_ms: 5.0 },
        ];
        let s = summarize(&pairs).unwrap();
        assert!(close(s.speedup, 2.0), "{}", s.speedup);
        // per-prompt logs: ln4, 0 -> sd = ln4/sqrt2, se = ln4/2, half-width = t95(1) * ln4 / 2
        let half = t95(1) * 4f64.ln() / 2.0;
        assert!(close(s.ci_low, (4f64.ln() / 2.0 - half).exp()) && close(s.ci_high, (4f64.ln() / 2.0 + half).exp()));
        assert!(close(s.median_ratio, 2.0));
    }

    #[test]
    fn one_prompt_has_no_interval_and_none_has_no_summary() {
        assert!(summarize(&[]).is_none());
        let s = summarize(&[Pair { prompt: 0, native_ms: 3.0, bridge_ms: 1.0 }]).unwrap();
        assert!(s.ci_low.is_nan() && s.ci_high.is_nan());
    }

    #[test]
    fn synthetic_logs_are_deterministic_and_differ_by_variant() {
        assert_eq!(incident_line(3, 7), incident_line(3, 7));
        let a: String = (0..8).map(|i| incident_line(0, i)).collect();
        let b: String = (0..8).map(|i| incident_line(1, i)).collect();
        assert_ne!(a, b);
        assert_ne!(prompt_hash(&a), prompt_hash(&b));
    }

    #[test]
    fn ncr_is_the_share_of_the_empty_to_native_gap_closed() {
        assert!(close(ncr(1.0, 0.0, 0.0), 1.0));
        assert!(close(ncr(1.0, 1.0, 0.0), 0.0));
        assert!(close(ncr(1.0, 0.25, 0.0), 0.75));
        assert!(ncr(1.0, 1.5, 0.0) < 0.0, "worse than empty is negative");
    }

    #[test]
    fn identical_documents_bootstrap_to_a_point() {
        let rows = vec![[1.0, 0.25, 0.0]; 8];
        let (lo, hi) = bootstrap_ncr(&rows, 1000, 7);
        assert!(close(lo, 0.75) && close(hi, 0.75), "{lo} {hi}");
    }

    #[test]
    fn the_bootstrap_brackets_the_estimate_and_repeats_for_a_seed() {
        let rows: Vec<[f64; 3]> = (0..8).map(|i| [1.0 + 0.1 * i as f64, 0.2 + 0.05 * (i % 3) as f64, 0.1]).collect();
        let point = {
            let m = |k: usize| rows.iter().map(|r| r[k]).sum::<f64>() / rows.len() as f64;
            ncr(m(0), m(1), m(2))
        };
        let (lo, hi) = bootstrap_ncr(&rows, 10_000, 42);
        assert!(lo < point && point < hi, "{lo} {point} {hi}");
        assert_eq!(bootstrap_ncr(&rows, 10_000, 42), (lo, hi));
    }

    #[test]
    fn every_arm_leads_equally_often() {
        for arms in [2, 3] {
            let mut leads = vec![0; arms];
            for p in 0..8 {
                for r in 0..3 {
                    leads[first_arm(p, r, arms)] += 1;
                }
            }
            let (lo, hi) = (*leads.iter().min().unwrap(), *leads.iter().max().unwrap());
            assert!(hi - lo <= 1, "{arms} arms led {leads:?} times");
        }
        assert_ne!(first_arm(0, 0, 2), first_arm(0, 1, 2));
    }
}
