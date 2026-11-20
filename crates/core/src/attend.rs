//! Scaled dot-product attention for every model file, behind `PRATTLE_ATTN=full|chunked`.
//!
//! FULL scores every query row against every key at once: one `[b, heads, n, keys]` f32 matrix.
//! CHUNKED scores `BLOCK` query rows at a time. Softmax is per row, so each block's rows are the full
//! path's rows, and memory grows with the prompt rather than its square — a 32B at ~4096 tokens holds
//! 768 MiB of scores per layer instead of 12 GiB.
//!
//! The full path is taken whenever gradients flow (the correction's backward) or a listener records the
//! probabilities (`crate::observe`): both need the whole matrix.

use candle_core::{Result, Tensor};

/// Query rows scored at once on the chunked path.
pub const BLOCK: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Full,
    Chunked,
}

/// The mode from `PRATTLE_ATTN`.
pub fn mode() -> Result<Mode> {
    mode_from(std::env::var("PRATTLE_ATTN").ok().as_deref())
}

/// The mode a `PRATTLE_ATTN` value names: `full` is FULL, `chunked` is CHUNKED; unset is CHUNKED, so
/// inference fits long prompts unless it asks otherwise. Gradients and a listener take FULL regardless.
pub fn mode_from(value: Option<&str>) -> Result<Mode> {
    match value {
        Some("full") => Ok(Mode::Full),
        None | Some("chunked") => Ok(Mode::Chunked),
        Some(other) => candle_core::bail!("PRATTLE_ATTN={other}: one of full, chunked"),
    }
}

/// The query rows each scoring pass covers, as `(first row, rows)`, in order and covering `0..n`.
pub fn blocks(n: usize, mode: Mode) -> Vec<(usize, usize)> {
    match mode {
        Mode::Full => vec![(0, n)],
        Mode::Chunked => (0..n).step_by(BLOCK).map(|r0| (r0, BLOCK.min(n - r0))).collect(),
    }
}

/// Attention of `q` `[b, heads, n, d]` (rows at absolute positions `pos..pos+n`) over `k`, `v`
/// `[b, kv_heads, keys, d]`, returned as `[b, n, heads·d]`. Scores are multiplied by `scale`; a
/// row at position p sees key j when j ≤ p and, with a `window`, p − j < window.
pub fn attend(q: &Tensor, k: Tensor, v: Tensor, pos: usize, scale: f64, window: Option<usize>, diff: bool) -> Result<Tensor> {
    let mode = if diff || crate::observe::listening() { Mode::Full } else { mode()? };
    attend_with(mode, q, k, v, pos, scale, window, diff)
}

#[allow(clippy::too_many_arguments)]
pub fn attend_with(mode: Mode, q: &Tensor, k: Tensor, v: Tensor, pos: usize, scale: f64, window: Option<usize>, diff: bool) -> Result<Tensor> {
    let (b, heads, n, d) = q.dims4()?;
    let group = heads / k.dim(1)?;
    let k = candle_transformers::utils::repeat_kv(k, group)?;
    let v = candle_transformers::utils::repeat_kv(v, group)?.contiguous()?;
    let kt = k.t()?.contiguous()?;
    let q = q.contiguous()?;
    let mut rows = Vec::new();
    for (r0, m) in blocks(n, mode) {
        let qb = if m == n { q.clone() } else { q.narrow(2, r0, m)? };
        let att = qb.matmul(&kt)?;
        let att = if scale == 1.0 { att } else { (att * scale)? };
        let att = mask(att, pos + r0, window)?;
        let p = if diff { candle_nn::ops::softmax(&att, candle_core::D::Minus1)? } else { candle_nn::ops::softmax_last_dim(&att)? };
        drop(att);
        if mode == Mode::Full {
            crate::observe::attention(&p)?;
        }
        rows.push(p.matmul(&v)?);
    }
    let y = if rows.len() == 1 { rows.pop().expect("one block") } else { Tensor::cat(&rows, 2)? };
    y.transpose(1, 2)?.reshape((b, n, heads * d))
}

/// Hide the keys rows `first..` may not see: row i (at absolute position first+i) sees key j when
/// j ≤ first+i and, with a window, first+i − j < window. Returned unchanged when nothing is hidden.
fn mask(att: Tensor, first: usize, window: Option<usize>) -> Result<Tensor> {
    let (rows, keys) = (att.dim(2)?, att.dim(3)?);
    let hidden: Vec<u8> = (0..rows)
        .flat_map(|i| {
            let p = first + i;
            (0..keys).map(move |j| u8::from(j > p || window.is_some_and(|w| p - j.min(p) >= w)))
        })
        .collect();
    if !hidden.contains(&1) {
        return Ok(att);
    }
    let m = Tensor::from_slice(&hidden, (rows, keys), att.device())?.broadcast_as(att.shape())?;
    let neg = Tensor::new(f32::NEG_INFINITY, att.device())?.to_dtype(att.dtype())?.broadcast_as(att.shape())?;
    m.where_cond(&neg, &att)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Inference that names no mode is chunked — full attention at ~2048 tokens takes a whole 24 GiB card
    /// under a Q4K 32B — and either mode can still be asked for by name.
    #[test]
    fn unset_is_chunked_and_both_modes_can_be_named() {
        assert_eq!(mode_from(None).unwrap(), Mode::Chunked);
        assert_eq!(mode_from(Some("full")).unwrap(), Mode::Full);
        assert_eq!(mode_from(Some("chunked")).unwrap(), Mode::Chunked);
        assert!(mode_from(Some("flash")).is_err());
    }

    #[test]
    fn a_long_prompt_is_scored_in_blocks_of_at_most_256_rows() {
        let n = 2048 + 37;
        let b = blocks(n, Mode::Chunked);
        assert!(b.iter().all(|&(_, m)| m <= BLOCK && m > 0), "a block holds more than {BLOCK} rows: {b:?}");
        let mut next = 0;
        for &(r0, m) in &b {
            assert_eq!(r0, next, "blocks leave a gap or overlap: {b:?}");
            next += m;
        }
        assert_eq!(next, n, "blocks cover {next} of {n} rows");
        assert_eq!(blocks(n, Mode::Full), vec![(0, n)], "the full path scores every row at once");
    }

    fn wave(shape: (usize, usize, usize, usize), seed: f32) -> Tensor {
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        let v: Vec<f32> = (0..len).map(|i| ((i as f32 + 1.0) * 0.73 + seed).sin()).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    /// The chunked path gives the full path's rows: fresh and after a cache, causal and windowed,
    /// grouped heads, a prompt that is not a whole number of blocks.
    #[test]
    fn chunked_gives_the_full_paths_rows() {
        let (heads, kv_heads, d, n) = (4, 2, 8, 600);
        for (pos, window) in [(0, None), (50, None), (0, Some(100)), (50, Some(300))] {
            let q = wave((1, heads, n, d), 0.1);
            let k = wave((1, kv_heads, pos + n, d), 0.2);
            let v = wave((1, kv_heads, pos + n, d), 0.3);
            let full = attend_with(Mode::Full, &q, k.clone(), v.clone(), pos, 0.35, window, false).unwrap();
            let chunked = attend_with(Mode::Chunked, &q, k, v, pos, 0.35, window, false).unwrap();
            assert_eq!(full.dims(), &[1, n, heads * d]);
            let worst = (&full - &chunked).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            assert!(worst < 1e-5, "pos {pos} window {window:?}: chunked differs from full by {worst:e}");
        }
    }
}
