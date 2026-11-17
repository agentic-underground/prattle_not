//! The numeric alignment between two tokenisations of one rendered prompt — upstream's
//! `codec.numeric_alignment`, which gives the translator one monotone, fractional source position per
//! target token. The two decoded byte streams are matched (common prefix and suffix, then Python
//! difflib's `SequenceMatcher` on what differs, ported exactly below), each target token's byte
//! midpoint is carried through the match, and placed between the source tokens' byte midpoints.

/// Each token's bytes: a special token as its text, any other through the byte-level BPE alphabet.
pub fn pieces(tok: &tokenizers::Tokenizer, ids: &[u32]) -> Vec<Vec<u8>> {
    let special: std::collections::HashSet<u32> =
        tok.get_added_tokens_decoder().iter().filter(|(_, t)| t.special).map(|(id, _)| *id).collect();
    let decode = byte_decoder();
    ids.iter()
        .map(|id| {
            let t = tok.id_to_token(*id).unwrap_or_else(|| panic!("token {id} is not in the vocabulary"));
            if special.contains(id) { t.into_bytes() } else { t.chars().map(|c| decode[&c]).collect() }
        })
        .collect()
}

/// GPT-2's `bytes_to_unicode`, inverted: the printable character standing for each byte.
fn byte_decoder() -> std::collections::HashMap<char, u8> {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars = printable.clone();
    let mut extra = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    printable.iter().zip(&chars).map(|(b, c)| (char::from_u32(*c).unwrap(), *b as u8)).collect()
}

/// One center per target token, in source-token coordinates: 0 ≤ center ≤ n − 1, non-decreasing.
pub fn centers(source: &[Vec<u8>], target: &[Vec<u8>]) -> Vec<f32> {
    let (a, b): (Vec<u8>, Vec<u8>) = (source.concat(), target.concat());
    let n = source.len();
    let spans = |p: &[Vec<u8>]| -> Vec<(usize, usize)> {
        let mut cursor = 0;
        p.iter().map(|x| { let s = (cursor, cursor + x.len()); cursor += x.len(); s }).collect()
    };
    let midpoints: Vec<f64> = spans(source).iter().map(|(x, y)| (*x + *y) as f64 / 2.0).collect();
    let ops: Vec<(usize, usize, usize, usize)> = byte_operations(&a, &b).into_iter().filter(|op| op.3 > op.2).collect();
    assert!(!ops.is_empty(), "no target byte spans were mapped");
    let ends: Vec<f64> = ops.iter().map(|op| op.3 as f64).collect();
    let bisect_right = |xs: &[f64], v: f64| xs.partition_point(|x| *x <= v);
    let mut out: Vec<f32> = Vec::with_capacity(target.len());
    let mut last = 0f64;
    for (start, end) in spans(target) {
        let point = (start + end) as f64 / 2.0;
        let (i, j, k, l) = ops[bisect_right(&ends, point).min(ops.len() - 1)];
        let mapped = i as f64 + (point - k as f64) * (j - i) as f64 / (l - k) as f64;
        let right = bisect_right(&midpoints, mapped);
        let center = if right == 0 {
            0.0
        } else if right == n {
            (n - 1) as f64
        } else {
            right as f64 - 1.0 + (mapped - midpoints[right - 1]) / (midpoints[right] - midpoints[right - 1])
        };
        last = last.max(center.min((n - 1) as f64));
        out.push(last as f32);
    }
    out
}

/// Upstream's `_byte_operations`: the common prefix, `SequenceMatcher` opcodes on the middle, the
/// common suffix — each as (a_start, a_end, b_start, b_end).
fn byte_operations(a: &[u8], b: &[u8]) -> Vec<(usize, usize, usize, usize)> {
    let mut prefix = 0;
    while prefix < a.len().min(b.len()) && a[prefix] == b[prefix] {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < a.len().min(b.len()) - prefix && a[a.len() - suffix - 1] == b[b.len() - suffix - 1] {
        suffix += 1;
    }
    let (ae, be) = (a.len() - suffix, b.len() - suffix);
    assert!((ae - prefix).max(be - prefix) <= 16384, "the changed template byte region exceeds the supported alignment bound");
    let mut ops = if prefix > 0 { vec![(0, prefix, 0, prefix)] } else { vec![] };
    for (i, j, k, l) in opcodes(&a[prefix..ae], &b[prefix..be]) {
        ops.push((i + prefix, j + prefix, k + prefix, l + prefix));
    }
    if suffix > 0 {
        ops.push((ae, a.len(), be, b.len()));
    }
    ops
}

/// `difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()`, spans only.
fn opcodes(a: &[u8], b: &[u8]) -> Vec<(usize, usize, usize, usize)> {
    let mut b2j: std::collections::HashMap<u8, Vec<usize>> = std::collections::HashMap::new();
    for (j, c) in b.iter().enumerate() {
        b2j.entry(*c).or_default().push(j);
    }
    let longest = |alo: usize, ahi: usize, blo: usize, bhi: usize| -> (usize, usize, usize) {
        let (mut besti, mut bestj, mut best) = (alo, blo, 0);
        let mut j2len: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for i in alo..ahi {
            let mut next = std::collections::HashMap::new();
            for &j in b2j.get(&a[i]).map(|v| v.as_slice()).unwrap_or(&[]) {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = if j > 0 { j2len.get(&(j - 1)).copied().unwrap_or(0) } else { 0 } + 1;
                next.insert(j, k);
                if k > best {
                    (besti, bestj, best) = (i + 1 - k, j + 1 - k, k);
                }
            }
            j2len = next;
        }
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            (besti, bestj, best) = (besti - 1, bestj - 1, best + 1);
        }
        while besti + best < ahi && bestj + best < bhi && a[besti + best] == b[bestj + best] {
            best += 1;
        }
        (besti, bestj, best)
    };
    let mut queue = vec![(0, a.len(), 0, b.len())];
    let mut blocks = vec![];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest(alo, ahi, blo, bhi);
        if k > 0 {
            blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    blocks.sort();
    let mut merged: Vec<(usize, usize, usize)> = vec![];
    for (i, j, k) in blocks {
        match merged.last_mut() {
            Some((i1, j1, k1)) if *i1 + *k1 == i && *j1 + *k1 == j => *k1 += k,
            _ => merged.push((i, j, k)),
        }
    }
    merged.push((a.len(), b.len(), 0));
    let (mut i, mut j) = (0, 0);
    let mut ops = vec![];
    for (ai, bj, size) in merged {
        if i < ai || j < bj {
            ops.push((i, ai, j, bj)); // replace, delete or insert
        }
        if size > 0 {
            ops.push((ai, ai + size, bj, bj + size)); // equal
        }
        (i, j) = (ai + size, bj + size);
    }
    ops
}
