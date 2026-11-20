//! The fitted KV bridge between two attention-only models of DIFFERENT shapes — docs/BRIDGE.md §1.4,
//! the paper's closed-form ridge maps, relaxed to unmatched geometry: each target layer's keys
//! (de-rotated) and values are fitted from the source's residual stream entering the proportionally
//! placed source layer. Mapping from the source's own K and V instead was measured and failed on
//! held-out questions (NCR −0.31); the residual stream carries what the 3→5 head change needs.

/// Target layer `l` of `target_layers` reads source layer round(l·(S−1)/(T−1)).
pub fn layer_of(l: usize, source_layers: usize, target_layers: usize) -> usize {
    if target_layers < 2 {
        return 0;
    }
    ((l * (source_layers - 1)) as f64 / (target_layers - 1) as f64).round() as usize
}

// ---- the translator ---------------------------------------------------------------------------

use crate::llama::Kv;
use candle_core::{Device, Result, Tensor};

/// Per target layer, one map for K and one for V, from `[n, features + 1]` — the source's residual
/// stream entering the mapped layer, plus a constant 1 for the intercept — to `[n, tgt_heads·dim]`
/// (target keys DE-ROTATED, re-rotated after mapping; target values as they are).
pub struct KvMap {
    pub layer_of: Vec<usize>,
    /// How many consecutive source residual streams feed each target layer, starting at `layer_of`.
    pub taps: usize,
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub src_heads: usize,
    pub tgt_heads: usize,
    pub dim: usize,
    /// The paper's correction (§1.5), folded: per target layer, `I + B Uᵀ` (`[dim, dim]`) applied to
    /// every head's translated keys (`ck`) and values (`cv`). Empty = no correction.
    pub ck: Vec<Tensor>,
    pub cv: Vec<Tensor>,
    /// How many rows a short-conv layer's state holds (LFM2's `L − 1`); 0 when no layer is one. A
    /// conv layer's map is `k` alone (`v` empty): it turns each token's features into the target's
    /// B·x row, and the last `conv_rows` rows are the state.
    pub conv_rows: usize,
    /// When true, a `k`-alone layer's rows are the target layer's normed INPUTS, from which the target
    /// builds the state itself (`LayerKeys::state_from_inputs`) — Granite's Mamba-2 — rather than the
    /// state's own rows.
    pub rows_are_inputs: bool,
    /// Per target layer, (kv heads, head width) where the layers differ (Gemma 4's global layers);
    /// empty when every layer is `tgt_heads × dim`.
    pub shapes: Vec<(usize, usize)>,
    /// What a tool adds to the translation. The runtime adds nothing.
    pub extension: Option<std::sync::Arc<dyn Extension>>,
}

/// A tool's addition to a map's translation. The runtime holds none; a tool that attaches one also
/// reads and writes its tensors (`set_reader`). Every method but the bookkeeping defaults to nothing.
pub trait Extension: Send + Sync {
    /// Computed once per translation, before any layer, from the prompt's token ids `[1, n]` (when the
    /// caller has them) and the target: whatever `features` reads.
    fn prepare(&self, _ids: Option<&Tensor>, _target: &dyn crate::decoder::LayerKeys) -> Result<Option<Tensor>> {
        Ok(None)
    }
    /// The features a mapped layer reads, from the map's own `f` `[n, cols]` (intercept last).
    fn features(&self, f: Tensor, _prepared: Option<&Tensor>) -> Result<Tensor> {
        Ok(f)
    }
    /// What target layer `l` adds to its (keys, values) rows, from the features `f`; `None` adds nothing.
    fn rows(&self, _l: usize, _f: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
        Ok(None)
    }
    /// The same extension with its tensors on `device`.
    fn to_device(&self, device: &Device) -> Result<std::sync::Arc<dyn Extension>>;
    /// Its tensors, by the names a bridge file stores them under.
    fn tensors(&self) -> Result<std::collections::HashMap<String, Tensor>>;
    /// Itself, for the tool that made it to recognise it.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// How `read` turns a bridge file into a map. A tool may install its own once, before any read.
static READER: std::sync::OnceLock<fn(&[u8]) -> Result<KvMap>> = std::sync::OnceLock::new();

/// Read bridge files with `f` from now on, in place of `KvMap::from_bytes`. The first call wins.
pub fn set_reader(f: fn(&[u8]) -> Result<KvMap>) {
    let _ = READER.set(f);
}

/// A bridge file's map, by the installed reader or else `KvMap::from_bytes`.
pub fn read(bytes: &[u8]) -> Result<KvMap> {
    READER.get().map_or_else(|| KvMap::from_bytes(bytes), |f| f(bytes))
}

impl KvMap {
    /// The placeholder for a target layer the map does not write — a recurrent layer, whose state
    /// crosses from the source untranslated. Never stored.
    pub fn no_map() -> Tensor {
        Tensor::zeros(0, candle_core::DType::F32, &Device::Cpu).expect("an empty tensor")
    }

    /// Which of `streams` source residual streams a translation reads: every tap of every mapped
    /// layer, clamped to the last stream as `tapped_features` clamps it. Only these need to cross to
    /// the target's device.
    pub fn streams_read(&self, streams: usize) -> Vec<bool> {
        let mut read = vec![false; streams];
        for (l, &s) in self.layer_of.iter().enumerate() {
            if self.k[l].elem_count() > 0 {
                for t in 0..self.taps {
                    read[(s + t).min(streams - 1)] = true;
                }
            }
        }
        read
    }

    /// Whether target layer `l`'s map writes attention K/V (both maps present).
    pub fn writes_kv(&self, l: usize) -> bool {
        self.k[l].elem_count() > 0 && self.v[l].elem_count() > 0
    }

    /// Target layer `l`'s (kv heads, head width).
    pub fn shape(&self, l: usize) -> (usize, usize) {
        self.shapes.get(l).copied().unwrap_or((self.tgt_heads, self.dim))
    }

    /// Whether target layer `l`'s map writes a short conv's rows (`k` alone).
    pub fn writes_conv(&self, l: usize) -> bool {
        self.k[l].elem_count() > 0 && self.v[l].elem_count() == 0
    }

    /// The same map with every tensor on `device` — where the model it feeds runs.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let all = |ts: &[Tensor]| ts.iter().map(|t| t.to_device(device)).collect::<Result<Vec<_>>>();
        Ok(Self { layer_of: self.layer_of.clone(), taps: self.taps, k: all(&self.k)?, v: all(&self.v)?, src_heads: self.src_heads, tgt_heads: self.tgt_heads, dim: self.dim, ck: all(&self.ck)?, cv: all(&self.cv)?, conv_rows: self.conv_rows, rows_are_inputs: self.rows_are_inputs, shapes: self.shapes.clone(), extension: self.extension.as_ref().map(|r| r.to_device(device)).transpose()? })
    }

    /// Load from safetensors bytes — the page fetches the file, the host reads it. A tensor this map
    /// does not read is refused by name rather than ignored: a bridge translated without part of
    /// itself would answer wrongly and say nothing.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let t = candle_core::safetensors::load_buffer(bytes, &Device::Cpu)?;
        let mut unread: Vec<&String> = t.keys().filter(|n| !Self::reads(n)).collect();
        if !unread.is_empty() {
            unread.sort();
            candle_core::bail!("this bridge carries tensors this runtime does not read: {}", unread.iter().take(4).map(|n| n.as_str()).collect::<Vec<_>>().join(", "));
        }
        Self::from_tensors(&t)
    }

    /// Whether `name` is one of the tensors a map reads.
    pub fn reads(name: &str) -> bool {
        matches!(name, "meta" | "layer_of" | "shapes") || name.split_once('.').is_some_and(|(p, l)| matches!(p, "k" | "v" | "ck" | "cv") && l.parse::<usize>().is_ok())
    }

    /// The map in `t`, every tensor it does not read left alone.
    pub fn from_tensors(t: &std::collections::HashMap<String, Tensor>) -> Result<Self> {
        let get = |n: &str| t.get(n).cloned().ok_or_else(|| candle_core::Error::Msg(format!("bridge file has no {n}")));
        let meta = get("meta")?.to_vec1::<u32>()?;
        if meta.get(6).is_some_and(|&f| f != 0) {
            candle_core::bail!("this bridge sets meta[6], a flag this runtime does not read");
        }
        let layer_of: Vec<usize> = get("layer_of")?.to_vec1::<u32>()?.into_iter().map(|l| l as usize).collect();
        // a layer absent from the file has no map — a recurrent layer
        let f32 = |n: String| match t.get(&n) {
            Some(x) => x.to_dtype(candle_core::DType::F32),
            None => Ok(Self::no_map()),
        };
        // the maps stay at the file's half precision — `translate_hidden` widens one layer at a time, the
        // same numbers at half the memory: Gemma 4 31B's are 2.8 GB, beside a 4-bit 31B on its card
        let stored = |n: String| match t.get(&n) {
            Some(x) => Ok(x.clone()),
            None => Ok(Self::no_map()),
        };
        let k: Vec<Tensor> = (0..layer_of.len()).map(|l| stored(format!("k.{l}"))).collect::<Result<_>>()?;
        let v = (0..layer_of.len()).map(|l| stored(format!("v.{l}"))).collect::<Result<_>>()?;
        if k.iter().all(|x| x.elem_count() == 0) {
            return Err(candle_core::Error::Msg("bridge file maps no layer".into()));
        }
        let taps = meta.get(3).copied().unwrap_or(1) as usize;
        let corr = |p: &str| -> Result<Vec<Tensor>> {
            if !(0..layer_of.len()).any(|l| t.contains_key(&format!("{p}.{l}"))) {
                return Ok(Vec::new());
            }
            (0..layer_of.len()).map(|l| f32(format!("{p}.{l}"))).collect()
        };
        let (ck, cv) = (corr("ck")?, corr("cv")?);
        let shapes = match t.get("shapes") {
            Some(s) => s.to_vec1::<u32>()?.chunks_exact(2).map(|p| (p[0] as usize, p[1] as usize)).collect(),
            None => Vec::new(),
        };
        Ok(Self { layer_of, taps, k, v, src_heads: meta[0] as usize, tgt_heads: meta[1] as usize, dim: meta[2] as usize, ck, cv, conv_rows: meta.get(4).copied().unwrap_or(0) as usize, rows_are_inputs: meta.get(5).copied().unwrap_or(0) == 1, shapes, extension: None })
    }
}

/// `[1, n, hidden]` as `[n, hidden + 1]` with the intercept column.
pub fn features_with_intercept(h: &Tensor) -> Result<Tensor> {
    let (_, n, d) = h.dims3()?;
    let rows = h.reshape((n, d))?;
    Tensor::cat(&[&rows, &Tensor::ones((n, 1), h.dtype(), h.device())?], 1)
}

/// The features for one target layer: `taps` consecutive source residual streams from `first`
/// (clamped to the last), concatenated, plus the intercept — `[n, taps·hidden + 1]`.
pub fn tapped_features(hidden: &[Tensor], first: usize, taps: usize) -> Result<Tensor> {
    let last = hidden.len() - 1;
    let parts: Vec<Tensor> = (0..taps).map(|t| hidden[(first + t).min(last)].clone()).collect();
    features_with_intercept(&Tensor::cat(&parts, 2)?)
}

/// Translate the source's residual streams (one per source layer, from `forward_capture`) into a
/// target state, position for position. A layer with no map comes back `None`.
pub fn translate_hidden<D: crate::decoder::LayerKeys>(map: &KvMap, hidden: &[Tensor], target: &D) -> Result<Kv> {
    translate_hidden_with(map, hidden, None, target)
}

/// As `translate_hidden`, with the prompt's token ids `[1, n]` on the target's device — what a map's
/// extension may read.
pub fn translate_hidden_with<D: crate::decoder::LayerKeys>(map: &KvMap, hidden: &[Tensor], ids: Option<&Tensor>, target: &D) -> Result<Kv> {
    translate_hidden_at(map, hidden, ids, target, 0)
}

/// As `translate_hidden_with`, for tokens whose first sits at absolute position `pos` — a turn appended
/// to a conversation the target already holds. Each token's rows depend only on its own features and
/// position, so a tail translated at its position is the tail of the whole prompt translated at 0.
pub fn translate_hidden_at<D: crate::decoder::LayerKeys>(map: &KvMap, hidden: &[Tensor], ids: Option<&Tensor>, target: &D, pos: usize) -> Result<Kv> {
    let prepared = match &map.extension {
        Some(e) => e.prepare(ids, target)?,
        None => None,
    };
    let to_heads = |y: Tensor, n: usize, l: usize| -> Result<Tensor> {
        y.reshape((1, n, map.shape(l).0, map.shape(l).1))?.transpose(1, 2)?.contiguous()
    };
    let layers = map
        .layer_of
        .iter()
        .enumerate()
        .map(|(l, &s)| {
            if map.k[l].elem_count() == 0 {
                return Ok(None); // recurrent: the caller fills it from the source's own state
            }
            if pos > 0 && map.writes_conv(l) {
                return Err(candle_core::Error::Msg(format!("layer {l}: a short-conv or scan state is built from the whole prompt, and cannot be translated from position {pos} on")));
            }
            let n = hidden[s].dim(1)?;
            let f = tapped_features(hidden, s, map.taps)?;
            let f = match &map.extension {
                Some(e) => e.features(f, prepared.as_ref())?,
                None => f,
            };
            let wk = map.k[l].to_dtype(f.dtype())?; // widened for this layer only
            let wv = || map.v[l].to_dtype(f.dtype());
            if map.writes_conv(l) && map.rows_are_inputs {
                // every token's input row: the target runs its own layer over them to build the state
                let inputs = f.matmul(&wk)?.unsqueeze(0)?; // [1, n, hidden]
                return target.state_from_inputs(l, &inputs).map(Some);
            }
            if map.writes_conv(l) {
                // the last `conv_rows` tokens' B·x rows, zeros in front of a shorter prompt
                let r = map.conv_rows;
                let take = n.min(r);
                let rows = f.narrow(0, n - take, take)?.matmul(&wk)?; // [take, width]
                let rows = if take < r { Tensor::cat(&[&rows.zeros_like()?.narrow(0, 0, 1)?.repeat((r - take, 1))?, &rows], 0)? } else { rows };
                let c = rows.t()?.contiguous()?.unsqueeze(0)?; // [1, width, r]
                return Ok(Some((c, Tensor::zeros(0, candle_core::DType::F32, f.device())?)));
            }
            let (mut yk, mut yv) = (f.matmul(&wk)?, f.matmul(&wv()?)?);
            if let Some((rk, rv)) = map.extension.as_ref().map(|r| r.rows(l, &f)).transpose()?.flatten() {
                (yk, yv) = ((yk + rk)?, (yv + rv)?);
            }
            let k = target.rotate_layer_keys(l, &to_heads(yk, n, l)?, pos, false)?;
            let v = to_heads(yv, n, l)?;
            match (map.ck.get(l), map.cv.get(l)) {
                // contiguous: at one token RoPE hands back strided keys, which CUDA's matmul refuses (a
                // short prompt read with many bridge tokens leaves the reader one token)
                (Some(ck), Some(cv)) if ck.elem_count() > 0 => Ok(Some((k.contiguous()?.broadcast_matmul(ck)?, v.contiguous()?.broadcast_matmul(cv)?))),
                _ => Ok(Some((k, v))),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Kv::from_layers(layers))
}

/// Fill every layer `translate_hidden` left `None` — a recurrent layer — with the source's own state at
/// that layer, which a matched pair installs as it is.
pub fn fill_from_source(translated: Kv, source: &Kv) -> Result<Kv> {
    let layers = translated
        .layers()
        .iter()
        .enumerate()
        .map(|(l, slot)| match slot {
            Some(kv) => Ok(Some(kv.clone())),
            None => match source.layers().get(l) {
                Some(Some(kv)) => Ok(Some(kv.clone())),
                _ => Err(candle_core::Error::Msg(format!("layer {l} has no map and the source holds no state there"))),
            },
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Kv::from_layers(layers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// Rotates nothing — enough of a target to translate into.
    struct Unrotated;
    impl crate::decoder::LayerKeys for Unrotated {
        fn rotate_layer_keys(&self, _layer: usize, k: &Tensor, _pos: usize, _inverse: bool) -> Result<Tensor> {
            Ok(k.clone())
        }
    }

    /// A short-conv layer's map turns each token's features into the target's B·x row; the state is
    /// the last `conv_rows` of them `[1, hidden, conv_rows]`, zero rows in front when the prompt is
    /// shorter.
    /// A layer whose state sums over every token: the map predicts the target layer's input rows, and
    /// the target builds the state from all of them.
    #[test]
    fn an_input_rows_layer_hands_every_row_to_the_target_to_build_its_state() {
        struct Summing;
        impl crate::decoder::LayerKeys for Summing {
            fn rotate_layer_keys(&self, _layer: usize, k: &Tensor, _pos: usize, _inverse: bool) -> Result<Tensor> {
                Ok(k.clone())
            }
            fn state_from_inputs(&self, _layer: usize, inputs: &Tensor) -> Result<(Tensor, Tensor)> {
                Ok((inputs.sum(1)?, inputs.dims3().map(|(_, n, _)| Tensor::new(n as f32, inputs.device()))??))
            }
        }
        let cpu = Device::Cpu;
        // features [x, 1] → the input row [2x, x + 1]
        let k = Tensor::new(&[[2f32, 1.], [0., 1.]], &cpu).unwrap();
        let map = KvMap { layer_of: vec![0], taps: 1, k: vec![k], v: vec![KvMap::no_map()], src_heads: 1, tgt_heads: 1, dim: 2, ck: vec![], cv: vec![], conv_rows: 0, rows_are_inputs: true, shapes: vec![], extension: None };
        let h = Tensor::new(&[[[1f32], [2.], [3.]]], &cpu).unwrap(); // three tokens
        let kv = translate_hidden(&map, &[h], &Summing).unwrap();
        let (sum, n) = kv.layers()[0].as_ref().unwrap();
        assert_eq!(sum.to_vec2::<f32>().unwrap(), vec![vec![12., 9.]], "every token's input row reaches the target");
        assert_eq!(n.to_scalar::<f32>().unwrap(), 3.0);
    }

    /// A tail translated at its own position is the tail of the whole prompt translated from 0 — what
    /// appending a turn to a conversation relies on.
    #[test]
    fn a_tail_translated_at_its_position_is_the_tail_of_the_whole() {
        /// Adds each key row's absolute position, so a key at the wrong position shows.
        struct Positioned;
        impl crate::decoder::LayerKeys for Positioned {
            fn rotate_layer_keys(&self, _layer: usize, k: &Tensor, pos: usize, _inverse: bool) -> Result<Tensor> {
                let n = k.dim(2)?;
                let at = Tensor::arange(pos as f32, (pos + n) as f32, k.device())?.reshape((1, 1, n, 1))?;
                k.broadcast_add(&at)
            }
        }
        let cpu = Device::Cpu;
        let w = Tensor::new(&[[1f32, 0.], [0.5, 2.]], &cpu).unwrap(); // features [x, 1] → [x + 0.5, 2]
        let map = KvMap { layer_of: vec![0], taps: 1, k: vec![w.clone()], v: vec![w], src_heads: 1, tgt_heads: 1, dim: 2, ck: vec![], cv: vec![], conv_rows: 0, rows_are_inputs: false, shapes: vec![(1, 2)], extension: None };
        let h = Tensor::new(&[[[1f32], [2.], [3.], [4.], [5.]]], &cpu).unwrap();
        let whole = translate_hidden(&map, std::slice::from_ref(&h), &Positioned).unwrap();
        let tail = translate_hidden_at(&map, &[h.narrow(1, 3, 2).unwrap()], None, &Positioned, 3).unwrap();
        let (wk, wv) = whole.layers()[0].as_ref().unwrap();
        let (tk, tv) = tail.layers()[0].as_ref().unwrap();
        let rows = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(rows(tk), rows(&wk.narrow(2, 3, 2).unwrap()), "a key translated at its position differs from the whole's");
        assert_eq!(rows(tv), rows(&wv.narrow(2, 3, 2).unwrap()));
    }

    #[test]
    fn a_conv_layer_translates_to_its_last_rows() {
        let cpu = Device::Cpu;
        // features [x, 1] → the row [2x, x + 1]
        let k = Tensor::new(&[[2f32, 1.], [0., 1.]], &cpu).unwrap();
        let map = KvMap { layer_of: vec![0], taps: 1, k: vec![k], v: vec![KvMap::no_map()], src_heads: 1, tgt_heads: 1, dim: 2, ck: vec![], cv: vec![], conv_rows: 2, rows_are_inputs: false, shapes: vec![], extension: None };
        assert!(map.writes_conv(0) && !map.writes_kv(0));
        let h = Tensor::new(&[[[1f32], [2.], [3.]]], &cpu).unwrap(); // three tokens
        let kv = translate_hidden(&map, &[h], &Unrotated).unwrap();
        let (c, _) = kv.layers()[0].as_ref().unwrap();
        assert_eq!(c.to_vec3::<f32>().unwrap(), vec![vec![vec![4., 6.], vec![3., 4.]]], "the last two tokens' rows, one column each");
        let one = Tensor::new(&[[[5f32]]], &cpu).unwrap();
        let kv = translate_hidden(&map, &[one], &Unrotated).unwrap();
        let (c, _) = kv.layers()[0].as_ref().unwrap();
        assert_eq!(c.to_vec3::<f32>().unwrap(), vec![vec![vec![0., 10.], vec![0., 6.]]], "a one-token prompt leaves a zero row in front");
    }

    #[test]
    fn a_map_reads_only_the_streams_its_mapped_layers_tap() {
        let w = Tensor::ones((3, 2), DType::F32, &Device::Cpu).unwrap();
        let none = KvMap::no_map();
        let map = KvMap {
            layer_of: vec![0, 5, 4],
            taps: 2,
            k: vec![w.clone(), none.clone(), w.clone()],
            v: vec![w.clone(), none, w],
            src_heads: 1,
            tgt_heads: 1,
            dim: 2,
            ck: vec![],
            cv: vec![],
            conv_rows: 0,
            rows_are_inputs: false,
            shapes: vec![],
            extension: None,
        };
        // layer 0 taps 0 and 1; layer 1 is unmapped; layer 2 taps 4 and 5, the last stream
        assert_eq!(map.streams_read(6), vec![true, true, false, false, true, true]);
    }

    #[test]
    fn the_source_fills_the_layers_the_map_left_empty() {
        let cpu = Device::Cpu;
        let t = |x: f32| Tensor::full(x, (1, 2), &cpu).unwrap();
        let translated = Kv::from_layers(vec![Some((t(1.0), t(1.0))), None]);
        let source = Kv::from_layers(vec![Some((t(7.0), t(7.0))), Some((t(9.0), t(9.0)))]);
        let kv = fill_from_source(translated, &source).unwrap();
        let first = |l: usize| kv.layers()[l].as_ref().unwrap().0.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert_eq!((first(0), first(1)), (1.0, 9.0), "translated layers stay; empty ones take the source's");
        assert!(fill_from_source(Kv::from_layers(vec![None]), &Kv::from_layers(vec![None])).is_err(), "nothing to fill from is an error");
    }

    /// The correction trains through a frozen target: a backward must not allocate gradients for the
    /// target's weights, or a 1.7B's correction outgrows a 24 GB card (vendor/candle-core/PATCHED.md).
    #[test]
    fn a_frozen_weight_gets_no_gradient() {
        let w = Tensor::ones((4, 3), DType::F32, &Device::Cpu).unwrap();
        let x = candle_core::Var::ones((2, 4), DType::F32, &Device::Cpu).unwrap();
        let grads = x.as_tensor().matmul(&w.t().unwrap().t().unwrap()).unwrap().sum_all().unwrap().backward().unwrap();
        assert!(grads.get(x.as_tensor()).is_some(), "the variable lost its gradient");
        assert!(grads.get_ids().count() == 1, "a frozen weight was given a gradient: {} entries", grads.get_ids().count());
    }

    /// A tensor the map does not read is refused by name — translating without part of a bridge
    /// would answer wrongly and say nothing — and a file of only what it reads loads.
    #[test]
    fn a_bridge_file_carrying_a_tensor_the_runtime_does_not_read_is_refused() {
        let cpu = Device::Cpu;
        let w = Tensor::ones((2, 1), DType::F16, &cpu).unwrap();
        let mut t = std::collections::HashMap::from([
            ("meta".to_string(), Tensor::new(&[1u32, 1, 1, 1, 0, 0, 0][..], &cpu).unwrap()),
            ("layer_of".to_string(), Tensor::new(&[0u32][..], &cpu).unwrap()),
            ("k.0".to_string(), w.clone()),
            ("v.0".to_string(), w.clone()),
        ]);
        let bytes = |t: &std::collections::HashMap<String, Tensor>| {
            let file = std::env::temp_dir().join(format!("unread-map-{}.safetensors", std::process::id()));
            candle_core::safetensors::save(t, &file).unwrap();
            let b = std::fs::read(&file).unwrap();
            std::fs::remove_file(&file).unwrap();
            b
        };
        assert!(KvMap::from_bytes(&bytes(&t)).is_ok(), "a plain map did not load");
        t.insert("extra.0".into(), w);
        let err = KvMap::from_bytes(&bytes(&t)).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("extra.0"), "an unread tensor was not refused by name: {err:?}");
        t.remove("extra.0");
        t.insert("meta".into(), Tensor::new(&[1u32, 1, 1, 1, 0, 0, 1][..], &cpu).unwrap());
        let err = KvMap::from_bytes(&bytes(&t)).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("meta[6]"), "a flag the runtime does not read was not refused: {err:?}");
    }

    #[test]
    fn layers_map_proportionally_end_to_end() {
        assert_eq!(layer_of(0, 30, 32), 0);
        assert_eq!(layer_of(31, 30, 32), 29);
        assert_eq!(layer_of(16, 30, 32), 15);
        assert!((0..32).map(|l| layer_of(l, 30, 32)).collect::<Vec<_>>().windows(2).all(|w| w[0] <= w[1]));
    }

}
