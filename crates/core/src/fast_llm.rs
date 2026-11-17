//! FAST_LLM — one safetensors file holding a small model, a large model and the bridge between them.
//! The small model reads the prompt, the bridge translates its reading, the large model answers.
//!
//! The file is plain safetensors. Each model's tensors keep their official bytes, unchanged, under a
//! prefix (`source.`, `target.`); the bridge's under `bridge.`. The header's `__metadata__` carries
//! both `config.json`s, the shared `tokenizer.json` and a `fast_llm` manifest naming where each part
//! came from and the sha256 of every file it was packed from. A multimodal checkpoint contributes its
//! language model only.

use std::collections::BTreeMap;
use std::io::Write;

/// The manifest's `format`: what a loader checks before reading anything else.
pub const FORMAT: &str = "fast_llm/1";

/// A safetensors header's tensor entries, by name.
pub type Header = BTreeMap<String, serde_json::Value>;

/// A safetensors buffer's tensor entries (no `__metadata__`), its metadata, and its data section.
pub fn parse(bytes: &[u8]) -> Result<(Header, BTreeMap<String, String>, &[u8]), String> {
    let n = u64::from_le_bytes(bytes.get(..8).ok_or("not a safetensors buffer")?.try_into().unwrap()) as usize;
    let mut h: Header = serde_json::from_slice(bytes.get(8..8 + n).ok_or("a safetensors header past the end")?).map_err(|e| e.to_string())?;
    let metadata = match h.remove("__metadata__") {
        Some(m) => serde_json::from_value(m).map_err(|e| format!("__metadata__: {e}"))?,
        None => BTreeMap::new(),
    };
    Ok((h, metadata, &bytes[8 + n..]))
}

/// A tensor's bytes in `body`.
pub fn data<'a>(info: &serde_json::Value, body: &'a [u8], name: &str) -> Result<&'a [u8], String> {
    let off = info["data_offsets"].as_array().ok_or(format!("{name}: no data_offsets"))?;
    let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
    body.get(a..b).ok_or(format!("{name}: data past the end"))
}

/// Write a safetensors file from `(name, entry, bytes)` in order — offsets laid end to end, the header
/// padded to 8 — and return the bytes written.
pub fn write(tensors: &[(String, serde_json::Value, &[u8])], metadata: &BTreeMap<String, String>, out: &mut impl Write) -> Result<u64, String> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".into(), serde_json::to_value(metadata).map_err(|e| e.to_string())?);
    }
    let mut at = 0usize;
    for (name, info, bytes) in tensors {
        let mut info = info.clone();
        info["data_offsets"] = serde_json::json!([at, at + bytes.len()]);
        at += bytes.len();
        if header.insert(name.clone(), info).is_some() {
            return Err(format!("{name} appears twice"));
        }
    }
    let mut h = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    while (8 + h.len()) % 8 != 0 {
        h.push(b' ');
    }
    let io = |e: std::io::Error| e.to_string();
    out.write_all(&(h.len() as u64).to_le_bytes()).map_err(io)?;
    out.write_all(&h).map_err(io)?;
    for (_, _, bytes) in tensors {
        out.write_all(bytes).map_err(io)?;
    }
    Ok((8 + h.len() + at) as u64)
}

/// A FAST_LLM file split back into what the loaders take: each model's weights and the bridge, each a
/// safetensors buffer of its own with the prefix removed, and the header's metadata.
pub struct FastLlm {
    pub metadata: BTreeMap<String, String>,
    pub source: Vec<u8>,
    pub target: Vec<u8>,
    pub bridge: Vec<u8>,
    /// A THIN file's model parts, by side ("source", "target"): each a reference to its publisher's
    /// files in place of tensors, its weights empty until `resolve` fetches and checks them.
    pub refs: BTreeMap<String, Ref>,
    /// A pre-quantized file's target linear weights: the GGML type, and per weight (keyed as
    /// `half::Shared::quantize_linears` keys it) its shape and raw blocks. Empty for a bf16 target.
    pub quantized: Option<Quantized>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Quantized {
    pub ggml: String,
    pub weights: BTreeMap<String, (Vec<usize>, Vec<u8>)>,
}

/// Where a THIN FAST_LLM's model lives: a Hugging Face repo at a pinned revision, and every weight
/// file with its size and sha256 — what `resolve` checks before a byte of it is loaded.
#[derive(Debug, Clone, PartialEq)]
pub struct Ref {
    pub repo: String,
    pub revision: String,
    pub files: Vec<RefFile>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

/// One file of a reference, fetched from the Hugging Face Hub into the usual HF cache (a second fetch
/// downloads nothing) and its local path returned. The reader's own login is used — `HF_TOKEN`, or the
/// token the HF tools saved — read by the hub client itself: this code never opens or prints it.
#[cfg(feature = "hub")]
pub fn hub_fetch(r: &Ref, name: &str) -> Result<std::path::PathBuf, String> {
    let api = hf_hub::api::sync::ApiBuilder::from_env().with_progress(false).build().map_err(|e| format!("the hub client: {e}"))?;
    let repo = api.repo(hf_hub::Repo::with_revision(r.repo.clone(), hf_hub::RepoType::Model, r.revision.clone()));
    repo.get(name).map_err(|e| format!("{} {name} @ {}: {e} — a gated model needs your own Hugging Face login and its licence accepted", r.repo, r.revision))
}

/// The `ref` a manifest part carries, if it is one.
fn parse_ref(part: &serde_json::Value) -> Result<Option<Ref>, String> {
    let Some(r) = part.get("ref") else { return Ok(None) };
    let s = |v: &serde_json::Value, k: &str| v[k].as_str().map(str::to_string).ok_or(format!("a reference with no {k}"));
    let files = r["files"]
        .as_array()
        .ok_or("a reference with no files")?
        .iter()
        .map(|f| Ok(RefFile { name: s(f, "name")?, bytes: f["bytes"].as_u64().ok_or("a reference file with no size")?, sha256: s(f, "sha256")? }))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Some(Ref { repo: s(r, "repo")?, revision: s(r, "revision")?, files }))
}

impl FastLlm {
    pub fn unpack(bytes: &[u8]) -> Result<FastLlm, String> {
        let (header, metadata, body) = parse(bytes)?;
        let manifest: serde_json::Value = serde_json::from_str(metadata.get("fast_llm").ok_or("not a FAST_LLM: no fast_llm manifest")?).map_err(|e| format!("fast_llm manifest: {e}"))?;
        if manifest["format"] != FORMAT {
            return Err(format!("FAST_LLM format {} — this loader reads {FORMAT}", manifest["format"]));
        }
        let mut refs = BTreeMap::new();
        for side in ["source", "target"] {
            if let Some(r) = parse_ref(&manifest[side])? {
                refs.insert(side.to_string(), r);
            }
        }
        let quantized = match manifest.get("quantized") {
            None => None,
            Some(q) => {
                let shapes = q["shapes"].as_object().ok_or("a quantized FAST_LLM with no shapes")?;
                let mut weights = BTreeMap::new();
                for (name, dims) in shapes {
                    let key = format!("{QUANTIZED}{name}");
                    let info = header.get(&key).ok_or(format!("the quantized weight {name} is missing"))?;
                    let dims: Vec<usize> = dims.as_array().ok_or(format!("{name}: no shape"))?.iter().filter_map(|d| d.as_u64().map(|d| d as usize)).collect();
                    weights.insert(name.clone(), (dims, data(info, body, &key)?.to_vec()));
                }
                Some(Quantized { ggml: q["ggml"].as_str().unwrap_or("").to_string(), weights })
            }
        };
        let part = |prefix: &str| -> Result<Vec<u8>, String> {
            let tensors = header
                .iter()
                .filter(|(n, _)| !n.starts_with(QUANTIZED))
                .filter_map(|(n, info)| n.strip_prefix(prefix).map(|rest| (rest.to_string(), info)))
                .map(|(rest, info)| data(info, body, &rest).map(|d| (rest, info.clone(), d)))
                .collect::<Result<Vec<_>, String>>()?;
            if tensors.is_empty() {
                return Err(format!("FAST_LLM file has no {prefix}* tensors"));
            }
            let mut out = Vec::new();
            write(&tensors, &BTreeMap::new(), &mut out)?;
            Ok(out)
        };
        let side = |name: &str| if refs.contains_key(name) { Ok(Vec::new()) } else { part(&format!("{name}.")) };
        Ok(FastLlm { source: side("source")?, target: side("target")?, bridge: part("bridge.")?, metadata, refs, quantized })
    }

    /// Fetch every referenced part with `fetch` (the local path of one file of a `Ref`), check each
    /// file's size and sha256 against the manifest, and load it as an embedded part would be.
    pub fn resolve(mut self, fetch: &dyn Fn(&Ref, &str) -> Result<std::path::PathBuf, String>) -> Result<FastLlm, String> {
        for (side, r) in std::mem::take(&mut self.refs) {
            let mut paths = Vec::new();
            for f in &r.files {
                let path = fetch(&r, &f.name)?;
                let bytes = std::fs::metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?.len();
                if bytes != f.bytes {
                    return Err(format!("{} {} @ {}: {bytes} bytes, the manifest says {} (sha256 not checked)", r.repo, f.name, r.revision, f.bytes));
                }
                let sha = sha256_file(&path)?;
                if sha != f.sha256 {
                    return Err(format!("{} {} @ {}: sha256 {sha}, the manifest says {} — refused", r.repo, f.name, r.revision, f.sha256));
                }
                paths.push(path.to_string_lossy().into_owned());
            }
            let merged = crate::weights::read_shards(&paths)?;
            // the language model of a multimodal checkpoint only, as a FULL file holds it
            let (header, _, body) = parse(&merged)?;
            let keep = language_model(&header.keys().cloned().collect::<Vec<_>>());
            let tensors = header.iter().filter(|(n, _)| keep(n)).map(|(n, info)| data(info, body, n).map(|d| (n.clone(), info.clone(), d))).collect::<Result<Vec<_>, String>>()?;
            let mut weights = Vec::new();
            write(&tensors, &BTreeMap::new(), &mut weights)?;
            match side.as_str() {
                "source" => self.source = weights,
                _ => self.target = weights,
            }
        }
        Ok(self)
    }

    /// A metadata entry — a `config.json`, the `tokenizer.json`, the manifest.
    pub fn get(&self, key: &str) -> Result<&str, String> {
        self.metadata.get(key).map(String::as_str).ok_or(format!("FAST_LLM file has no {key}"))
    }
}

/// The sha256 of a file, in hex, read in blocks.
pub fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    use sha2::Digest;
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut hash, mut block) = (sha2::Sha256::new(), vec![0u8; 8 << 20]);
    loop {
        let n = f.read(&mut block).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Where a quantized file keeps one quantized weight: `target.q.<name as quantize_linears keys it>`.
pub const QUANTIZED: &str = "target.q.";

/// A GGML type's name in a manifest, and back.
pub fn ggml_name(q: candle_core::quantized::GgmlDType) -> String {
    format!("{q:?}").to_lowercase()
}
pub fn ggml_from_name(name: &str) -> Result<candle_core::quantized::GgmlDType, String> {
    use candle_core::quantized::GgmlDType as G;
    match name {
        "q4k" => Ok(G::Q4K),
        "q8_0" => Ok(G::Q8_0),
        other => Err(format!("a FAST_LLM quantized as {other:?} — this loader reads q4k and q8_0")),
    }
}

/// `PRATTLE_TARGET_FAST_LLM`: a FAST_LLM file the tools take their TARGET from in place of its folder.
/// A Q4K file's linear layers are read as stored, so a 70B arrives as ~43 GB and is never quantized
/// again on the host — the quantizing is done once, when the file is packed.
pub fn target_file() -> Option<std::path::PathBuf> {
    std::env::var_os("PRATTLE_TARGET_FAST_LLM").filter(|v| !v.is_empty()).map(Into::into)
}

/// The target every tool runs: from `target_file()` when it is set, else `dir`'s own weights — at
/// PRATTLE_TARGET_DTYPE either way (`crate::device`).
pub fn target_decoder(dir: &str, device: &candle_core::Device) -> Result<crate::decoder::Decoder, String> {
    target_decoder_from(dir, target_file().as_deref(), device, crate::device::target_dtype()?, crate::device::target_quant()?)
}

/// As `target_decoder`, the file and the precision given. A pre-quantized file's own GGML type wins
/// over `quant`. A file is refused unless its target IS the model `dir` holds (the two `config.json`s
/// agree): a fit or an evaluation against another model's weights would report numbers about the wrong
/// model, and nothing downstream could tell.
pub fn target_decoder_from(
    dir: &str,
    file: Option<&std::path::Path>,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    quant: Option<candle_core::quantized::GgmlDType>,
) -> Result<crate::decoder::Decoder, String> {
    let config = std::fs::read(format!("{dir}/config.json")).map_err(|e| format!("{dir}/config.json: {e}"))?;
    let Some(file) = file else {
        return crate::model::load_decoder_q(&config, crate::weights::read_dir(dir)?, device, dtype, quant);
    };
    let path = file.to_string_lossy();
    let f = FastLlm::unpack(&crate::weights::read_file(&path)?)?;
    if let Some(r) = f.refs.get("target") {
        return Err(format!("{path}: a THIN FAST_LLM, its target is {} @ {} — give a FULL or Q4K file", r.repo, r.revision));
    }
    let json = |b: &[u8]| serde_json::from_slice::<serde_json::Value>(b).map_err(|e| format!("{path}: {e}"));
    if json(f.get("target.config.json")?.as_bytes())? != json(&config)? {
        return Err(format!("{path}: its target is not the model in {dir} — the two config.json files differ"));
    }
    let FastLlm { target, quantized, .. } = f;
    match quantized {
        None => crate::model::load_decoder_q(&config, target, device, dtype, quant),
        Some(q) => {
            let (ggml, ready) = q.qtensors()?;
            crate::model::load_decoder_q_ready(&config, target, device, dtype, Some(ggml), Some(ready))
        }
    }
}

impl Quantized {
    /// The stored blocks as QTensors, exactly as they are — never quantized again.
    pub fn qtensors(self) -> Result<(candle_core::quantized::GgmlDType, std::collections::HashMap<String, candle_core::quantized::QTensor>), String> {
        let ggml = ggml_from_name(&self.ggml)?;
        let ready = self
            .weights
            .into_iter()
            .map(|(name, (dims, blocks))| {
                let storage = candle_core::quantized::QStorage::from_data(std::borrow::Cow::Owned(blocks), &candle_core::Device::Cpu, ggml).map_err(|e| e.to_string())?;
                Ok((name, candle_core::quantized::QTensor::new(storage, dims).map_err(|e| e.to_string())?))
            })
            .collect::<Result<_, String>>()?;
        Ok((ggml, ready))
    }
}

/// Keep a checkpoint's language-model tensors: all of them, or — when it is multimodal (its language
/// model's names carry `language_model.`) — those alone, the towers left out.
pub fn language_model(names: &[String]) -> impl Fn(&str) -> bool + use<> {
    let multimodal = names.iter().any(|n| n.contains("language_model."));
    move |n: &str| !multimodal || n.contains("language_model.")
}
