//! A model's weights as the ONE safetensors buffer the decoder loads — read whole, or merged from the
//! shards a larger model ships in, byte for byte (tensor bytes unchanged, only the header rewritten).

use std::collections::BTreeMap;

/// What a loader takes: a safetensors buffer of its own, or a model read in place from a larger one
/// (`crate::half::Shared::part`). A `Vec<u8>` converts, so a caller holding one passes it as it is.
pub enum Weights {
    Bytes(Vec<u8>),
    Shared(crate::half::Shared),
}

impl Weights {
    /// The weights as a `Shared`; a buffer of its own is moved in, not copied.
    pub fn shared(self) -> Result<crate::half::Shared, String> {
        match self {
            Weights::Bytes(b) => crate::half::Shared::new(b).map_err(|e| e.to_string()),
            Weights::Shared(s) => Ok(s),
        }
    }

    /// A VarBuilder over the weights, at `dtype` on `device`.
    pub fn builder(self, dtype: candle_core::DType, device: &candle_core::Device) -> Result<candle_nn::VarBuilder<'static>, String> {
        Ok(self.shared()?.builder(dtype, device))
    }

    /// No tensors at all — a THIN FAST_LLM's part before `resolve`.
    pub fn is_empty(&self) -> bool {
        matches!(self, Weights::Bytes(b) if b.is_empty())
    }

    /// Whether these are read in place from the same buffer as `other`.
    pub fn shares_buffer(&self, other: &Weights) -> bool {
        matches!((self, other), (Weights::Shared(a), Weights::Shared(b)) if a.shares_buffer(b))
    }
}

impl From<Vec<u8>> for Weights {
    fn from(b: Vec<u8>) -> Self {
        Weights::Bytes(b)
    }
}

impl From<crate::half::Shared> for Weights {
    fn from(s: crate::half::Shared) -> Self {
        Weights::Shared(s)
    }
}

/// Merge safetensors buffers into one. A tensor named in two buffers refuses.
pub fn merge(buffers: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut data: Vec<u8> = Vec::new();
    for bytes in buffers {
        let n = u64::from_le_bytes(bytes.get(..8).ok_or("not a safetensors buffer")?.try_into().unwrap()) as usize;
        let h: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| e.to_string())?;
        let body = &bytes[8 + n..];
        for (name, mut info) in h {
            if name == "__metadata__" {
                continue;
            }
            let off = info["data_offsets"].as_array().ok_or(format!("{name}: no data_offsets"))?;
            let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
            let start = data.len();
            data.extend_from_slice(&body[a..b]);
            info["data_offsets"] = serde_json::json!([start, data.len()]);
            if header.insert(name.clone(), info).is_some() {
                return Err(format!("{name} is in two shards"));
            }
        }
    }
    let mut h = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    while (8 + h.len()) % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    Ok(out)
}

/// A model folder's weights: `model.safetensors`; else the shards `model.safetensors.index.json`
/// names (as transformers reads them — some publishers name even one shard
/// `model.safetensors-00001-of-00001.safetensors`); else every `model-*.safetensors`, merged.
pub fn read_dir(dir: &str) -> Result<Vec<u8>, String> {
    let whole = format!("{dir}/model.safetensors");
    if std::path::Path::new(&whole).exists() {
        return read_file(&whole);
    }
    let index = format!("{dir}/model.safetensors.index.json");
    if let Ok(text) = std::fs::read_to_string(&index) {
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{index}: {e}"))?;
        let mut files: Vec<String> =
            v["weight_map"].as_object().ok_or(format!("{index}: no weight_map"))?.values().filter_map(|f| f.as_str().map(str::to_string)).collect();
        files.sort();
        files.dedup();
        return read_shards(&files.iter().map(|f| format!("{dir}/{f}")).collect::<Vec<_>>());
    }
    let mut shards: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{dir}: {e}"))?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(format!("{dir}: no model.safetensors and no shards"));
    }
    read_shards(&shards.iter().map(|s| format!("{dir}/{s}")).collect::<Vec<_>>())
}

/// Shard files merged into one buffer, streamed: each shard's header first, then ONE allocation of
/// the merged size, each shard's tensor bytes read straight into place — so a model of tens of GB
/// needs its own size in memory once, not once per copy.
pub fn read_shards(paths: &[String]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    // Pass 1: every shard's header — each shard's body keeps its own layout, at a base offset.
    let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut bodies: Vec<(u64, usize)> = Vec::with_capacity(paths.len()); // (offset in the file, length)
    let mut base = 0usize;
    for path in paths {
        let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|e| format!("{path}: {e}"))?;
        let n = u64::from_le_bytes(len) as usize;
        let mut h = vec![0u8; n];
        f.read_exact(&mut h).map_err(|e| format!("{path}: {e}"))?;
        let h: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&h).map_err(|e| format!("{path}: {e}"))?;
        let body = f.metadata().map_err(|e| format!("{path}: {e}"))?.len() as usize - 8 - n;
        for (name, mut info) in h {
            if name == "__metadata__" {
                continue;
            }
            let off = info["data_offsets"].as_array().ok_or(format!("{name}: no data_offsets"))?;
            let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
            info["data_offsets"] = serde_json::json!([base + a, base + b]);
            if header.insert(name.clone(), info).is_some() {
                return Err(format!("{name} is in two shards"));
            }
        }
        bodies.push(((8 + n) as u64, body));
        base += body;
    }
    let mut h = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    while (8 + h.len()) % 8 != 0 {
        h.push(b' ');
    }
    // Pass 2: one allocation, each shard's body read straight into its place.
    let start = 8 + h.len();
    let mut out = vec![0u8; start + base];
    out[..8].copy_from_slice(&(h.len() as u64).to_le_bytes());
    out[8..start].copy_from_slice(&h);
    let mut at = start;
    for (path, (offset, len)) in paths.iter().zip(bodies) {
        read_into(path, offset, &mut out[at..at + len])?;
        at += len;
    }
    Ok(out)
}

/// A whole file into memory, held once: `read_into` from its start.
pub fn read_file(path: &str) -> Result<Vec<u8>, String> {
    let len = std::fs::metadata(path).map_err(|e| format!("{path}: {e}"))?.len() as usize;
    let mut out = vec![0u8; len];
    read_into(path, 0, &mut out)?;
    Ok(out)
}

/// `buf.len()` bytes of `path` from `offset` into `buf`, in chunks, each dropped from the page cache
/// once it is in `buf`. A plain read leaves every byte in the cache as well, and a container's memory
/// limit counts both copies: a 70B's 141 GB of weights came to ~282 GB against a 256 GB limit.
pub fn read_into(path: &str, offset: u64, buf: &mut [u8]) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};
    const CHUNK: usize = 256 << 20;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    f.seek(SeekFrom::Start(offset)).map_err(|e| format!("{path}: {e}"))?;
    let mut done = 0usize;
    while done < buf.len() {
        let n = CHUNK.min(buf.len() - done);
        f.read_exact(&mut buf[done..done + n]).map_err(|e| format!("{path}: {e}"))?;
        drop_cached(&f, offset + done as u64, n);
        done += n;
    }
    Ok(())
}

/// Ask the kernel to let go of a range of `f` it has cached. Advice only: where it is not supported
/// (another platform, some network mounts) nothing changes and the read is still correct.
#[cfg(unix)]
fn drop_cached(f: &std::fs::File, offset: u64, len: usize) {
    use std::os::unix::io::AsRawFd;
    unsafe { libc::posix_fadvise(f.as_raw_fd(), offset as libc::off_t, len as libc::off_t, libc::POSIX_FADV_DONTNEED) };
}

#[cfg(not(unix))]
fn drop_cached(_: &std::fs::File, _: u64, _: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn shard(dir: &std::path::Path, name: &str, tensors: &[(&str, Vec<f32>)]) -> String {
        let map: std::collections::HashMap<String, Tensor> =
            tensors.iter().map(|(n, v)| (n.to_string(), Tensor::new(v.as_slice(), &Device::Cpu).unwrap())).collect();
        let path = dir.join(name);
        candle_core::safetensors::save(&map, &path).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn streamed_shards_hold_the_tensors_the_in_memory_merge_does() {
        let dir = std::env::temp_dir().join(format!("shards-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = shard(&dir, "a.safetensors", &[("x", vec![1.0, 2.0, 3.0]), ("y", vec![4.0])]);
        let b = shard(&dir, "b.safetensors", &[("z", vec![5.0, 6.0])]);
        let merged = merge(&[std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap()]).unwrap();
        let streamed = read_shards(&[a, b]).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let load = |bytes: &[u8]| candle_core::safetensors::load_buffer(bytes, &Device::Cpu).unwrap();
        let (m, s) = (load(&merged), load(&streamed));
        assert_eq!(m.len(), s.len());
        for (name, t) in &m {
            let u = &s[name];
            assert_eq!(t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), u.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), "{name}");
        }
    }
}
