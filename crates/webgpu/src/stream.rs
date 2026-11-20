//! A safetensors file read AS IT ARRIVES: header, then each wanted tensor's bytes converted and handed
//! on in pieces, so no whole file — nor any whole tensor bound for the GPU — is ever held in memory.
//! A browser's wasm has 4 GiB in all and Qwen3.5-2B's file is 4.5 GB; this is how it loads anyway.
//!
//! What a tensor becomes is the `Sink`'s choice: `Half` (f16 on the GPU, streamed), `Full` (f32 on the
//! GPU, streamed) or `Small` (f32 values handed over whole, for the host to adjust before upload).
//! Pieces handed to `Sink::write` start at a 4-byte boundary and are a multiple of 4 bytes long — what
//! a GPU buffer write needs — and a `Half` tensor of odd length ends in one zero half.

/// What a sink keeps of one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Store {
    Half,
    Full,
    Small,
}

impl Store {
    /// Bytes per value once converted (`Small` is handed over as f32 values, not bytes).
    fn out_bytes(self) -> usize {
        match self {
            Store::Half => 2,
            Store::Full | Store::Small => 4,
        }
    }
}

pub trait Sink {
    /// What to keep of tensor `name`, of `shape`; `None` skips it.
    fn store(&self, name: &str, shape: &[usize]) -> Option<Store>;
    /// A kept tensor's bytes are about to arrive: `bytes` in all, as `store`, padded to 4.
    fn begin(&mut self, name: &str, store: Store, shape: &[usize], bytes: u64);
    /// Converted little-endian bytes of `name` at byte `offset`.
    fn write(&mut self, name: &str, offset: u64, bytes: &[u8]);
    /// A `Small` tensor's values, whole.
    fn small(&mut self, name: &str, shape: &[usize], values: Vec<f32>);
}

#[derive(Debug, Clone, Copy)]
enum Src {
    F32,
    F16,
    Bf16,
}

impl Src {
    fn bytes(self) -> usize {
        match self {
            Src::F32 => 4,
            Src::F16 | Src::Bf16 => 2,
        }
    }
    fn value(self, b: &[u8]) -> f32 {
        match self {
            Src::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Src::F16 => half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
            Src::Bf16 => half::bf16::from_le_bytes([b[0], b[1]]).to_f32(),
        }
    }
}

struct Entry {
    name: String,
    src: Src,
    shape: Vec<usize>,
    start: u64,
    end: u64,
    store: Store,
    /// Source bytes received and not yet converted.
    pending: Vec<u8>,
    /// Converted bytes written so far.
    out: u64,
    done: bool,
}

impl Entry {
    /// Source bytes per write group: values enough that the converted group is a multiple of 4 bytes.
    fn group(&self) -> usize {
        self.src.bytes() * (4 / self.store.out_bytes())
    }

    fn convert(&self, src: &[u8]) -> Vec<u8> {
        let n = self.src.bytes();
        let mut out = Vec::with_capacity(src.len() / n * self.store.out_bytes());
        for b in src.chunks_exact(n) {
            let v = self.src.value(b);
            match self.store {
                Store::Half => out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()),
                _ => out.extend_from_slice(&v.to_le_bytes()),
            }
        }
        out
    }

    /// Convert and hand on every whole group pending; at the end, the rest too, padded to 4 bytes.
    fn drain(&mut self, sink: &mut impl Sink, last: bool) {
        if self.store == Store::Small {
            if last {
                let n = self.src.bytes();
                let values = self.pending.chunks_exact(n).map(|b| self.src.value(b)).collect();
                self.pending = Vec::new();
                sink.small(&self.name, &self.shape, values);
            }
            return;
        }
        let whole = self.pending.len() / self.group() * self.group();
        let take = if last { self.pending.len() } else { whole };
        if take == 0 {
            return;
        }
        let mut bytes = self.convert(&self.pending[..take]);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        self.pending.drain(..take);
        sink.write(&self.name, self.out, &bytes);
        self.out += bytes.len() as u64;
    }
}

enum Phase {
    /// The 8-byte header length.
    Length,
    /// The JSON header, this many bytes long.
    Header(usize),
    /// The data section; `at` bytes of it consumed.
    Data,
}

/// One file's reading. Feed it the file's bytes in any pieces, in order; `finish` says whether every
/// kept tensor arrived whole.
pub struct Stream {
    phase: Phase,
    buf: Vec<u8>,
    entries: Vec<Entry>,
    at: u64,
    next: usize,
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}

impl Stream {
    pub fn new() -> Stream {
        Stream { phase: Phase::Length, buf: Vec::new(), entries: Vec::new(), at: 0, next: 0 }
    }

    pub fn feed(&mut self, mut chunk: &[u8], sink: &mut impl Sink) -> Result<(), String> {
        while !chunk.is_empty() {
            match self.phase {
                Phase::Length => {
                    let take = (8 - self.buf.len()).min(chunk.len());
                    self.buf.extend_from_slice(&chunk[..take]);
                    chunk = &chunk[take..];
                    if self.buf.len() == 8 {
                        let n = u64::from_le_bytes(self.buf[..8].try_into().unwrap());
                        if n > 100 << 20 {
                            return Err(format!("a safetensors header of {n} bytes: this is not a safetensors file"));
                        }
                        self.buf.clear();
                        self.phase = Phase::Header(n as usize);
                    }
                }
                Phase::Header(n) => {
                    let take = (n - self.buf.len()).min(chunk.len());
                    self.buf.extend_from_slice(&chunk[..take]);
                    chunk = &chunk[take..];
                    if self.buf.len() == n {
                        self.plan(sink)?;
                        self.buf = Vec::new();
                        self.phase = Phase::Data;
                    }
                }
                Phase::Data => {
                    self.data(chunk, sink);
                    chunk = &[];
                }
            }
        }
        Ok(())
    }

    /// Which tensors are kept, in file order.
    fn plan(&mut self, sink: &mut impl Sink) -> Result<(), String> {
        let header: serde_json::Value = serde_json::from_slice(&self.buf).map_err(|e| format!("safetensors header: {e}"))?;
        let header = header.as_object().ok_or("the safetensors header is not an object")?;
        for (name, t) in header {
            if name == "__metadata__" {
                continue;
            }
            let shape: Vec<usize> = t["shape"].as_array().ok_or(format!("{name} has no shape"))?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect();
            let Some(store) = sink.store(name, &shape) else { continue };
            let src = match t["dtype"].as_str() {
                Some("F32") => Src::F32,
                Some("F16") => Src::F16,
                Some("BF16") => Src::Bf16,
                other => return Err(format!("{name} is {other:?}: only F32, F16 and BF16 weights are read")),
            };
            let off = t["data_offsets"].as_array().ok_or(format!("{name} has no data_offsets"))?;
            let (start, end) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
            let values: usize = shape.iter().product();
            if (end - start) as usize != values * src.bytes() {
                return Err(format!("{name}: {} bytes for {values} values", end - start));
            }
            let entry = Entry { name: name.clone(), src, shape, start, end, store, pending: Vec::new(), out: 0, done: false };
            if store != Store::Small {
                let bytes = (values * store.out_bytes()).next_multiple_of(4) as u64;
                sink.begin(name, store, &entry.shape, bytes);
            }
            self.entries.push(entry);
        }
        self.entries.sort_by_key(|e| e.start);
        // a tensor of no bytes is whole before any data arrives
        for e in self.entries.iter_mut().filter(|e| e.start == e.end) {
            e.drain(sink, true);
            e.done = true;
        }
        Ok(())
    }

    fn data(&mut self, chunk: &[u8], sink: &mut impl Sink) {
        let (from, to) = (self.at, self.at + chunk.len() as u64);
        let mut i = self.next;
        while i < self.entries.len() && self.entries[i].start < to {
            let e = &mut self.entries[i];
            let (a, b) = (e.start.max(from), e.end.min(to));
            if a < b {
                e.pending.extend_from_slice(&chunk[(a - from) as usize..(b - from) as usize]);
                let last = b == e.end;
                e.drain(sink, last);
                e.done |= last;
            }
            if e.end <= to {
                self.next = i + 1;
            }
            i += 1;
        }
        self.at = to;
    }

    /// Every kept tensor arrived whole, or the first that did not.
    pub fn finish(&self) -> Result<(), String> {
        match self.entries.iter().find(|e| !e.done) {
            Some(e) => Err(format!("the file ended before {} arrived whole", e.name)),
            None if matches!(self.phase, Phase::Data) => Ok(()),
            None => Err("the file ended inside its safetensors header".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Keeps what a store rule says, and records every byte and value it is handed.
    #[derive(Default)]
    struct Record {
        begun: HashMap<String, (Store, u64)>,
        bytes: HashMap<String, Vec<u8>>,
        small: HashMap<String, Vec<f32>>,
    }

    impl Sink for Record {
        fn store(&self, name: &str, _shape: &[usize]) -> Option<Store> {
            match name {
                "half" | "half_from_f32" | "odd" => Some(Store::Half),
                "full" => Some(Store::Full),
                "small" => Some(Store::Small),
                _ => None,
            }
        }
        fn begin(&mut self, name: &str, store: Store, _shape: &[usize], bytes: u64) {
            self.begun.insert(name.into(), (store, bytes));
            self.bytes.insert(name.into(), vec![0xAA; bytes as usize]);
        }
        fn write(&mut self, name: &str, offset: u64, bytes: &[u8]) {
            assert_eq!(offset % 4, 0, "{name}: a write at {offset}");
            assert_eq!(bytes.len() % 4, 0, "{name}: a write of {} bytes", bytes.len());
            let buf = self.bytes.get_mut(name).expect("a write before its begin");
            buf[offset as usize..offset as usize + bytes.len()].copy_from_slice(bytes);
        }
        fn small(&mut self, name: &str, _shape: &[usize], values: Vec<f32>) {
            self.small.insert(name.into(), values);
        }
    }

    fn vals(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + 1.0) * 0.37 + seed).sin() * 3.0).collect()
    }

    /// A safetensors file: (name, dtype, shape, values), in this order in its data section.
    fn file(tensors: &[(&str, &str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, v) in tensors {
            let start = data.len();
            for &x in v {
                match *dtype {
                    "F32" => data.extend_from_slice(&x.to_le_bytes()),
                    "F16" => data.extend_from_slice(&half::f16::from_f32(x).to_le_bytes()),
                    _ => data.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes()),
                }
            }
            header.insert(name.to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let h = serde_json::to_vec(&header).unwrap();
        [(h.len() as u64).to_le_bytes().to_vec(), h, data].concat()
    }

    fn halves(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
    }

    fn floats(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn every_kept_tensor_arrives_converted_whatever_the_pieces_and_the_rest_are_skipped() {
        let (h, hf, s, f, skip, odd) = (vals(24, 0.1), vals(6, 0.2), vals(5, 0.3), vals(7, 0.4), vals(9, 0.5), vals(3, 0.6));
        let bytes = file(&[
            ("skipped", "BF16", vec![9], skip),
            ("half", "BF16", vec![4, 6], h.clone()),
            ("small", "F32", vec![5], s.clone()),
            ("full", "F16", vec![7], f.clone()),
            ("half_from_f32", "F32", vec![2, 3], hf.clone()),
            ("odd", "BF16", vec![3], odd.clone()),
        ]);
        for piece in [1, 3, 7, 64, bytes.len()] {
            let (mut stream, mut sink) = (Stream::new(), Record::default());
            for c in bytes.chunks(piece) {
                stream.feed(c, &mut sink).unwrap();
            }
            stream.finish().unwrap_or_else(|e| panic!("pieces of {piece}: {e}"));
            assert!(!sink.begun.contains_key("skipped"), "pieces of {piece}: a skipped tensor was begun");
            let bf = |v: &[f32]| v.iter().map(|&x| half::f16::from_f32(half::bf16::from_f32(x).to_f32()).to_f32()).collect::<Vec<_>>();
            let hh = |v: &[f32]| v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect::<Vec<_>>();
            assert_eq!(halves(&sink.bytes["half"]), bf(&h), "pieces of {piece}: half");
            assert_eq!(halves(&sink.bytes["half_from_f32"]), hh(&hf), "pieces of {piece}: half from f32");
            assert_eq!(floats(&sink.bytes["full"]), hh(&f), "pieces of {piece}: full");
            assert_eq!(sink.small["small"], s, "pieces of {piece}: small");
            assert_eq!(sink.begun["odd"], (Store::Half, 8), "pieces of {piece}: an odd half tensor is padded to 4 bytes");
            assert_eq!(halves(&sink.bytes["odd"]), [bf(&odd), vec![0.0]].concat(), "pieces of {piece}: odd, padded with a zero");
        }
    }

    #[test]
    fn a_file_cut_short_names_the_tensor_that_did_not_arrive() {
        let bytes = file(&[("half", "BF16", vec![4, 6], vals(24, 0.1)), ("small", "F32", vec![5], vals(5, 0.2))]);
        let (mut stream, mut sink) = (Stream::new(), Record::default());
        stream.feed(&bytes[..bytes.len() - 3], &mut sink).unwrap();
        let err = stream.finish().unwrap_err();
        assert!(err.contains("small"), "{err}");
    }
}
