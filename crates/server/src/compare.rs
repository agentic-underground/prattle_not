//! /compare — a FAST_LLM (`prattle_core::fast_llm`) beside the three files it was packed from, run in
//! Prattle's own Rust here on the server. One question, three arms: the large model reads the prompt
//! and answers; the small model reads it and the large one answers through the bridge, from the three
//! files; and the same from the one FAST_LLM file. One question at a time, the arms in rotated order,
//! so no arm is always first.
//!
//! DEVELOPMENT HAS THE GPUS FIRST. On CUDA the comparison loads only while development is idle — its
//! sentinel `/tmp/prattle-gpu-idle` exists, nobody holds `/tmp/prattle-gpu.lock`, and no other process
//! holds memory on a card — and unloads within seconds when that stops being true. Asking while it is
//! not true asks the person first ("are you sure?"); a yes loads the models for them until development
//! is idle again.
//!
//! `PRATTLE_COMPARE` names the FAST_LLM. The three files are found from its manifest under
//! `PRATTLE_MODELS`: `<source name>/`, `<target name>/` and `bridges/<bridge name>`. Devices, dtypes
//! and quantization follow `PRATTLE_DEVICES`, `PRATTLE_SOURCE_DTYPE`, `PRATTLE_TARGET_DTYPE`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use prattle_core::{Arm, Model, Step};
use serde_json::{Value, json};

/// The most tokens one arm writes: enough for a short answer and a decode rate.
const MAX_TOKENS: usize = 48;

/// The two models, and what loading each cost.
struct Loaded {
    three: Model,
    fast: Model,
    load: Value,
}

pub struct Compare {
    status: Mutex<Value>,
    loaded: Mutex<Option<Loaded>>,
    asked: Mutex<usize>,
    file: PathBuf,
    models: PathBuf,
    /// A person said yes to running while development has the GPUs; cleared when development is idle.
    overridden: Mutex<bool>,
}

/// Development is idle while this exists (`just gpu-idle`); `just gpu-busy` removes it.
const IDLE_SENTINEL: &str = "/tmp/prattle-gpu-idle";
/// What every GPU job holds while it runs (`flock`).
const GPU_LOCK: &str = "/tmp/prattle-gpu.lock";

/// Whether this server's models are on GPUs at all; on the CPU there is nothing to compete for.
fn uses_gpu() -> bool {
    std::env::var("PRATTLE_DEVICES").is_ok_and(|d| d.contains("cuda"))
}

/// Whether a GPU job holds the lock now. The lock is only tried, never taken, and let go at once.
fn lock_held() -> bool {
    let Ok(f) = std::fs::File::open(GPU_LOCK) else { return false };
    match f.try_lock() {
        Ok(()) => {
            let _ = f.unlock();
            false
        }
        Err(_) => true,
    }
}

/// Every process holding GPU memory, one row per card it is on.
fn gpu_uses() -> Vec<GpuUse> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"]).output();
    let Ok(out) = out else { return Vec::new() };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut f = l.splitn(3, ',').map(str::trim);
            let (pid, mib, name) = (f.next()?.parse().ok()?, f.next()?.parse().ok()?, f.next()?);
            Some(GpuUse { pid, mib, name: name.rsplit('/').next().unwrap_or(name).to_string() })
        })
        .collect()
}

/// Why running now would compete with development, on a GPU server; nothing on the CPU.
fn reasons() -> Vec<String> {
    if !uses_gpu() {
        return Vec::new();
    }
    contention(std::path::Path::new(IDLE_SENTINEL).exists(), lock_held(), &gpu_uses(), std::process::id())
}

/// Files and bytes of a model folder's weights, config and tokenizer — what a person downloads.
fn folder(dir: &Path) -> (usize, u64) {
    std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_str().is_some_and(|n| n.ends_with(".safetensors") || n == "config.json" || n == "tokenizer.json"))
                .fold((0, 0), |(n, b), e| (n + 1, b + e.metadata().map(|m| m.len()).unwrap_or(0)))
        })
        .unwrap_or((0, 0))
}

fn load(file: &Path, models: &Path) -> Result<Loaded, String> {
    use prattle_core::device;
    let (sd, td, quant) = (device::source_dtype()?, device::target_dtype()?, device::target_quant()?);
    // one handle per card, shared by the models on it (a model's source and target on one card must be
    // ONE device, or the handoff copies between them); each started before either clock runs, or
    // whichever loads first pays for CUDA's start-up
    let spec = std::env::var("PRATTLE_DEVICES").unwrap_or_else(|_| "cpu".into());
    let names: Vec<&str> = spec.split(',').map(str::trim).collect();
    let ([three_s, three_t], [fast_s, fast_t]) = layout(&names);
    let mut cards: Vec<(String, candle_core::Device)> = Vec::new();
    for n in [three_s, three_t, fast_s, fast_t] {
        if !cards.iter().any(|(m, _)| m == n) {
            let d = device::named(n)?;
            candle_core::Tensor::zeros(1, candle_core::DType::F32, &d).and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?;
            cards.push((n.to_string(), d));
        }
    }
    let card = |n: &str| cards.iter().find(|(m, _)| m == n).map(|(_, d)| d.clone()).expect("every card is made above");

    let started = Instant::now();
    let bytes = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    // a THIN file fetches its models from the hub here, each file checked against the manifest's sha256
    let parts = prattle_core::fast_llm::FastLlm::unpack(&bytes)?;
    let parts = if parts.refs.is_empty() { parts } else { parts.resolve(&prattle_core::fast_llm::hub_fetch)? };
    let fast = Model::from_fast_llm_parts(parts, card(fast_s), card(fast_t), sd, td, quant)?;
    let fast_ms = started.elapsed().as_secs_f64() * 1000.0;
    let fast_bytes = bytes.len() as u64;
    let manifest: Value = serde_json::from_str(prattle_core::fast_llm::FastLlm::unpack(&bytes)?.get("fast_llm")?).map_err(|e| e.to_string())?;
    drop(bytes);

    let name = |k: &str| manifest[k]["name"].as_str().map(str::to_string).ok_or(format!("the manifest names no {k}"));
    let (src, tgt) = (models.join(name("source")?), models.join(name("target")?));
    let bridge = models.join("bridges").join(name("bridge")?);
    let started = Instant::now();
    let read = |dir: &Path, f: &str| std::fs::read(dir.join(f)).map_err(|e| format!("{}/{f}: {e}", dir.display()));
    let weights = |dir: &Path| prattle_core::weights::read_dir(&dir.to_string_lossy());
    let mut three = Model::from_bytes_in(&read(&src, "config.json")?, &read(&src, "tokenizer.json")?, weights(&src)?, card(three_s), sd)?;
    three.place_target_on(card(three_t));
    three.target_weights_in(td);
    three.target_weights_quantized(quant);
    let bridge_bytes = std::fs::read(&bridge).map_err(|e| format!("{}: {e}", bridge.display()))?;
    three.attach_target_with(&read(&tgt, "config.json")?, weights(&tgt)?, Some(&bridge_bytes))?;
    let three_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ((sn, sb), (tn, tb)) = (folder(&src), folder(&tgt));

    let load = json!({
        "source": name("source")?, "target": name("target")?, "bridge": name("bridge")?,
        "fast": { "files": 1, "bytes": fast_bytes, "ms": fast_ms, "name": file.file_name().map(|n| n.to_string_lossy().into_owned()) },
        "cards": { "bridged": three_s, "fast": fast_s },
        // a pair is three things to fetch — two model folders and a bridge — however many files each folder holds
        "bridged": { "files": 3, "folder_files": sn + tn + 1, "bytes": sb + tb + bridge_bytes.len() as u64, "ms": three_ms },
    });
    Ok(Loaded { three, fast, load })
}

impl Compare {
    /// Starts loading — unless development has the GPUs — and returns at once; `/api/compare/status`
    /// says when the models are ready. A watcher keeps the models off the GPUs while development uses them.
    pub fn spawn(file: PathBuf, models: PathBuf) -> Arc<Compare> {
        let c = Arc::new(Compare {
            status: Mutex::new(json!({"state": "loading"})),
            loaded: Mutex::new(None),
            asked: Mutex::new(0),
            file,
            models,
            overridden: Mutex::new(false),
        });
        let first = reasons();
        if first.is_empty() {
            let me = c.clone();
            tokio::task::spawn_blocking(move || me.ensure_loaded());
        } else {
            *c.status.lock().unwrap() = json!({"state": "paused", "reasons": first});
        }
        let me = c.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let w = me.clone();
                let _ = tokio::task::spawn_blocking(move || w.watch()).await;
            }
        });
        c
    }

    /// Load the models if they are not loaded; the status says how it went.
    fn ensure_loaded(&self) -> Option<Value> {
        let mut guard = self.loaded.lock().unwrap();
        if guard.is_none() {
            *self.status.lock().unwrap() = json!({"state": "loading"});
            match load(&self.file, &self.models) {
                Ok(l) => {
                    eprintln!("compare ready: {} → {}", l.load["source"], l.load["target"]);
                    *guard = Some(l);
                }
                Err(e) => {
                    *self.status.lock().unwrap() = json!({"state": "error", "error": e});
                    return None;
                }
            }
        }
        let load = guard.as_ref().map(|l| l.load.clone());
        *self.status.lock().unwrap() = json!({"state": "ready", "load": load});
        load
    }

    /// Every two seconds on a GPU server: development busy and nobody said yes — the models leave the
    /// GPUs; development idle again — the yes lapses and the models come back.
    fn watch(&self) {
        if !uses_gpu() {
            return;
        }
        let why = reasons();
        if why.is_empty() {
            *self.overridden.lock().unwrap() = false;
            let paused = self.status.lock().unwrap()["state"] == "paused";
            if paused {
                self.ensure_loaded();
            }
            return;
        }
        if *self.overridden.lock().unwrap() {
            return;
        }
        // a question in flight holds the models; the next pass unloads them
        let Ok(mut guard) = self.loaded.try_lock() else { return };
        if guard.take().is_some() {
            eprintln!("compare paused — development has the GPUs: {why:?}");
        }
        *self.status.lock().unwrap() = json!({"state": "paused", "reasons": why});
    }

    fn ask(&self, question: &str, confirmed: bool) -> Value {
        let why = reasons();
        if !why.is_empty() {
            let mut overridden = self.overridden.lock().unwrap();
            if !confirmed && !*overridden {
                return json!({"needs_confirm": why});
            }
            *overridden = true;
        }
        if self.ensure_loaded().is_none() {
            return json!({"error": format!("could not load: {}", self.status.lock().unwrap())});
        }
        let mut guard = self.loaded.lock().unwrap();
        let Some(l) = guard.as_mut() else {
            return json!({"error": format!("not ready: {}", self.status.lock().unwrap())});
        };
        let n = {
            let mut asked = self.asked.lock().unwrap();
            *asked += 1;
            *asked - 1
        };
        let mut arms = vec![Value::Null; 3];
        for k in 0..3 {
            let i = (prattle_core::measure::first_arm(n, 0, 3) + k) % 3;
            let (model, arm, name) = match i {
                0 => (&mut l.three, Arm::Target, "large"),
                1 => (&mut l.three, Arm::Bridge, "bridged"),
                _ => (&mut l.fast, Arm::Bridge, "fast"),
            };
            arms[i] = match run(model, question, arm) {
                Ok(mut r) => {
                    r["arm"] = json!(name);
                    r["order"] = json!(k);
                    r
                }
                Err(e) => json!({"arm": name, "error": e}),
            };
        }
        let same = arms[1]["reply"] == arms[2]["reply"];
        json!({"arms": arms, "same_reply": same, "load": l.load})
    }
}

/// One arm: the time to the first token (`start_arm` then the first step, as `speedup` and the page
/// time it), the reply, and who read the prompt.
fn run(m: &mut Model, question: &str, arm: Arm) -> Result<Value, String> {
    let started = Instant::now();
    m.start_arm(question, arm)?;
    let (mut reply, mut tokens, mut first) = (String::new(), 0usize, None);
    for _ in 0..MAX_TOKENS {
        let step = m.step()?;
        first.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0);
        match step {
            Step::Text(t) => {
                reply.push_str(&t);
                tokens += 1;
            }
            Step::Done => break,
        }
    }
    let total = started.elapsed().as_secs_f64() * 1000.0;
    let ttft = first.unwrap_or(total);
    let c = m.counters();
    let decode_s = (total - ttft) / 1000.0;
    Ok(json!({
        "reply": reply.trim(),
        "ttft_ms": ttft,
        "total_ms": total,
        "tokens": tokens,
        "tokens_per_s": if decode_s > 0.0 && tokens > 1 { (tokens - 1) as f64 / decode_s } else { 0.0 },
        "counters": { "prompt": c.prompt_tokens, "reader": c.reader_tokens, "history": c.generator_history, "bridge": c.bridge_tokens },
    }))
}

pub fn routes(c: Arc<Compare>) -> Router {
    Router::new()
        .route("/compare", get(|| async { Html(include_str!("compare.html")) }))
        .route("/api/compare/status", get(status))
        .route("/api/compare/gpus", get(gpus))
        .route("/api/compare", post(ask))
        .with_state(c)
}

async fn status(State(c): State<Arc<Compare>>) -> impl IntoResponse {
    Json(c.status.lock().unwrap().clone())
}

/// What the page asks before a question: would running now compete with development?
async fn gpus(State(_): State<Arc<Compare>>) -> impl IntoResponse {
    let why = tokio::task::spawn_blocking(reasons).await.unwrap_or_default();
    Json(json!({"gpu": uses_gpu(), "reasons": why}))
}

async fn ask(State(c): State<Arc<Compare>>, Json(req): Json<Value>) -> impl IntoResponse {
    let question = req["question"].as_str().unwrap_or("").trim().to_string();
    if question.is_empty() {
        return Json(json!({"error": "an empty question"}));
    }
    let confirmed = req["confirmed"] == true;
    let answer = tokio::task::spawn_blocking(move || c.ask(&question, confirmed)).await;
    Json(answer.unwrap_or_else(|e| json!({"error": format!("the comparison stopped: {e}")})))
}

/// One process holding GPU memory, as `nvidia-smi --query-compute-apps` lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuUse {
    pub pid: u32,
    pub mib: u64,
    pub name: String,
}

/// Why running the comparison now would compete with something else for the GPUs — empty when nothing
/// would. Development is idle only while its sentinel exists; a held GPU lock is a job running now;
/// another process's memory on a card is something already loaded there. This server's own is not.
pub fn contention(idle: bool, lock_held: bool, gpus: &[GpuUse], own_pid: u32) -> Vec<String> {
    let mut why = Vec::new();
    if !idle {
        why.push(format!("development may be using the GPUs (no {IDLE_SENTINEL}; `just gpu-idle` sets it)"));
    }
    if lock_held {
        why.push(format!("a GPU job is running now (it holds {GPU_LOCK})"));
    }
    for g in gpus.iter().filter(|g| g.pid != own_pid) {
        why.push(format!("{} (pid {}) holds {} MiB on a GPU", g.name, g.pid, g.mib));
    }
    why
}

/// Where each model of the comparison runs, from the devices `PRATTLE_DEVICES` names: the three-file
/// model's (source, target), then the FAST_LLM's.
pub fn layout<'a>(devices: &[&'a str]) -> ([&'a str; 2], [&'a str; 2]) {
    let first = devices.first().copied().unwrap_or("cpu");
    let second = devices.get(1).copied().unwrap_or(first);
    ([first, first], [second, second])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cards: the three-file model on one, the FAST_LLM on the other — each arm a whole card, so two
    /// 8B targets never share one. One device: everything there.
    #[test]
    fn the_three_files_take_one_card_and_the_fast_llm_the_other() {
        assert_eq!(layout(&["cuda:0", "cuda:1"]), (["cuda:0", "cuda:0"], ["cuda:1", "cuda:1"]));
        assert_eq!(layout(&["cuda:1"]), (["cuda:1", "cuda:1"], ["cuda:1", "cuda:1"]));
        assert_eq!(layout(&["cpu"]), (["cpu", "cpu"], ["cpu", "cpu"]));
    }

    fn using(pid: u32, mib: u64, name: &str) -> GpuUse {
        GpuUse { pid, mib, name: name.into() }
    }

    #[test]
    fn the_comparison_asks_first_only_when_something_else_has_the_gpus() {
        let me = 100;
        assert!(contention(true, false, &[using(me, 1400, "prattle-server"), using(me, 3300, "prattle-server")], me).is_empty(), "idle development and only this server's own memory: no question");
        let busy = contention(false, false, &[], me);
        assert_eq!(busy.len(), 1, "no idle sentinel means development may be using the GPUs: {busy:?}");
        let locked = contention(true, true, &[], me);
        assert!(locked.iter().any(|r| r.contains("lock")), "a held GPU lock is a job running now: {locked:?}");
        let other = contention(true, false, &[using(me, 1400, "prattle-server"), using(7, 20000, "other-job")], me);
        assert!(other.len() == 1 && other[0].contains("other-job") && other[0].contains("20000"), "another process's memory names it: {other:?}");
    }
}
