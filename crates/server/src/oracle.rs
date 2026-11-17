//! /chat — LatentPort's own 4B → 27B pair, answered by upstream's runtime in a worker process
//! (scripts/oracle_worker.py) over JSON lines. One question at a time: both models share the GPUs,
//! and a second question timed while the first runs would be a number about contention.

use std::process::Stdio;
use std::sync::Arc;

use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, RwLock};

struct Pipe {
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
    next_id: u64,
}

pub struct Worker {
    /// `loading`, `ready` or `error`, with the pair's names or the reason — what /api/oracle answers.
    status: RwLock<Value>,
    pipe: Mutex<Option<Pipe>>,
}

impl Worker {
    /// Starts the worker and returns at once; the models load in the background. The worker ends
    /// when its stdin closes, which is when this process does.
    pub fn spawn(command: &str) -> Arc<Worker> {
        let worker = Arc::new(Worker { status: RwLock::new(json!({"state": "loading"})), pipe: Mutex::new(None) });
        let spawned = Command::new("sh").args(["-c", command]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn();
        let w = worker.clone();
        tokio::spawn(async move {
            let mut child = match spawned {
                Ok(c) => c,
                Err(e) => return *w.status.write().await = json!({"state": "error", "error": format!("cannot start the oracle worker: {e}")}),
            };
            let mut pipe = w.pipe.lock().await; // held until ready, so a question waits for the models
            let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
            let first = lines.next_line().await.ok().flatten().and_then(|l| serde_json::from_str::<Value>(&l).ok());
            match first {
                Some(v) if v["ready"] == true => {
                    eprintln!("oracle worker ready: {} → {}", v["source"], v["target"]);
                    *w.status.write().await = json!({"state": "ready", "source": v["source"], "target": v["target"]});
                    *pipe = Some(Pipe { stdin: child.stdin.take().unwrap(), lines, next_id: 1 });
                    drop(child); // not killed on drop: the worker ends when its stdin closes
                }
                _ => {
                    let code = child.wait().await.ok().and_then(|s| s.code());
                    *w.status.write().await = json!({"state": "error", "error": format!(
                        "the oracle worker stopped before it was ready (exit {code:?}) — its log is target/oracle/chat-native.log")});
                }
            }
        });
        worker
    }

    /// One line to the worker, one line back. `req` gains the request id.
    async fn request(&self, mut req: Value) -> Value {
        let mut guard = self.pipe.lock().await;
        let Some(pipe) = guard.as_mut() else {
            return json!({"error": format!("the oracle is not ready: {}", self.status.read().await)});
        };
        let id = pipe.next_id;
        pipe.next_id += 1;
        req["id"] = json!(id);
        let line = req.to_string() + "\n";
        if let Err(e) = pipe.stdin.write_all(line.as_bytes()).await {
            *guard = None;
            *self.status.write().await = json!({"state": "error", "error": format!("the oracle worker went away: {e}")});
            return json!({"error": "the oracle worker went away"});
        }
        match pipe.lines.next_line().await {
            Ok(Some(l)) => serde_json::from_str(&l).unwrap_or_else(|e| json!({"error": format!("unreadable reply from the worker: {e}")})),
            _ => {
                *guard = None;
                *self.status.write().await = json!({"state": "error", "error": "the oracle worker stopped"});
                json!({"error": "the oracle worker stopped"})
            }
        }
    }
}

pub fn routes(worker: Arc<Worker>) -> Router {
    Router::new()
        .route("/chat", get(|| async { Html(include_str!("chat.html")) }))
        .route("/api/oracle", get(status))
        .route("/api/chat", post(chat))
        .route("/api/closeness", post(closeness))
        .with_state(worker)
}

async fn status(State(w): State<Arc<Worker>>) -> impl IntoResponse {
    Json(w.status.read().await.clone())
}

async fn chat(State(w): State<Arc<Worker>>, Json(req): Json<Value>) -> impl IntoResponse {
    let question = req["question"].as_str().unwrap_or("").trim();
    if question.is_empty() {
        return Json(json!({"error": "an empty question"}));
    }
    let mode = if req["mode"] == "native" { "native" } else { "bridge" };
    // A conversation is the 27B's retained state in the worker; the page names it, and a new name
    // (or a change of mode) starts a fresh one there.
    let conversation = req["conversation"].as_str().unwrap_or("");
    Json(w.request(json!({"question": question, "mode": mode, "conversation": conversation})).await)
}

/// How close the bridged 27B is to the 27B reading `question` itself: the native 27B's own reply
/// scored teacher-forced three ways (scripts/oracle_scoring.py), and NCR from prattle_core — the
/// same measurement as the fidelity table, on this one prompt.
async fn closeness(State(w): State<Arc<Worker>>, Json(req): Json<Value>) -> impl IntoResponse {
    let question = req["question"].as_str().unwrap_or("").trim();
    if question.is_empty() {
        return Json(json!({"error": "an empty question"}));
    }
    let mut r = w.request(json!({"closeness": question})).await;
    let nll = |k: &str| r["nll"][k].as_f64();
    if let (Some(empty), Some(bridged), Some(native)) = (nll("empty"), nll("bridged"), nll("native")) {
        r["ncr"] = json!(prattle_core::measure::ncr(empty, bridged, native));
    }
    Json(r)
}
