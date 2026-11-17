//! The page's model session, run inside a Web Worker by `www/worker.js`: the worker fetches (and
//! caches) the weights and hands this the bytes, then generates one step per call. `www/index.html`
//! drives the worker and exposes `window.prattle`.

use prattle_core::{Arm, Model, Step};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Session(Model);

#[wasm_bindgen]
impl Session {
    /// A session from a model's config.json, tokenizer.json and model.safetensors.
    pub fn from_bytes(config: Vec<u8>, tokenizer: Vec<u8>, weights: Vec<u8>) -> Result<Session, JsError> {
        Model::from_bytes(&config, &tokenizer, weights).map(Session).map_err(|e| JsError::new(&e))
    }

    /// A session from ONE FAST_LLM file (`prattle_core::fast_llm`): the reading model, the answering
    /// model and their bridge, attached as `attach_target_bytes` would. The file's bytes are let go as
    /// soon as they are split, so the page's 4 GiB holds one copy of the weights while the models build.
    pub fn from_fast_llm(bytes: Vec<u8>) -> Result<Session, JsError> {
        let parts = prattle_core::fast_llm::FastLlm::unpack(&bytes).map_err(|e| JsError::new(&e))?;
        drop(bytes);
        let (cpu, f32) = (candle_core::Device::Cpu, candle_core::DType::F32);
        Model::from_fast_llm_parts(parts, cpu.clone(), cpu, f32, f32, None).map(Session).map_err(|e| JsError::new(&e))
    }

    /// Attach the TARGET that answers from this model's reading when the bridge is on, from its
    /// config.json and model.safetensors; it shares this session's tokens. `bridge` is a fitted bridge
    /// file, or empty for direct copy between equal shapes.
    pub fn attach_target_bytes(&mut self, config: Vec<u8>, weights: Vec<u8>, bridge: Vec<u8>) -> Result<(), JsError> {
        let map = if bridge.is_empty() { None } else { Some(bridge.as_slice()) };
        self.0.attach_target_with(&config, weights, map).map_err(|e| JsError::new(&e))
    }

    pub fn has_target(&self) -> bool {
        self.0.has_target()
    }

    pub fn start(&mut self, question: &str, bridge: bool) -> Result<(), JsError> {
        self.0.start(question, bridge).map_err(|e| JsError::new(&e))
    }

    /// Begin one control arm: `native`, `bridge`, `restore`, or `donor` (from `donor`'s memory).
    pub fn start_arm(&mut self, question: &str, arm: &str, donor: &str) -> Result<(), JsError> {
        let arm = match arm {
            "native" => Arm::Native,
            "target" => Arm::Target,
            "source" => Arm::Source,
            "bridge" => Arm::Bridge,
            "restore" => Arm::Restore,
            "donor" => Arm::WrongDonor(donor.to_string()),
            other => return Err(JsError::new(&format!("no arm called {other:?}"))),
        };
        self.0.start_arm(question, arm).map_err(|e| JsError::new(&e))
    }

    /// A synthetic prompt of at least `tokens` prompt tokens; `variant` makes it distinct.
    pub fn synthetic_prompt(&self, tokens: usize, variant: u32) -> Result<String, JsError> {
        self.0.synthetic_prompt(tokens, variant).map_err(|e| JsError::new(&e))
    }

    pub fn prompt_tokens(&self, question: &str) -> Result<usize, JsError> {
        self.0.prompt_tokens(question).map_err(|e| JsError::new(&e))
    }

    /// `[prompt tokens, reader tokens, generator history, bridge tokens]` for the reply in progress.
    pub fn counters(&self) -> Vec<u32> {
        let c = self.0.counters();
        [c.prompt_tokens, c.reader_tokens, c.generator_history, c.bridge_tokens].map(|n| n as u32).to_vec()
    }

    /// The text the next token adds, or `undefined` when the reply is finished.
    pub fn step(&mut self) -> Result<Option<String>, JsError> {
        match self.0.step().map_err(|e| JsError::new(&e))? {
            Step::Text(t) => Ok(Some(t)),
            Step::Done => Ok(None),
        }
    }
}

/// The paired statistic over `(prompt, native ms, bridge ms)` triples, as
/// `[speedup, ci low, ci high, median ratio, prompts, pairs]`, or empty with no pairs.
#[wasm_bindgen]
pub fn summarize(prompts: Vec<u32>, native_ms: Vec<f64>, bridge_ms: Vec<f64>) -> Vec<f64> {
    let pairs: Vec<prattle_core::measure::Pair> = prompts
        .iter()
        .zip(native_ms.iter().zip(&bridge_ms))
        .map(|(&p, (&n, &b))| prattle_core::measure::Pair { prompt: p as usize, native_ms: n, bridge_ms: b })
        .collect();
    match prattle_core::measure::summarize(&pairs) {
        Some(s) => vec![s.speedup, s.ci_low, s.ci_high, s.median_ratio, s.prompts as f64, s.pairs as f64],
        None => vec![],
    }
}

#[wasm_bindgen]
pub fn first_arm(prompt: u32, rep: u32, arms: u32) -> u32 {
    prattle_core::measure::first_arm(prompt as usize, rep as usize, arms as usize) as u32
}

#[wasm_bindgen]
pub fn prompt_hash(text: &str) -> String {
    prattle_core::measure::prompt_hash(text)
}
