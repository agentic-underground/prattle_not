//! The WebGPU page's `Session`: the same methods as `prattle-web`'s, so one worker drives either
//! engine — except that loading and `step` return promises, because a GPU read completes on the
//! browser's event loop.

use std::cell::RefCell;
use std::rc::Rc;

use prattle_core::{Arm, Step};
use wasm_bindgen::prelude::*;

use crate::model::GpuModel;

#[wasm_bindgen]
pub struct Session(Rc<RefCell<GpuModel>>);

fn js(e: String) -> JsValue {
    JsError::new(&e).into()
}

#[wasm_bindgen]
impl Session {
    /// A session from a model's config.json, tokenizer.json and model.safetensors: a promise.
    pub fn from_bytes(config: Vec<u8>, tokenizer: Vec<u8>, weights: Vec<u8>) -> js_sys::Promise {
        // A panic traps the whole module with "unreachable"; keep its message where the worker can
        // read it and tell the page what actually went wrong.
        std::panic::set_hook(Box::new(|info| {
            let _ = js_sys::Reflect::set(&js_sys::global(), &"prattlePanic".into(), &info.to_string().into());
        }));
        wasm_bindgen_futures::future_to_promise(async move {
            let m = GpuModel::from_bytes(&config, &tokenizer, &weights).await.map_err(js)?;
            Ok(Session(Rc::new(RefCell::new(m))).into())
        })
    }

    /// A session whose reading model's weights then arrive in pieces — `feed`, `next_file` between
    /// the files of a sharded checkpoint, `finish_source` at the end — so a file larger than this
    /// module's 4 GiB of memory still loads: a promise.
    pub fn open(config: Vec<u8>, tokenizer: Vec<u8>) -> js_sys::Promise {
        std::panic::set_hook(Box::new(|info| {
            let _ = js_sys::Reflect::set(&js_sys::global(), &"prattlePanic".into(), &info.to_string().into());
        }));
        wasm_bindgen_futures::future_to_promise(async move {
            let m = GpuModel::open(&config, &tokenizer).await.map_err(js)?;
            Ok(Session(Rc::new(RefCell::new(m))).into())
        })
    }

    /// The next piece of the file arriving, in order.
    pub fn feed(&self, bytes: &[u8]) -> Result<(), JsError> {
        self.0.borrow_mut().feed(bytes).map_err(|e| JsError::new(&e))
    }

    /// The model arriving goes on to its next file.
    pub fn next_file(&self) -> Result<(), JsError> {
        self.0.borrow_mut().next_file().map_err(|e| JsError::new(&e))
    }

    pub fn finish_source(&self) -> Result<(), JsError> {
        self.0.borrow_mut().finish_source().map_err(|e| JsError::new(&e))
    }

    /// Begin the answering model; its weights arrive through `feed`, then `finish_target`.
    pub fn begin_target(&self, config: Vec<u8>) -> Result<(), JsError> {
        self.0.borrow_mut().begin_target(&config).map_err(|e| JsError::new(&e))
    }

    pub fn finish_target(&self, bridge: Vec<u8>) -> Result<(), JsError> {
        let bridge = (!bridge.is_empty()).then_some(bridge.as_slice());
        self.0.borrow_mut().finish_target(bridge).map_err(|e| JsError::new(&e))
    }

    pub fn attach_target_bytes(&self, config: Vec<u8>, weights: Vec<u8>, bridge: Vec<u8>) -> Result<(), JsError> {
        let bridge = (!bridge.is_empty()).then_some(bridge.as_slice());
        self.0.borrow_mut().attach_target(&config, &weights, bridge).map_err(|e| JsError::new(&e))
    }

    pub fn has_target(&self) -> bool {
        self.0.borrow().has_target()
    }

    pub fn start(&self, question: &str, bridge: bool) -> Result<(), JsError> {
        let arm = if bridge { Arm::Bridge } else { Arm::Native };
        self.0.borrow_mut().start_arm(question, arm).map_err(|e| JsError::new(&e))
    }

    pub fn start_arm(&self, question: &str, arm: &str, donor: &str) -> Result<(), JsError> {
        let arm = match arm {
            "native" => Arm::Native,
            "target" => Arm::Target,
            "source" => Arm::Source,
            "bridge" => Arm::Bridge,
            "restore" => Arm::Restore,
            "donor" => Arm::WrongDonor(donor.to_string()),
            other => return Err(JsError::new(&format!("no arm called {other:?}"))),
        };
        self.0.borrow_mut().start_arm(question, arm).map_err(|e| JsError::new(&e))
    }

    pub fn synthetic_prompt(&self, tokens: usize, variant: u32) -> Result<String, JsError> {
        self.0.borrow().synthetic_prompt(tokens, variant).map_err(|e| JsError::new(&e))
    }

    pub fn prompt_tokens(&self, question: &str) -> Result<usize, JsError> {
        self.0.borrow().prompt_tokens(question).map_err(|e| JsError::new(&e))
    }

    pub fn counters(&self) -> Vec<u32> {
        let c = self.0.borrow().counters();
        [c.prompt_tokens, c.reader_tokens, c.generator_history, c.bridge_tokens].map(|n| n as u32).to_vec()
    }

    /// A promise of the text the next token adds, or `undefined` when the reply is finished. The
    /// worker awaits each step before the next, so the model is never borrowed twice.
    pub fn step(&self) -> js_sys::Promise {
        let m = self.0.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            #[allow(clippy::await_holding_refcell_ref)]
            let r = m.borrow_mut().step().await.map_err(js)?;
            Ok(match r {
                Step::Text(t) => JsValue::from_str(&t),
                Step::Done => JsValue::UNDEFINED,
            })
        })
    }
}

/// prattle-core's paired statistic, as `prattle-web` exports it.
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
