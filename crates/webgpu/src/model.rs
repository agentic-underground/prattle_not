//! The page's model on the GPU engine: prattle-core's `Model` contract — arms, counters, greedy
//! decoding with the same repeat penalty, one step at a time — with every forward pass on WebGPU.
//! The arms' schedule is core's `start_arm`, line for line, so the two pages measure the same thing.

use prattle_core::model::{Counters, MAX_NEW_TOKENS, REPEAT_PENALTY, REPEAT_WINDOW};
use prattle_core::text::{Template, delta};
use prattle_core::{Arm, Step};
use tokenizers::Tokenizer;

use crate::Gpu;
use crate::bridge::Bridge;
use crate::llama::Llama;

/// Positions of KV cache per model: a 2K-token prompt and a full reply.
const MAX_CTX: usize = 2560;

pub struct GpuModel {
    gpu: Gpu,
    source: Llama,
    target: Option<Llama>,
    bridge: Option<Bridge>,
    tokenizer: Tokenizer,
    template: Template,
    stop: Vec<u32>,
    /// How many of the prompt's last tokens the answering model reads itself when bridged.
    bridge_tokens: usize,
    /// Whether the reply in progress is the target's.
    bridged: bool,
    pending: Vec<u32>,
    generated: Vec<u32>,
    shown: String,
    counters: Counters,
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// candle's `apply_repeat_penalty`: each distinct recent token's logit pushed away from zero once.
fn penalize(logits: &mut [f32], recent: &[u32]) {
    let mut seen = std::collections::HashSet::new();
    for &t in recent {
        if seen.insert(t)
            && let Some(l) = logits.get_mut(t as usize)
        {
            *l = if *l >= 0.0 { *l / REPEAT_PENALTY } else { *l * REPEAT_PENALTY };
        }
    }
}

/// candle's greedy `LogitsProcessor`: the largest logit, the last of equals.
fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0)
}

impl GpuModel {
    /// The reading model, from its config.json, tokenizer.json and safetensors bytes, on the best GPU.
    pub async fn from_bytes(config: &[u8], tokenizer: &[u8], weights: &[u8]) -> Result<GpuModel, String> {
        let gpu = Gpu::new().await?;
        let source = Llama::load(&gpu, config, weights, MAX_CTX)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer).map_err(err)?;
        let stop = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "<|end_of_text|>"].iter().filter_map(|t| tokenizer.token_to_id(t)).collect();
        Ok(GpuModel {
            template: Template::from_config(config)?,
            gpu,
            source,
            target: None,
            bridge: None,
            tokenizer,
            stop,
            bridge_tokens: 1,
            bridged: false,
            pending: Vec::new(),
            generated: Vec::new(),
            shown: String::new(),
            counters: Counters::default(),
        })
    }

    /// Attach the model that answers when bridged, through a fitted bridge. One token crosses with the
    /// paper's correction; this engine runs corrected bridges only.
    pub fn attach_target(&mut self, config: &[u8], weights: &[u8], bridge: Option<&[u8]>) -> Result<(), String> {
        let bridge = bridge.ok_or("the WebGPU engine bridges through a fitted bridge file only")?;
        self.target = Some(Llama::load(&self.gpu, config, weights, MAX_CTX)?);
        self.bridge = Some(Bridge::load(&self.gpu, bridge)?);
        self.bridge_tokens = 1;
        Ok(())
    }

    pub fn has_target(&self) -> bool {
        self.target.is_some()
    }

    pub fn counters(&self) -> Counters {
        self.counters
    }

    fn encode(&self, question: &str) -> Result<Vec<u32>, String> {
        Ok(self.tokenizer.encode(self.template.prompt(question), false).map_err(err)?.get_ids().to_vec())
    }

    pub fn prompt_tokens(&self, question: &str) -> Result<usize, String> {
        Ok(self.encode(question)?.len())
    }

    /// prattle-core's synthetic incident log of at least `tokens` prompt tokens, variant `v`.
    pub fn synthetic_prompt(&self, tokens: usize, v: u32) -> Result<String, String> {
        use prattle_core::measure::{SYNTHETIC_QUESTION, incident_line};
        let mut text = String::new();
        let mut i = 0u32;
        while self.prompt_tokens(&format!("{text}{SYNTHETIC_QUESTION}"))? < tokens {
            text.push_str(&incident_line(v, i));
            i += 1;
        }
        Ok(format!("{text}{SYNTHETIC_QUESTION}"))
    }

    /// Begin one arm, as prattle-core's `Model::start_arm` does: the history is read here, the last
    /// token (or `bridge_tokens`) in the first `step`.
    pub fn start_arm(&mut self, question: &str, arm: Arm) -> Result<(), String> {
        let ids = self.encode(question)?;
        let n = ids.len();
        if n < 2 {
            return Err("a one-token prompt leaves nothing to bridge".into());
        }
        self.generated.clear();
        self.shown.clear();
        let memory = match &arm {
            Arm::Native => None,
            Arm::Bridge => Some(ids[..n - self.bridge_tokens.min(n - 1)].to_vec()),
            Arm::Restore | Arm::Target | Arm::Source => Some(ids[..n - 1].to_vec()),
            Arm::WrongDonor(donor) => {
                let d = self.encode(donor)?;
                Some(d[..d.len() - self.bridge_tokens.min(d.len() - 1)].to_vec())
            }
        };
        let Some(history) = memory else {
            self.source.reset();
            self.pending = ids;
            self.bridged = false;
            self.counters = Counters { prompt_tokens: n, reader_tokens: 0, generator_history: n, bridge_tokens: 0 };
            return Ok(());
        };
        let (reads_target, answers_target, own) = match arm {
            Arm::Target => (true, true, true),
            Arm::Source => (false, false, true),
            Arm::Restore => (false, false, false),
            _ => (false, true, false),
        };
        if (reads_target || answers_target) && self.target.is_none() {
            return Err("this arm needs a target attached — turn the bridge on".into());
        }
        if answers_target && !reads_target {
            let (target, bridge) = (self.target.as_mut().unwrap(), self.bridge.as_ref().unwrap());
            self.source.reset();
            let hidden = self.source.prefill_capture(&self.gpu, &history);
            let kvs = bridge.translate(&self.gpu, &hidden, target);
            target.install(&self.gpu, &kvs, history.len());
        } else {
            let reader = if reads_target { self.target.as_mut().unwrap() } else { &mut self.source };
            reader.reset();
            reader.prefill(&self.gpu, &history);
        }
        let tail = if matches!(arm, Arm::Bridge | Arm::WrongDonor(_)) { self.bridge_tokens.min(n - 1) } else { 1 };
        self.pending = ids[n - tail..].to_vec();
        self.bridged = answers_target;
        self.counters = if own {
            Counters { prompt_tokens: n, reader_tokens: 0, generator_history: n, bridge_tokens: 0 }
        } else {
            Counters { prompt_tokens: n, reader_tokens: history.len(), generator_history: 0, bridge_tokens: tail }
        };
        Ok(())
    }

    /// Run the answering model once — the pending prompt or bridge token, then one token at a time —
    /// and return the text that token added.
    pub async fn step(&mut self) -> Result<Step, String> {
        if self.pending.is_empty() || self.generated.len() >= MAX_NEW_TOKENS {
            return Ok(Step::Done);
        }
        let model = match (&mut self.target, self.bridged) {
            (Some(t), true) => t,
            _ => &mut self.source,
        };
        let mut logits = model.forward(&self.gpu, &self.pending).await;
        penalize(&mut logits, &self.generated[self.generated.len().saturating_sub(REPEAT_WINDOW)..]);
        let next = argmax(&logits);
        if self.stop.contains(&next) {
            self.pending.clear();
            return Ok(Step::Done);
        }
        self.pending = vec![next];
        self.generated.push(next);
        let now = self.tokenizer.decode(&self.generated, true).map_err(err)?;
        let new = delta(&self.shown, &now).to_string();
        if !new.is_empty() || !now.starts_with(&self.shown) {
            self.shown = now;
        }
        Ok(Step::Text(new))
    }
}
