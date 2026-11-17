//! A Llama-architecture model, loaded from bytes, generating one step at a time so the caller
//! decides when to paint. Bytes rather than paths: the page has no filesystem.
//!
//! THE BRIDGE (docs/BRIDGE.md §1, slice 1). With a TARGET attached, `start(.., bridge = true)` has
//! this model — the SOURCE — prefill every prompt token but the last, copies its attention state
//! into the target (direct copy, so the two must share geometry), and the target processes the
//! last token as the one BRIDGE token and answers. The target never reads the prompt history.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::LlamaConfig;
use tokenizers::Tokenizer;

use crate::bridge::{KvMap, fill_from_source, translate_hidden_with};
use crate::decoder::Decoder;
use crate::llama::{Kv, Llama};
use crate::text::{Template, delta};

/// When the target is reached through a fitted bridge WITHOUT the paper's correction, it reads this
/// many of the prompt's last tokens itself. Measured on held-out questions (135M → 360M): with 1, the
/// uncorrected 360M keeps the shape of an answer but not its content (3 of 7 right); with 8 it answers
/// as well as reading everything (6 of 7). WITH the correction (docs/BRIDGE.md §1.5) one token — the
/// paper's handoff — answers 6 of 7, so a corrected bridge uses 1.
pub const TRANSLATED_BRIDGE_TOKENS: usize = 8;

/// A cap on one reply, so a model that never says stop still finishes.
pub const MAX_NEW_TOKENS: usize = 256;

/// Greedy decoding loops on a small model; this discourages repeating the reply's recent tokens.
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_WINDOW: usize = 64;

/// One arm of a comparison: who reads the prompt history, and who answers.
pub enum Arm {
    /// This model reads everything and answers.
    Native,
    /// The TARGET reads everything and answers — the native arm of a speed-up measurement.
    /// Measurement schedule (see `start_arm`).
    Target,
    /// This model reads everything and answers, on the measurement schedule — the denominator of
    /// the speed-up ceiling.
    Source,
    /// This model reads the history; the target answers from its state.
    Bridge,
    /// This model reads the history, its state is installed back into ITSELF, and it answers —
    /// which must reproduce `Native` exactly.
    Restore,
    /// This model reads ANOTHER question's history; the target answers the asked question's bridge
    /// token from that state.
    WrongDonor(String),
}

pub enum Step {
    /// New visible text — possibly empty, when a token completes no character yet.
    Text(String),
    Done,
}

/// Who read what, for the reply in progress. The page shows these; the story asserts them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    pub prompt_tokens: usize,
    /// Prompt tokens the SOURCE prefilled on the answering model's behalf.
    pub reader_tokens: usize,
    /// Prompt tokens the ANSWERING model prefilled from the prompt history itself.
    pub generator_history: usize,
    /// Prompt tokens the answering model processed on top of transported state.
    pub bridge_tokens: usize,
}

pub struct Model {
    llama: Decoder,
    target: Option<Decoder>,
    /// The fitted bridge into the target, when the two do not share geometry (docs/BRIDGE.md §1.4).
    map: Option<KvMap>,
    /// Whether the reply in progress is the target's.
    bridged: bool,
    /// How many of the prompt's last tokens the answering model reads itself when bridged (≥ 1).
    bridge_tokens: usize,
    tokenizer: Tokenizer,
    /// The reading model's prompt template, from its config.
    template: Template,
    /// A system turn for every prompt (`set_system`); None is the family's default.
    system: Option<String>,
    device: Device,
    /// Where the target and its bridge live — `device` unless `place_target_on` says otherwise.
    target_device: Device,
    /// The precision the target's weights are held in (`load_decoder_in`).
    target_dtype: DType,
    target_quant: Option<candle_core::quantized::GgmlDType>,
    stop: Vec<u32>,
    sampler: LogitsProcessor,
    kv: Kv,
    /// Tokens not yet fed to the model: the prompt (or the bridge token) first, then one per step.
    pending: Vec<u32>,
    pos: usize,
    generated: Vec<u32>,
    shown: String,
    counters: Counters,
}

/// Every slot of `kv` on `device`.
fn kv_moved(kv: &Kv, device: &Device) -> candle_core::Result<Kv> {
    let slot = |s: &Option<(Tensor, Tensor)>| -> candle_core::Result<Option<(Tensor, Tensor)>> {
        s.as_ref().map(|(a, b)| Ok((crate::device::moved(a, device)?, crate::device::moved(b, device)?))).transpose()
    };
    Ok(Kv::from_layers(kv.layers().iter().map(slot).collect::<candle_core::Result<_>>()?))
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The decoder a `config.json` names — Gemma 3 (`model_type` gemma3 / gemma3_text, or a multimodal
/// checkpoint whose text_config is) or the Llama family — from one safetensors buffer.
pub fn load_decoder(config_json: &[u8], weights: Vec<u8>, device: &Device) -> Result<Decoder, String> {
    load_decoder_in(config_json, weights, device, DType::F32)
}

/// As `load_decoder`, its weights held in `dtype` — half precision where f32 would not fit the card
/// (`crate::half`). Everything but Gemma 3, so far.
/// As `load_decoder_in`, every linear layer's weights quantized at load to `quant` (`crate::half`) —
/// a target too large for its card even at half precision. `None` is `load_decoder_in`.
pub fn load_decoder_q(config_json: &[u8], weights: Vec<u8>, device: &Device, dtype: DType, quant: Option<candle_core::quantized::GgmlDType>) -> Result<Decoder, String> {
    load_decoder_q_ready(config_json, weights, device, dtype, quant, None)
}

/// As `load_decoder_q`, its linear weights given already quantized (`ready`, keyed as
/// `half::Shared::quantize_linears` keys them) — a pre-quantized FAST_LLM — so none is quantized again.
pub fn load_decoder_q_ready(config_json: &[u8], weights: Vec<u8>, device: &Device, dtype: DType, quant: Option<candle_core::quantized::GgmlDType>, ready: Option<std::collections::HashMap<String, candle_core::quantized::QTensor>>) -> Result<Decoder, String> {
    let Some(quant) = quant else { return load_decoder_in(config_json, weights, device, dtype) };
    let json: serde_json::Value = serde_json::from_slice(config_json).map_err(err)?;
    let kind = json["text_config"]["model_type"].as_str().or(json["model_type"].as_str()).unwrap_or("");
    if kind.starts_with("gemma3") || kind.starts_with("qwen3_5") || kind == "lfm2" {
        return Err(format!("{kind}: quantized weights are not supported yet — the Llama family (Llama, Qwen3, SmolLM2) and Granite 4.0 load quantized"));
    }
    if kind == "gemma4_text" || kind == "gemma4" {
        let shared = crate::half::Shared::new(weights).map_err(err)?;
        let card = shared.builder(dtype, device);
        let built = crate::half::quantize_with(quant, &shared, ready, device, || Ok(crate::gemma4::Gemma4::from_builder(config_json, card)));
        return built.map_err(err)?.map(Decoder::Gemma4);
    }
    if kind == "granitemoehybrid" {
        let shared = crate::half::Shared::new(weights).map_err(err)?;
        let card = shared.builder(dtype, device);
        let built = crate::half::quantize_with(quant, &shared, ready, device, || Ok(crate::granite::Granite::from_builder(config_json, card)));
        return built.map_err(err)?.map(Decoder::Granite);
    }
    let raw: LlamaConfig = serde_json::from_slice(config_json).map_err(err)?;
    let cfg = raw.into_config(false);
    let head_dim = json["head_dim"].as_u64().map_or(cfg.hidden_size / cfg.num_attention_heads, |d| d as usize);
    let shared = crate::half::Shared::new(weights).map_err(err)?;
    let card = shared.builder(dtype, device);
    crate::half::quantize_with(quant, &shared, ready, device, || Llama::load_with_head_dim(card, &cfg, head_dim)).map(Decoder::Llama).map_err(err)
}

pub fn load_decoder_in(config_json: &[u8], weights: Vec<u8>, device: &Device, dtype: DType) -> Result<Decoder, String> {
    let json: serde_json::Value = serde_json::from_slice(config_json).map_err(err)?;
    let kind = json["text_config"]["model_type"].as_str().or(json["model_type"].as_str()).unwrap_or("");
    if dtype != DType::F32 && kind.starts_with("gemma3") {
        return Err(format!("{kind}: weights in {dtype:?} are not supported yet — the Llama family, Qwen3.5, Granite 4.0 and LFM2 load at half precision"));
    }
    if kind.starts_with("gemma3") {
        Ok(Decoder::Gemma3(crate::gemma3::Gemma3::load(config_json, weights, device)?))
    } else if kind.starts_with("gemma4") {
        Ok(Decoder::Gemma4(crate::gemma4::Gemma4::load_in(config_json, weights, device, dtype)?))
    } else if kind == "lfm2" {
        Ok(Decoder::Lfm2(crate::lfm2::Lfm2::load_in(config_json, weights, device, dtype)?))
    } else if kind == "granitemoehybrid" {
        Ok(Decoder::Granite(crate::granite::Granite::load_in(config_json, weights, device, dtype)?))
    } else if kind.starts_with("qwen3_5") {
        Ok(Decoder::Qwen35(crate::qwen3_5::Qwen35::load_in(config_json, weights, device, dtype)?))
    } else {
        Ok(Decoder::Llama(load_in(config_json, weights, device, dtype)?))
    }
}

/// A Llama-family decoder from its `config.json` and one safetensors buffer.
pub fn load(config_json: &[u8], weights: Vec<u8>, device: &Device) -> Result<Llama, String> {
    load_in(config_json, weights, device, DType::F32)
}

/// As `load`, its weights held in `dtype`.
pub fn load_in(config_json: &[u8], weights: Vec<u8>, device: &Device, dtype: DType) -> Result<Llama, String> {
    let raw: LlamaConfig = serde_json::from_slice(config_json).map_err(err)?;
    let json: serde_json::Value = serde_json::from_slice(config_json).map_err(err)?;
    let cfg = raw.into_config(false);
    let head_dim = json["head_dim"].as_u64().map_or(cfg.hidden_size / cfg.num_attention_heads, |d| d as usize);
    let vb = VarBuilder::from_buffered_safetensors(weights, dtype, device).map_err(err)?;
    Llama::load_with_head_dim(vb, &cfg, head_dim).map_err(err)
}

impl Model {
    pub fn from_bytes(config_json: &[u8], tokenizer_json: &[u8], weights: Vec<u8>) -> Result<Self, String> {
        Self::from_bytes_on(config_json, tokenizer_json, weights, Device::Cpu)
    }

    /// As `from_bytes`, on `device` — a GPU on the host (`crate::device`); the page is always the CPU.
    /// An attached target and its bridge run on the same device, unless `place_target_on` moves them.
    pub fn from_bytes_on(config_json: &[u8], tokenizer_json: &[u8], weights: Vec<u8>, device: Device) -> Result<Self, String> {
        Self::from_bytes_in(config_json, tokenizer_json, weights, device, DType::F32)
    }

    /// As `from_bytes_on`, its weights held in `dtype` (`load_decoder_in`).
    pub fn from_bytes_in(config_json: &[u8], tokenizer_json: &[u8], weights: Vec<u8>, device: Device, dtype: DType) -> Result<Self, String> {
        Self::from_bytes_q(config_json, tokenizer_json, weights, device, dtype, None)
    }

    /// A FAST_LLM file (`crate::fast_llm`) as a bridged `Model`: its small model reads on `device`, its
    /// large model and bridge sit on `target_device` — the same model `from_bytes_in` then
    /// `attach_target_with` builds from the three files it was packed from.
    pub fn from_fast_llm(bytes: &[u8], device: Device, target_device: Device, source_dtype: DType, target_dtype: DType, quant: Option<candle_core::quantized::GgmlDType>) -> Result<Self, String> {
        Self::from_fast_llm_parts(crate::fast_llm::FastLlm::unpack(bytes)?, device, target_device, source_dtype, target_dtype, quant)
    }

    /// As `from_fast_llm`, from a file already split apart — so a caller short of memory (the page, in
    /// 4 GiB of wasm) can let the file's own bytes go before the models are built.
    pub fn from_fast_llm_parts(f: crate::fast_llm::FastLlm, device: Device, target_device: Device, source_dtype: DType, target_dtype: DType, quant: Option<candle_core::quantized::GgmlDType>) -> Result<Self, String> {
        if let Some((side, r)) = f.refs.iter().next() {
            return Err(format!("a THIN FAST_LLM: its {side} is {} @ {}, not yet fetched — resolve it first (fast_llm::hub_fetch)", r.repo, r.revision));
        }
        let (source_config, target_config) = (f.get("source.config.json")?.as_bytes().to_vec(), f.get("target.config.json")?.as_bytes().to_vec());
        let tokenizer = f.get("tokenizer.json")?.as_bytes().to_vec();
        let crate::fast_llm::FastLlm { source, target, bridge, quantized, .. } = f;
        let mut m = Self::from_bytes_in(&source_config, &tokenizer, source, device, source_dtype)?;
        m.place_target_on(target_device.clone());
        m.target_weights_in(target_dtype);
        let Some(q) = quantized else {
            m.target_weights_quantized(quant);
            m.attach_target_with(&target_config, target, Some(&bridge))?;
            return Ok(m);
        };
        // stored already quantized: the blocks become QTensors as they are, and are never quantized again
        let (ggml, ready) = q.qtensors()?;
        m.target_weights_quantized(Some(ggml));
        let decoder = load_decoder_q_ready(&target_config, target, &target_device, target_dtype, Some(ggml), Some(ready))?;
        m.attach_target_decoder(decoder, Some(&bridge))?;
        Ok(m)
    }

    /// As `from_bytes_in`, its linear layers quantized at load (`load_decoder_q`).
    pub fn from_bytes_q(config_json: &[u8], tokenizer_json: &[u8], weights: Vec<u8>, device: Device, dtype: DType, quant: Option<candle_core::quantized::GgmlDType>) -> Result<Self, String> {
        let llama = load_decoder_q(config_json, weights, &device, dtype, quant)?;
        Self::from_decoder(config_json, tokenizer_json, llama, device)
    }

    /// As `from_bytes_q`, the decoder already built on `device` — a target read from a pre-quantized
    /// FAST_LLM (`fast_llm::target_decoder`).
    pub fn from_decoder(config_json: &[u8], tokenizer_json: &[u8], llama: Decoder, device: Device) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(err)?;
        let stop = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "<|end_of_text|>", "<turn|>"].iter().filter_map(|t| tokenizer.token_to_id(t)).collect();
        let template = Template::from_config(config_json)?;
        Ok(Self {
            template,
            system: None,
            kv: llama.empty_state(),
            llama,
            target: None,
            map: None,
            bridged: false,
            bridge_tokens: 1,
            tokenizer,
            target_device: device.clone(),
            target_dtype: DType::F32,
            target_quant: None,
            device,
            stop,
            // Greedy: the same question gets the same answer, which is what makes a run comparable.
            sampler: LogitsProcessor::new(0, None, None),
            pending: Vec::new(),
            pos: 0,
            generated: Vec::new(),
            shown: String::new(),
            counters: Counters::default(),
        })
    }

    /// Attach the model that ANSWERS when the bridge is on, from this model's reading. It shares
    /// this model's tokenizer, and direct copy requires identical attention geometry.
    pub fn attach_target(&mut self, config_json: &[u8], weights: Vec<u8>) -> Result<(), String> {
        self.attach_target_with(config_json, weights, None)
    }

    /// Put the target and its bridge on `device` when they are attached — a second GPU, where the
    /// two models do not fit on one. The state crosses once, when the reading is handed over.
    pub fn place_target_on(&mut self, device: Device) {
        self.target_device = device;
    }

    /// Hold the target's weights in `dtype` when it is attached — half precision, where f32 would not
    /// fit its card.
    pub fn target_weights_in(&mut self, dtype: DType) {
        self.target_dtype = dtype;
    }

    /// Hold the target's linear layers quantized when it is attached (`load_decoder_q`) — a target too
    /// large for its card even in bf16.
    pub fn target_weights_quantized(&mut self, quant: Option<candle_core::quantized::GgmlDType>) {
        self.target_quant = quant;
    }

    /// Attach the target, with a fitted bridge when the two models' shapes differ. Without one,
    /// direct copy requires identical geometry.
    pub fn attach_target_with(&mut self, config_json: &[u8], weights: Vec<u8>, bridge: Option<&[u8]>) -> Result<(), String> {
        let target = load_decoder_q(config_json, weights, &self.target_device, self.target_dtype, self.target_quant)?;
        self.attach_target_decoder(target, bridge)
    }

    /// As `attach_target_with`, the target already built (on `place_target_on`'s device).
    pub fn attach_target_decoder(&mut self, target: Decoder, bridge: Option<&[u8]>) -> Result<(), String> {
        let map = match bridge {
            Some(bytes) => Some(self.fitted_map(&target, bytes)?),
            None if target.geometry() != self.llama.geometry() => {
                return Err(format!(
                    "the target's geometry {:?} does not match the source's {:?} — direct copy needs them equal, or a fitted bridge",
                    target.geometry(),
                    self.llama.geometry()
                ));
            }
            None => None,
        };
        self.bridge_tokens = match &map {
            Some(m) if m.ck.is_empty() => TRANSLATED_BRIDGE_TOKENS,
            _ => 1,
        };
        self.target = Some(target);
        self.map = map;
        Ok(())
    }

    /// Answer through another fit of the same pair, the attached target kept as it is — so one load of
    /// a large target serves every bridge file. A bridge that does not fit the target is refused and
    /// the one in place stays. Resets the bridge tokens as attaching does.
    pub fn swap_bridge(&mut self, bridge: &[u8]) -> Result<(), String> {
        let target = self.target.as_ref().ok_or("no target attached — attach one before swapping its bridge")?;
        let map = self.fitted_map(target, bridge)?;
        self.bridge_tokens = if map.ck.is_empty() { TRANSLATED_BRIDGE_TOKENS } else { 1 };
        self.map = Some(map);
        Ok(())
    }

    /// A bridge file's map on the target's device, refused unless it writes `target`'s own shapes.
    fn fitted_map(&self, target: &Decoder, bytes: &[u8]) -> Result<KvMap, String> {
        let map = crate::bridge::read(bytes).and_then(|m| m.to_device(&self.target_device)).map_err(err)?;
        let g = target.geometry();
        if map.layer_of.len() != g.layers || (0..g.layers).any(|l| map.writes_kv(l) && map.shape(l) != target.layer_shape(l)) {
            return Err(format!("the bridge maps into {} layers × {} heads × {}, the target is {g:?}", map.layer_of.len(), map.tgt_heads, map.dim));
        }
        Ok(map)
    }

    /// How many of the prompt's last tokens the answering model reads itself when bridged.
    pub fn set_bridge_tokens(&mut self, k: usize) {
        self.bridge_tokens = k.max(1);
    }

    pub fn has_target(&self) -> bool {
        self.target.is_some()
    }

    pub fn counters(&self) -> Counters {
        self.counters
    }

    /// Begin a fresh reply to one question. With `bridge`, this model reads the history here and
    /// the target answers from the first `step`; without, this model answers.
    pub fn start(&mut self, question: &str, bridge: bool) -> Result<(), String> {
        self.start_arm(question, if bridge { Arm::Bridge } else { Arm::Native })
    }

    /// Begin one ARM of a comparison. Every arm ends its prompt on the same bridge token, so with
    /// transported state the reply can only follow the memory.
    ///
    /// ONE SCHEDULE FOR EVERY MEASURED ARM (docs/BRIDGE.md §8): `Target`, `Source` and `Bridge` all
    /// prefill the history HERE and process the last token in the first `step`. Measured in the page:
    /// with the native prefill in `step` instead, one session in four timed it 2.4x slower while the
    /// bridge's prefill here was not — the arms were being timed through different compiled paths.
    /// `Native` keeps the one-shot prefill, so `Restore` still compares two schedules.
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
        match memory {
            None => {
                self.kv = self.llama.empty_state();
                self.pending = ids;
                self.pos = 0;
                self.bridged = false;
                self.counters = Counters { prompt_tokens: n, reader_tokens: 0, generator_history: n, bridge_tokens: 0 };
            }
            Some(history) => {
                // Who reads the history, and who answers from it.
                let (reads_target, answers_target, own) = match arm {
                    Arm::Target => (true, true, true),
                    Arm::Source => (false, false, true),
                    Arm::Restore => (false, false, false),
                    _ => (false, true, false),
                };
                if (reads_target || answers_target) && self.target.is_none() {
                    return Err("this arm needs a target attached — turn the bridge on".into());
                }
                let reader = match (&self.target, reads_target) {
                    (Some(t), true) => t,
                    _ => &self.llama,
                };
                let mut state = reader.empty_state();
                let on = if reads_target { &self.target_device } else { &self.device };
                let t = Tensor::new(history.as_slice(), on).and_then(|t| t.unsqueeze(0)).map_err(err)?;
                // Crossing into the target through a fitted bridge: capture the source's residual
                // streams as it reads, and translate them. Otherwise the state is copied as it is.
                let translate = answers_target && !reads_target && self.map.is_some();
                let mut hidden = Vec::new();
                reader.forward_capture(&t, 0, &mut state, translate.then_some(&mut hidden)).map_err(err)?;
                // Handed to the target: only what it reads crosses to its device, once — the residual
                // streams when translating (plus the source's own slots for recurrent layers), the
                // state itself when copying.
                if translate {
                    let (map, target) = (self.map.as_ref().unwrap(), self.target.as_ref().unwrap());
                    // only the streams the map reads cross; the rest stay where they are, unread
                    let read = map.streams_read(hidden.len());
                    let hidden = hidden.iter().zip(read).map(|(h, r)| if r { crate::device::moved(h, &self.target_device) } else { Ok(h.clone()) }).collect::<candle_core::Result<Vec<_>>>().map_err(err)?;
                    // a map's extension may read the prompt's token ids
                    let ids = match map.extension {
                        Some(_) => Some(Tensor::new(history.as_slice(), &self.target_device).and_then(|t| t.unsqueeze(0)).map_err(err)?),
                        None => None,
                    };
                    state = fill_from_source(translate_hidden_with(map, &hidden, ids.as_ref(), target).map_err(err)?, &state).map_err(err)?;
                }
                if answers_target && !reads_target {
                    state = kv_moved(&state, &self.target_device).map_err(err)?;
                }
                self.kv = state;
                self.pos = history.len();
                // The answering model reads the rest of the prompt itself: one token on the measurement
                // schedule, `bridge_tokens` when bridged.
                let tail = if matches!(arm, Arm::Bridge | Arm::WrongDonor(_)) { self.bridge_tokens.min(n - 1) } else { 1 };
                self.pending = ids[n - tail..].to_vec();
                self.bridged = answers_target;
                self.counters = if own {
                    Counters { prompt_tokens: n, reader_tokens: 0, generator_history: n, bridge_tokens: 0 }
                } else {
                    Counters { prompt_tokens: n, reader_tokens: history.len(), generator_history: 0, bridge_tokens: tail }
                };
            }
        }
        Ok(())
    }

    /// How many tokens `question` becomes once wrapped in the chat template.
    pub fn prompt_tokens(&self, question: &str) -> Result<usize, String> {
        Ok(self.encode(question)?.len())
    }

    /// A deterministic synthetic incident log of at least `tokens` prompt tokens, ending in a
    /// question — variant `v` differs in every line, so eight variants are eight distinct prompts.
    pub fn synthetic_prompt(&self, tokens: usize, v: u32) -> Result<String, String> {
        let mut text = String::new();
        let mut i = 0u32;
        while self.prompt_tokens(&format!("{text}{}", crate::measure::SYNTHETIC_QUESTION))? < tokens {
            text.push_str(&crate::measure::incident_line(v, i));
            i += 1;
        }
        Ok(format!("{text}{}", crate::measure::SYNTHETIC_QUESTION))
    }

    fn encode(&self, question: &str) -> Result<Vec<u32>, String> {
        let prompt = match &self.system {
            Some(system) => self.template.prompt_with_system(system, question),
            None => self.template.prompt(question),
        };
        Ok(self.tokenizer.encode(prompt, false).map_err(err)?.get_ids().to_vec())
    }

    /// A system turn for every prompt that follows (`None` for the family's default). Bridged, the
    /// reading model reads it with the rest of the prompt and the answering model reads none of it.
    pub fn set_system(&mut self, system: Option<String>) {
        self.system = system;
    }

    /// Run the model once — the pending prompt or bridge token on the first call, one token after —
    /// and return the text that token added.
    pub fn step(&mut self) -> Result<Step, String> {
        if self.pending.is_empty() || self.generated.len() >= MAX_NEW_TOKENS {
            return Ok(Step::Done);
        }
        let on = if self.bridged && self.target.is_some() { &self.target_device } else { &self.device };
        let input = Tensor::new(self.pending.as_slice(), on).and_then(|t| t.unsqueeze(0)).map_err(err)?;
        let model = match (&self.target, self.bridged) {
            (Some(target), true) => target,
            _ => &self.llama,
        };
        let logits = model.forward(&input, self.pos, &mut self.kv).and_then(|l| l.squeeze(0)).map_err(err)?;
        // The same penalty in every arm, over the reply only, so a comparison stays like for like.
        let recent = &self.generated[self.generated.len().saturating_sub(REPEAT_WINDOW)..];
        let logits = candle_transformers::utils::apply_repeat_penalty(&logits, REPEAT_PENALTY, recent).map_err(err)?;
        self.pos += self.pending.len();
        let next = self.sampler.sample(&logits).map_err(err)?;
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
