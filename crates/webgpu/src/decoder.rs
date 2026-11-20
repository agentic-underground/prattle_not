//! The decoders this engine runs, behind one set of methods: the Llama family (SmolLM2, Qwen3),
//! Qwen3.5's hybrid and LFM2's. Which one a checkpoint is comes from its config.json.

use crate::bridge::{Carry, Kind};
use crate::lfm2::{self, Lfm2};
use crate::llama::Llama;
use crate::qwen35::{self, Qwen35};
use crate::{Gpu, GpuTensor, Rule, Weights};

pub enum Decoder {
    Llama(Llama),
    Qwen35(Qwen35),
    Lfm2(Lfm2),
}

/// Which tensors of a file this config's decoder keeps, and as what.
pub fn rule(config: &[u8]) -> Rule {
    if qwen35::is(config) {
        qwen35::rule
    } else if lfm2::is(config) {
        lfm2::rule
    } else if crate::llama::is_qwen3(config) {
        crate::llama::rule_half
    } else {
        crate::llama::rule
    }
}

macro_rules! each {
    ($self:expr, $d:ident => $e:expr) => {
        match $self {
            Decoder::Llama($d) => $e,
            Decoder::Qwen35($d) => $e,
            Decoder::Lfm2($d) => $e,
        }
    };
}

impl Decoder {
    /// The decoder for `config`, from weights streamed by `rule(config)`.
    pub fn build(gpu: &Gpu, config: &[u8], weights: Weights, max_ctx: usize) -> Result<Decoder, String> {
        if qwen35::is(config) {
            Ok(Decoder::Qwen35(Qwen35::from_weights(gpu, config, weights, max_ctx)?))
        } else if lfm2::is(config) {
            Ok(Decoder::Lfm2(Lfm2::from_weights(gpu, config, weights, max_ctx)?))
        } else {
            Ok(Decoder::Llama(Llama::from_weights(gpu, config, weights, max_ctx)?))
        }
    }

    pub async fn forward(&mut self, gpu: &Gpu, ids: &[u32]) -> Vec<f32> {
        each!(self, d => d.forward(gpu, ids).await)
    }

    pub fn prefill_capture(&mut self, gpu: &Gpu, ids: &[u32]) -> Vec<GpuTensor> {
        each!(self, d => d.prefill_capture(gpu, ids))
    }

    pub fn prefill(&mut self, gpu: &Gpu, ids: &[u32]) {
        each!(self, d => d.prefill(gpu, ids))
    }

    /// Install a bridged history of `n` positions: what the bridge translated for each layer, and —
    /// where it translated nothing — the source's own state.
    pub fn install(&mut self, gpu: &Gpu, carries: &[Carry], n: usize, source: &Decoder) -> Result<(), String> {
        match (self, source) {
            (Decoder::Llama(d), _) => {
                let kvs = carries
                    .iter()
                    .enumerate()
                    .map(|(l, c)| match c {
                        Carry::Kv(k, v) => Ok((k.clone(), v.clone())),
                        _ => Err(format!("layer {l} has no translated keys")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                d.install(gpu, &kvs, n);
                Ok(())
            }
            (Decoder::Lfm2(d), _) => d.install(gpu, carries, n),
            (Decoder::Qwen35(d), Decoder::Qwen35(s)) => d.install(gpu, carries, n, s),
            (Decoder::Qwen35(_), _) => Err("a Qwen3.5 target copies its recurrent state from a Qwen3.5 source".into()),
        }
    }

    /// Rotate keys of an attention layer (n × kv width) in place to positions pos..pos+n.
    pub fn rotate(&self, gpu: &Gpu, k: &GpuTensor, pos: usize) {
        each!(self, d => d.rotate(gpu, k, pos))
    }

    pub fn layer_count(&self) -> usize {
        each!(self, d => d.layer_count())
    }

    pub fn kv_width(&self) -> usize {
        each!(self, d => d.kv_width())
    }

    /// What layer `l` takes from a bridge.
    pub fn kind(&self, l: usize) -> Kind {
        match self {
            Decoder::Llama(_) => Kind::Kv,
            Decoder::Qwen35(d) => d.kind(l),
            Decoder::Lfm2(d) => d.kind(l),
        }
    }

    /// How many B·x rows a short-conv layer's state holds; 0 for a decoder with none.
    pub fn conv_rows(&self) -> usize {
        match self {
            Decoder::Lfm2(d) => d.conv_rows(),
            _ => 0,
        }
    }

    pub fn reset(&mut self, gpu: &Gpu) {
        match self {
            Decoder::Llama(d) => d.reset(),
            Decoder::Qwen35(d) => d.reset(gpu),
            Decoder::Lfm2(d) => d.reset(gpu),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::Store;

    #[test]
    fn qwen3_and_qwen35_matrices_load_at_half_and_smollm2_stays_at_f32() {
        let matrix = &[2048, 1024][..];
        let qwen3 = br#"{"model_type": "qwen3"}"#;
        let qwen35 = br#"{"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}"#;
        let smollm2 = br#"{"model_type": "llama"}"#;
        assert_eq!(rule(qwen3)("model.layers.0.mlp.up_proj.weight", matrix), Some(Store::Half));
        assert_eq!(rule(qwen3)("model.layers.0.self_attn.q_norm.weight", &[128]), Some(Store::Full));
        assert_eq!(rule(qwen35)("model.language_model.layers.0.mlp.up_proj.weight", matrix), Some(Store::Half));
        assert_eq!(rule(smollm2)("model.layers.0.mlp.up_proj.weight", matrix), Some(Store::Full));
    }
}
