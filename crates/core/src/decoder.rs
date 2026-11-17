//! The decoders a bridge can read from and install into: Llama-family (SmolLM2, Qwen3, Llama 3) in
//! `llama.rs`, Gemma 3 in `gemma3.rs`, Qwen3.5's hybrid in `qwen3_5.rs`, Granite 4.0's in `granite.rs`, LFM2's in `lfm2.rs`. `Model` holds one per side;
//! `model::load_decoder` picks it from the config. Each keeps its persistent state OPEN (`Kv`) and
//! offers the same interface.

use candle_core::{Result, Tensor};

use crate::gemma3::Gemma3;
use crate::gemma4::Gemma4;
use crate::granite::Granite;
use crate::lfm2::Lfm2;
use crate::qwen3_5::Qwen35;
use crate::llama::{Geometry, Kv, Llama};

/// Rotating one layer's keys with THAT layer's RoPE — Gemma 3 rotates local and global layers with
/// different bases, so the layer is part of the question. What `bridge::translate_hidden` needs.
pub trait LayerKeys {
    fn rotate_layer_keys(&self, layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor>;

    /// A recurrent layer's state built by the model itself from that layer's normed INPUTS `[1, n,
    /// hidden]` — what a bridge does where the state sums over every token and no per-token map reaches
    /// it (Granite's Mamba-2 scan). A family without such layers refuses.
    fn state_from_inputs(&self, layer: usize, _inputs: &Tensor) -> Result<(Tensor, Tensor)> {
        candle_core::bail!("layer {layer}: this family builds no recurrent state from its inputs")
    }

    /// The model's own embeddings of `ids` `[1, n]` → `[1, n, hidden]`. A family that does not hand
    /// them out refuses.
    fn embed_tokens(&self, _ids: &Tensor) -> Result<Tensor> {
        candle_core::bail!("this family does not hand out its token embeddings")
    }
}

impl LayerKeys for Llama {
    fn rotate_layer_keys(&self, _layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        self.rotate_keys(k, pos, inverse)
    }
    fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        Llama::embed_tokens(self, ids)
    }
}

impl LayerKeys for Qwen35 {
    fn rotate_layer_keys(&self, _layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        self.rotate_keys(k, pos, inverse)
    }
}

impl LayerKeys for Granite {
    fn rotate_layer_keys(&self, _layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        self.rotate_keys(k, pos, inverse)
    }
    fn state_from_inputs(&self, layer: usize, inputs: &Tensor) -> Result<(Tensor, Tensor)> {
        Granite::state_from_inputs(self, layer, inputs)
    }
}

impl LayerKeys for Lfm2 {
    fn rotate_layer_keys(&self, _layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        self.rotate_keys(k, pos, inverse)
    }
}

impl LayerKeys for Gemma4 {
    fn rotate_layer_keys(&self, layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        self.rotate_keys(layer, k, pos, inverse)
    }
}

impl LayerKeys for Gemma3 {
    fn rotate_layer_keys(&self, layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        self.rotate_keys(layer, k, pos, inverse)
    }
}

pub enum Decoder {
    Llama(Llama),
    Gemma3(Gemma3),
    /// Gemma 4: global layers' heads differ from the sliding layers' (`layer_shape`); a KV-shared
    /// layer's slot in `Kv` is empty.
    Gemma4(Gemma4),
    /// Qwen3.5's hybrid: its GDN layers' slots in `Kv` carry (C, R), not (K, V).
    Qwen35(Qwen35),
    /// Granite 4.0's hybrid: its Mamba-2 layers' slots in `Kv` carry (C, S), not (K, V).
    Granite(Granite),
    /// LFM2's hybrid: its short-conv layers' slots in `Kv` carry (C, ∅), not (K, V).
    Lfm2(Lfm2),
}

macro_rules! each {
    ($self:ident, $d:ident => $e:expr) => {
        match $self {
            Decoder::Llama($d) => $e,
            Decoder::Gemma3($d) => $e,
            Decoder::Gemma4($d) => $e,
            Decoder::Qwen35($d) => $e,
            Decoder::Granite($d) => $e,
            Decoder::Lfm2($d) => $e,
        }
    };
}

impl Decoder {
    pub fn empty_state(&self) -> Kv {
        each!(self, d => d.empty_state())
    }
    pub fn geometry(&self) -> Geometry {
        each!(self, d => d.geometry())
    }
    pub fn hidden_size(&self) -> usize {
        each!(self, d => d.hidden_size())
    }
    /// Layer `layer`'s attention shape (kv heads, head width) — `geometry`'s for every family whose
    /// layers all share one; Gemma 4's global layers have their own.
    pub fn layer_shape(&self, layer: usize) -> (usize, usize) {
        match self {
            Decoder::Gemma4(d) => d.layer_shape(layer),
            _ => {
                let g = self.geometry();
                let _ = layer;
                (g.kv_heads, g.head_dim)
            }
        }
    }
    /// Gradients through the decoder, for the correction: norms, softmax and RoPE from basic ops.
    pub fn set_differentiable(&mut self, on: bool) {
        each!(self, d => d.set_differentiable(on))
    }

    /// Per layer, whether its state is recurrent (C, R) — copied across a matched pair, never mapped.
    pub fn recurrent_layers(&self) -> Vec<bool> {
        match self {
            Decoder::Qwen35(d) => d.config().attention.iter().map(|a| !a).collect(),
            Decoder::Granite(d) => d.config().attention.iter().map(|a| !a).collect(),
            Decoder::Lfm2(d) => d.config().attention.iter().map(|a| !a).collect(),
            _ => vec![false; self.geometry().layers],
        }
    }
    /// A short-conv layer's B·x rows from the residual stream entering it — what a bridge fits a conv
    /// map to; None for every other layer (and every other family, whose recurrent state is copied).
    pub fn conv_rows(&self, layer: usize, residual: &Tensor) -> Result<Option<Tensor>> {
        match self {
            Decoder::Lfm2(d) => d.conv_rows(layer, residual),
            _ => Ok(None),
        }
    }

    /// The rows a bridge fits for recurrent layer `layer`, from the residual stream entering it: a short
    /// conv's B·x (LFM2) or a Mamba-2 layer's normed input (Granite). None for any other layer.
    pub fn fitted_rows(&self, layer: usize, residual: &Tensor) -> Result<Option<Tensor>> {
        match self {
            Decoder::Lfm2(d) => d.conv_rows(layer, residual),
            Decoder::Granite(d) => d.layer_input(layer, residual),
            _ => Ok(None),
        }
    }

    /// Whether `fitted_rows` are the layer's inputs, from which the model builds its own state.
    pub fn rows_are_inputs(&self) -> bool {
        matches!(self, Decoder::Granite(_))
    }

    /// How many rows a short-conv layer's state holds; 0 for a family with none.
    pub fn conv_state_rows(&self) -> usize {
        match self {
            Decoder::Lfm2(d) => d.conv_state_rows(),
            _ => 0,
        }
    }

    pub fn forward(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        each!(self, d => d.forward(ids, pos, kv))
    }
    pub fn forward_all(&self, ids: &Tensor, pos: usize, kv: &mut Kv) -> Result<Tensor> {
        each!(self, d => d.forward_all(ids, pos, kv))
    }
    pub fn forward_capture(&self, ids: &Tensor, pos: usize, kv: &mut Kv, capture: Option<&mut Vec<Tensor>>) -> Result<Tensor> {
        each!(self, d => d.forward_capture(ids, pos, kv, capture))
    }
}

impl LayerKeys for Decoder {
    fn rotate_layer_keys(&self, layer: usize, k: &Tensor, pos: usize, inverse: bool) -> Result<Tensor> {
        each!(self, d => d.rotate_layer_keys(layer, k, pos, inverse))
    }
    fn state_from_inputs(&self, layer: usize, inputs: &Tensor) -> Result<(Tensor, Tensor)> {
        each!(self, d => d.state_from_inputs(layer, inputs))
    }
    fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        each!(self, d => LayerKeys::embed_tokens(d, ids))
    }
}
