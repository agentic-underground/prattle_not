//! The arithmetic that runs in the page and on the server alike: no DOM, no socket, no GPU API.

pub mod align;
pub mod attend;
pub mod bridge;
pub mod decoder;
pub mod device;
pub mod fast_llm;
pub mod gdn;
pub mod gemma3;
pub mod gemma4;
pub mod granite;
pub mod lfm2;
pub mod half;
pub mod llama;
pub mod measure;
pub mod observe;
pub mod qwen35;
pub mod qwen3_5;
pub mod speedup;
pub mod translator;
pub mod weights;
pub mod model;
pub mod text;

pub use model::{Arm, Model, Role, Step, TRANSLATED_BRIDGE_TOKENS, Turns};
