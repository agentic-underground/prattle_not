# candle-core 0.11.0, patched

This is candle-core 0.11.0 from crates.io (MIT OR Apache-2.0, `LICENSE`), used through the workspace's
`[patch.crates-io]`. **One change**, in `src/backprop.rs`, each site marked `PATCHED`:

> The backward pass computes no gradient for an operand that tracks none — for `matmul` and the
> four arithmetic binary ops.

Upstream computes both operands' gradients unconditionally. Through a frozen model — the target a
bridge's correction trains through — that allocates a gradient the size of every weight matrix, and
nothing ever reads it. The correction through SmolLM2-1.7B ran out of a 24 GB GPU for that reason.

`crates/core/src/bridge.rs`'s `a_frozen_weight_gets_no_gradient` pins it. A gradient that a tracked
operand receives is unchanged, so every training result is the same; only the unused ones are gone.

**And one warning fix**, in `src/cpu/erf.rs`: a `use std::f64;` import made `f64::INFINITY` name the
deprecated module constant instead of the type's own; the import is gone, the values are the same.

Drop this directory and the `[patch.crates-io]` entry once candle skips untracked operands itself.
