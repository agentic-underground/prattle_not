[prattle_not](../README.md) / [documents](README.md) / **EXECUTIVE SUMMARY**

# LatentPort, reproduced on two RTX 3090s — executive summary

**LatentPort is Simon Villani's work** ([latentport.com](https://latentport.com/),
[`LatentPort/latentport`](https://github.com/LatentPort/latentport), Apache-2.0; paper
arXiv:2609.25053v1). Characters and story by Simon Villani. This is an independent reproduction,
downstream of it.

## The claim

A small model reads a long prompt; a learned bridge translates its internal state — attention
memory, recurrent memory and its final hidden state — into a large model; the large model answers
**without reading the prompt itself**. Upstream reports, for Qwen3.5-4B → Qwen3.8-27B on one RTX 5090,
a speed-up to the first token that grows with prompt length — **2.19× at ~16K tokens** — while
answer quality stays well above the small model's and below the large model's.

## What was reproduced

Upstream's published weights and bridge, upstream's own timing code, upstream's experimental design
(8 prompts × 3 paired repetitions per length) and statistic (geometric mean, 95% interval) — on two
RTX 3090s.

| prompt length | large model reads it | small model reads it, bridge, large model answers | **speed-up (95% CI)** | upstream, RTX 5090 |
|---|---|---|---|---|
| 128 | 0.146 s | 0.108 s | **1.36×** (1.33–1.39) | 1.06× (0.87–1.29) |
| ~2K | 1.585 s | 0.540 s | **2.94×** (2.92–2.95) | 1.65× (1.61–1.69) |
| ~8K | 6.528 s | 1.983 s | **3.29×** (3.27–3.31) | 2.01× (1.96–2.05) |
| ~16K | 13.759 s | 4.049 s | **3.39×** (3.38–3.40) | 2.19× (2.15–2.23) |

In all 192 timed runs the large model read **zero** prompt tokens when bridged. Record:
[`results/e120-rtx3090x2/`](../results/e120-rtx3090x2/README.md).

**And what the speed costs.** Scoring the large model's own reply token by token, the bridged large
model keeps **84–91%** of the way from "knows nothing" to "read the prompt itself" (NCR 0.914 /
0.838 / 0.840 / 0.846 across the four lengths); the small model alone keeps 68–82%. The intervals
separate at every length. Record: [`results/fidelity-rtx3090x2/`](../results/fidelity-rtx3090x2/README.md).

## And carried to models LatentPort never used — in a browser

The paper's own handoff — closed-form maps between the models' states, its learned correction, and
**one** token read by the receiver — ported to SmolLM2-135M → SmolLM2-360M, a pair whose state shapes
do not match (30 vs 32 layers, 3 vs 5 attention heads). On questions it was never fitted on, the 360M
answering from the 135M's translated memory gets **6 of 7 right — as many as reading everything** —
and keeps 93% of the way to its own behaviour (NCR 0.930; the paper's own corrected pair: 0.918). It
runs entirely in a browser, **1.87× faster to the first token**:
[agentic-underground.github.io/prattle_not](https://agentic-underground.github.io/prattle_not/).

## And across seven families, on two RTX 3090s

The same handoff — the paper's maps and correction, one bridge token — carries
between pairs LatentPort never used, from plain attention to three kinds of recurrent layer. **A
bridge passes when the large model answering from it gets as many of seven held-out questions right
as the large model reading the prompt itself.** NCR is held-out fidelity, as above; the speed-up is to
the first token at ~256 tokens, beside the ceiling (the large model's time over the small one's).

| bridge | layers that carry state | NCR | answers (large alone) | speed-up (ceiling) |
|---|---|---|---|---|
| Qwen3-0.6B → 1.7B | attention | 0.805 | 7 (7) | 1.47× |
| SmolLM2-360M → 1.7B | attention | 0.845 | 7 (7) | 1.67× |
| Llama 3.2 1B → 3B | attention | 0.965 | 6 (6) | 1.70× (2.54×), one card |
| Qwen3-4B → 8B | attention | 0.890 | 7 (7) | 1.12× (1.61×), two cards |
| Qwen3.5-0.8B → 2B | attention + Gated DeltaNet | 0.917 | 7 (7) | 1.36× (1.50×), one card |
| **Qwen3.5-4B → 9B Base — the paper's own pair** | attention + Gated DeltaNet | **0.989** | 7 (7) | 1.16× (1.38×), two cards |
| LFM2-350M → 1.2B | attention + short convolution | 0.978 | 7 (7) | 1.88× (2.50×), one card |
| Qwen3-0.6B → 8B | attention | 0.803 | 7 (7) | 2.82× (5.97×), one card |
| Granite 4.0 h-1b → h-tiny | attention + Mamba-2 | 0.924 | 7 (7) | ~2.0× (~3.9×), one card |
| **Qwen3-0.6B → 32B, the 32B at 4-bit** | attention | 0.866 | 6 (7) | **4.20×** (11.5×), **7.16× at ~1K tokens**, two cards |

Every one is on Hugging Face under
[`agentic-underground`](https://huggingface.co/agentic-underground), credited to LatentPort and
Simon Villani; the 0.6B → 32B super-bridge is published with its one miss stated (the plants
question's framing). Two pairs fall short of the bar: **Gemma 3 270M → 1B** (1.91× sooner, NCR 0.870, but
6 of 7 where the 1B alone gets 7) is on the hub with a card that says it brings no answer gain, and
**Qwen3-1.7B → 4B** (6 of 7 against 7) is held, unpublished.

What decides the speed-up is the gap between the two models' prompt reading. Where the large model
reads barely slower than the small one (Qwen3.5-4B → 9B), there is little to save; where the gap is wide (LFM2, Llama 3.2), the bridge
approaches its ceiling.

## What this means

- **The direction of the claim holds.** On different hardware, the bridge makes the first token
  arrive sooner, and more so the longer the prompt — the shape upstream published.
- **The size of the speed-up depends on the hardware.** Ours is larger at every length. A plausible
  reading — not yet measured — is that the large model's prompt reading is relatively slower on a
  3090 than on a 5090, which leaves more for the bridge to save.
- **The bridged answer stays measurably closer to the large model than the small model does**, so
  the speed is not bought by simply answering with the small model.

## What it does not show

- **Not upstream's exact run.** Upstream's E120 prompts are unpublished; ours are 8 published
  synthetic incident logs. Upstream used NF4 weights on a 5090; we used the published Q4_K_M GGUFs.
- **Not one GPU.** Upstream's runtime targets a single 32 GB card; on 24 GB cards it ran only at
  ~2K (3.14×). 8K and 16K used a small, disclosed patch splitting the models across two cards,
  with the state crossing PCIe inside the timed path (~45 ms at 2K).
- **Not answer correctness.** Fidelity is closeness to the large model's own reply, not graded
  answers; upstream's graded task panels are unpublished. Our wrong-prompt control is weak, because
  all prompts share one question.
- **The 4B → 27B numbers are not yet from Rust.** They come from upstream's runtime, built here from
  its pinned source; the Rust port tracks it (the whole 4B and the whole 27B within ~5%, which is the oracle's own
  rounding spread), reproduces its translator to 10⁻⁴ and its aligner exactly, and the whole handoff in
  Rust gives upstream's first token. A fast GPU runtime for it is the work ahead. The
  in-browser proof is a small pair on the CPU; a larger pair on WebGPU is ahead.

## Run it yourself

The step-by-step guide, with the output each step should print, is
[`REPRODUCTION.md`](REPRODUCTION.md). Every number above is regenerated by it; the records in
[`results/`](../results/) list every run.
