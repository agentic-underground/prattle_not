[prattle_not](../README.md) / [documents](README.md) / **REPRODUCTION**

# Reproducing every number — the deep-dive

From a clean Linux machine to each published table and a published bridge's speed-up, with the output
each step should print, what each check catches, and what to do when yours differs. Read
[`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) for what the numbers mean. LatentPort is **Simon
Villani's** work ([`LatentPort/latentport`](https://github.com/LatentPort/latentport), Apache-2.0).

**Times marked *measured* are from the published run** (two RTX 3090s, 32 CPU threads); others are
not timed and depend on your machine.

---

## 0 · What you need

| | |
|---|---|
| GPUs | **two NVIDIA GPUs with 24 GB each** for the 8K and 16K rows; one 24 GB GPU reaches ~2K (§3.3) |
| Driver | one that supports **CUDA 13** (the published run: 615.71) — the toolkit itself is installed in user space by step 1 |
| Disk | **about 35 GB** *measured*: weights 21 GB, CUDA toolkit 7.1 GB, upstream's build and Python 5.1 GB; plus 1.3 GB for the browser pair |
| Tools | [Rust](https://rustup.rs), [`just`](https://github.com/casey/just), [`cargo-nextest`](https://nexte.st) (`cargo install cargo-nextest`), [`uv`](https://docs.astral.sh/uv/), `git`, `cmake`, a C/C++ compiler, `curl`, Python 3 |
| Network | GitHub, Hugging Face, NVIDIA's download site, and a Python package index |

Clone [`agentic-underground/prattle_not`](https://github.com/agentic-underground/prattle_not) and run
everything from its root. `just --list` shows every recipe.

## 1 · The toolchain and upstream's runtime

    just cuda

Downloads NVIDIA's CUDA 13.0.2 runfile (4.3 GB) and installs **the toolkit only** to
`~/.cache/prattle/cuda-13.0.2` — no sudo, no driver change. Ends with
`Build cuda_13.0.r13.0/compiler…`.

    just oracle

Fetches LatentPort at its pinned commit `6be1956f` and llama.cpp at `b19cbe92`, applies upstream's
own patch (`native/patch_llama.py`) and then ours (`scripts/patch_oracle_two_gpu.py` — inert unless
`PRATTLE_ORACLE_GPUS` is set), and builds for the compute capability `nvidia-smi` reports. **Expect:**

    KV/head permutation, recurrent transpose, convolution and speculative boundaries passed
    ✅ oracle built for sm_86: …/build/bin/libllama.so

The first line is **upstream's own layout test** — it catches a build whose state layouts are wrong
for your architecture. The patch script refuses with `Patch anchor changed:` if upstream's source
ever differs from the pin.

    just oracle-python
    just oracle-doctor

The first builds a `uv` venv with torch 2.11 and upstream's package at its pin; expect
`✅ torch 2.11.0+cu130 cuda 13.0`. The second loads the runtime from Rust; expect `oracle ABI 2`.
A failure here mentioning `libcublas.so.13` means the toolkit from `just cuda` is not being found —
run the recipes through `just`, which sets the paths.

## 2 · Upstream's weights

    just oracle-home

Downloads upstream's three GGUFs (22.1 GB) at the revision its installation guide pins, **verifies
each against Hugging Face's SHA-256**, and hard-links them into the cache upstream's Python reads, so
nothing is downloaded twice. Expect `✅ … verified` three times, the four pinned tokenizer and template
files verified, and three `linked as <sha256>` lines. A failed verification stops the recipe — a
truncated 17 GB file is the failure it catches.

A single-model check before the full runs, *measured* on one 3090:

    just oracle-native 16384

Expect `16416 prompt tokens, first token 760 after ~13.2 s (prefill counter +16416)` — the 27B
reading upstream's ~16K demo prompt itself. The prefill counter equalling the prompt length is the
proof the model read every token.

## 3 · The speed-up table

    just reproduce

*Measured* ~20 minutes. Writes 8 synthetic prompts, loads both models once, and times 8 prompts × 3
paired repetitions at 128, ~2K, ~8K and ~16K tokens through **upstream's own `run_arm`**. Prints, and
writes to `target/oracle/reproduce-*/table.md`:

| length | native median | composite median | speed-up (95% CI) | upstream E120 |
|---|---|---|---|---|
| 128 | 0.146 s | 0.108 s | 1.36× (1.33–1.39) | 1.06× (0.87–1.29) |
| 2048 | 1.585 s | 0.540 s | 2.94× (2.92–2.95) | 1.65× (1.61–1.69) |
| 8192 | 6.528 s | 1.983 s | 3.29× (3.27–3.31) | 2.01× (1.96–2.05) |
| 16384 | 13.759 s | 4.049 s | 3.39× (3.38–3.40) | 2.19× (2.15–2.23) |

### 3.1 What to expect on your machine

- **Two RTX 3090s:** the speed-ups within a few percent of these, and intervals as narrow.
- **Other GPUs:** different absolute times and ratios. What must hold is the **shape**: the speed-up
  rises with length. Upstream's own 5090 numbers are lower than ours at every length.
- **The table refuses to print** if any composite run's target read prompt tokens, or any native run
  used the source — `reproduce table` checks the counters of every run before computing anything.

### 3.2 The record

`target/oracle/reproduce-*/results.json` holds every run: arm, prompt, repetition, time, first token
and both counters. Compare it with [`results/e120-rtx3090x2/results.json`](../results/e120-rtx3090x2/results.json).

### 3.3 One 24 GB GPU

    just oracle-ttft 2048 3 ""

`GPUS=""` is upstream's single-GPU layout. *Measured* on one 3090 at ~2K: native 1.56 s, composite
0.50 s, median **3.14×** over 3 rounds. At 8K and 16K one 24 GB card runs out of memory — upstream
qualifies a 32 GB card.

## 4 · The fidelity table

    just fidelity

*Measured* ~10 minutes. Scores the native 27B's own 64-token reply teacher-forced under five
conditions and prints NCR per length. **Expect** the composite at **0.914 / 0.838 / 0.840 / 0.846**,
the 4B alone at 0.819 / 0.682 / 0.707 / 0.713, and the empty 27B at 0 by definition. Decoding is
greedy, so on the same GPUs and weights the values repeat; on other GPUs expect them within a few
hundredths. The **wrong donor** (0.59–0.64 here) must stay below the composite at every length — it
is the check that the state carries *this* prompt, not prompts in general.

## 5 · A published bridge's speed-up

The bridge the page loads, from
[agentic-underground/smollm2-135m-to-360m-bridge](https://huggingface.co/agentic-underground/smollm2-135m-to-360m-bridge)
at the commit the page pins, with no token:

    just models          # SmolLM2-135M-Instruct, -135M, -360M-Instruct
    just bridge-fetch    # the published bridge, into models/bridges/

**Expect** `✅ … the published bridge, sha256 534ece36…`. Then time it against the 360M reading the
prompt itself — the browser Measure's arms and token clock, 8 prompts × 3 paired repetitions:

    just speedup models/SmolLM2-135M-Instruct models/SmolLM2-360M-Instruct models/bridges/smollm2-135m-to-360m.safetensors

`PRATTLE_DEVICES=cpu` runs it on the CPU. Run it on a quiet machine: a timing beside other work is a
timing of contention. Every other bridge in the summary is on Hugging Face under
[`agentic-underground`](https://huggingface.co/agentic-underground), each card naming its pair and its
pinned models; `just speedup <source> <target> <bridge>` times any of them.

## 6 · The Rust ports

    just oracle-state
    cargo nextest run -p prattle-core --release --run-ignored only -E 'binary(gdn_oracle) | binary(qwen35_oracle)'

The first dumps the oracle's canonical 4B and 27B states after a fixed 50-token prompt. The tests
compare our Rust against them. **Expect:** layer 0 within 0.4% (4B) and 0.6% (27B); every later field
at ~4–7% (the 27B's worst 8%), and the final hidden state at 4.5% and 4.6% (relative Frobenius). That ~5% after layer 0 is the oracle's own 8-bit
activation rounding, amplified with depth — two configurations of the oracle disagree by 3.1% at the
final hidden state; the two configurations were measured against each other.

The translator, given upstream's own inputs:

    just oracle-translator
    cargo nextest run -p prattle-core --release --run-ignored only -E 'binary(translator_oracle)'

    cargo nextest run -p prattle-core --release --run-ignored only -E 'binary(align_oracle)'

**Expect** every test to pass: each translator stage within 10⁻⁵ of upstream's (ours: ≤ 6.5×10⁻⁷),
every translated 27B field within 10⁻³ (ours: ≤ 1.2×10⁻⁴), the aligner's 60 centers identical, and
the whole Rust handoff — Rust 4B, aligner, translator, the 27B's output head — giving upstream's
first token.

## 7 · When yours differs

| symptom | likely cause | what to do |
|---|---|---|
| `could not initialize full-GPU target context` or CUDA out of memory | one GPU, or 8K/16K on 24 GB | two GPUs with the default `GPUS="0,1"`, or stay at ≤ ~2K |
| `libcublas.so.13: cannot open shared object file` | the user-space toolkit is not on the path | run through `just`; check `just cuda` finished |
| `Patch anchor changed: …` | upstream's or llama.cpp's source is not at its pin | the recipes fetch by commit; a modified `~/.cache/prattle/oracle` needs deleting |
| `FileExistsError` from the demo | an output directory already exists | the recipes create fresh, timestamped ones |
| a much smaller speed-up | another GPU, or the machine busy | close other GPU work; compare the shape, not the number |
| `Composite processed prompt tokens on the target` | the bridge was not used | a real failure — do not publish a number from that run |
