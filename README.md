# prattle_not

**Language models that answer from another model's memory — in your browser, or on your own GPUs.**
A small model reads the prompt; a bridge carries what it read into a larger model; the larger model
answers without reading the prompt itself, and the first token arrives sooner.

**The idea, the method and the vision are [LatentPort](https://github.com/LatentPort/latentport)'s —
Simon Villani's work** ([latentport.com](https://latentport.com/), arXiv:2609.25053v1). Characters
and story by Simon Villani. Everything here is downstream of that work: a reproduction in Rust and
WebAssembly, carried to model families LatentPort never used. The name follows its predecessor,
*PrattleNot* — an anagram of *LatentPort*.

## See it in your browser

**[agentic-underground.github.io/prattle_not](https://agentic-underground.github.io/prattle_not/)** —
models running entirely in your browser. Turn the **bridge** on: SmolLM2-135M reads your question and
SmolLM2-360M, almost three times larger, answers from its translated memory, reading one token
itself — the counters show who read what. **Controls** show the answer changes with the memory that
crossed; **Measure** times it against the 360M reading the prompt itself.

## Run it yourself

Linux, [Rust](https://rustup.rs) and [`just`](https://github.com/casey/just) (and
[`wasm-pack`](https://rustwasm.github.io/wasm-pack/) for the page):

    just run      # the page and the server on 0.0.0.0:8192 — it prints its own address
    just test     # every test that needs no downloaded model

The server serves the page, **/compare** — one FAST_LLM file beside the three files it was packed
from — and **/chat**, which answers with LatentPort's own 4B → 27B pair through its runtime
(`just chat`, after the oracle steps in [`documents/REPRODUCTION.md`](documents/REPRODUCTION.md)). Measure a published bridge's speed-up to first token on your own machine:

    just models && just bridge-fetch
    just speedup models/SmolLM2-135M-Instruct models/SmolLM2-360M-Instruct models/bridges/smollm2-135m-to-360m.safetensors

## The results

LatentPort's own 4B → 27B pair, reproduced on two RTX 3090s with upstream's timing code, design and
statistic:

| prompt length | speed-up to first token (95% CI) | upstream, RTX 5090 | bridged fidelity (NCR) | 4B alone |
|---|---|---|---|---|
| 128 | **1.36×** (1.33–1.39) | 1.06× | 0.914 | 0.819 |
| ~2K | **2.94×** (2.92–2.95) | 1.65× | 0.838 | 0.682 |
| ~8K | **3.29×** (3.27–3.31) | 2.01× | 0.840 | 0.707 |
| ~16K | **3.39×** (3.38–3.40) | 2.19× | 0.846 | 0.713 |

And carried to other families, each bridge answering held-out questions as its large model does
reading the prompt itself — every one published on Hugging Face under
[`agentic-underground`](https://huggingface.co/agentic-underground):

| bridge | NCR | speed-up to first token (~256 tokens) |
|---|---|---|
| SmolLM2-135M → 360M (in the browser) | 0.930 | 1.87× |
| Qwen3-0.6B → 1.7B | 0.805 | 1.47× |
| SmolLM2-360M → 1.7B | 0.845 | 1.67× |
| Llama 3.2 1B → 3B | 0.965 | 1.70× |
| Qwen3-4B → 8B | 0.890 | 1.12× |
| Qwen3.5-0.8B → 2B (Gated DeltaNet) | 0.917 | 1.36× |
| **Qwen3.5-4B → 9B Base — LatentPort's own pair** | **0.989** | 1.16× |
| LFM2-350M → 1.2B (short convolution) | 0.978 | **1.88×** |

**Read [`documents/EXECUTIVE_SUMMARY.md`](documents/EXECUTIVE_SUMMARY.md) before quoting a number** —
it says what this does and does not show. Every run is in [`results/`](results/). How to reproduce
each table is [`documents/REPRODUCTION.md`](documents/REPRODUCTION.md); the one-file format the
server and the page load is [`documents/FAST_LLM.md`](documents/FAST_LLM.md).

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE), which lists what is derived from
LatentPort and from other work, and how.
