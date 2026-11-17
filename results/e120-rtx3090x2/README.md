# E120's protocol, reproduced on two RTX 3090s

LatentPort 4B → 27B (upstream `latentport/latentport-4-27b-research-preview-GGUF`, Q4_K_M), timed by
upstream's own `examples/ttft_demo.py::run_arm`, statistic by `prattle_core::measure` (E120's:
geometric mean over prompts of per-prompt mean log ratios, 95% Student interval). Regenerate with
`just reproduce`. `table.md` is the result; `results.json` is every run.

| length | native 27B | composite 4B→bridge→27B | speed-up (95% CI) | upstream E120, RTX 5090 |
|---|---|---|---|---|
| 128 | 0.146 s | 0.108 s | **1.36×** (1.33–1.39) | 1.06× (0.87–1.29) |
| 2K | 1.585 s | 0.540 s | **2.94×** (2.92–2.95) | 1.65× (1.61–1.69) |
| 8K | 6.528 s | 1.983 s | **3.29×** (3.27–3.31) | 2.01× (1.96–2.05) |
| 16K | 13.759 s | 4.049 s | **3.39×** (3.38–3.40) | 2.19× (2.15–2.23) |

**Read these conditions before quoting a number.**

- **Same design, not the same run.** 8 prompts × 3 paired repetitions per length, alternating order,
  one discarded warm-up per arm, both models resident, greedy, MTP off — E120's design. E120's own
  prompts are unpublished; these are 8 synthetic incident logs (`reproduce prompts`), distinct by
  SHA-256, with one question.
- **Two cards, patched.** Upstream's runtime pins everything to one GPU and qualifies a 32 GB card;
  24 GB does not hold 8K or 16K. A disclosed patch (`scripts/patch_oracle_two_gpu.py`) puts the 27B
  on GPU 0 and the 4B and translator on GPU 1. These cards have no peer-to-peer access, so the
  translated state crosses PCIe **inside the composite clock** (~45 ms at 2K against one card).
- **Other weights and hardware.** Q4_K_M GGUF on RTX 3090s here; NF4 weights with BF16 compute on an
  RTX 5090 for E120. Absolute times and ratios differ; the shape — the gain rising with length — is
  what agrees.
- **Counters held in all 192 runs**: every composite run's target read 0 prompt tokens; every native
  run's source read 0.
- **This is speed, not quality.** The previews are the first 8 tokens of a thinking-mode reply and
  restate the question, so they are identical across prompts and say nothing about whether the
  composite answer is right. A speed-up without its fidelity beside it is not a LatentPort result:
  **[`results/fidelity-rtx3090x2/`](../fidelity-rtx3090x2/README.md)** is that panel, on the same prompts.
