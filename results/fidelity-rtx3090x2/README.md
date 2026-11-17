# Fidelity beside the speed-up — LatentPort 4B → 27B on two RTX 3090s

`docs/BRIDGE.md` §1.7, adapted to the product pair. Regenerate with `just fidelity`. `fidelity.md`
is the full table; `results.json` holds every document's losses and the start of its reference.

**Reference:** for each of the same 8 prompts per length as `results/e120-rtx3090x2/`, the native
27B's own greedy 64-token reply. It is scored teacher-forced, in nats per byte of reference, under:

| condition | what reads the prompt |
|---|---|
| native | the 27B itself |
| composite | the 4B, translated by upstream's bridge into the 27B |
| source | the 4B itself, scoring the reference in its own tokens |
| empty | nobody — the 27B with an empty user turn |
| wrong_donor | the 4B, but on ANOTHER prompt, translated into the 27B |

`NCR = (ℓ_empty − ℓ_c) / (ℓ_empty − ℓ_native)` (paper Eq. 8): the share of the gap between an empty
27B and the native 27B that a condition closes. 95% intervals: 10,000 paired bootstrap resamples
over prompts, seed 1 (`prattle_core::measure::bootstrap_ncr`).

| length | composite | 4B alone | wrong donor |
|---|---|---|---|
| 128 | **0.914** (0.903–0.923) | 0.819 (0.791–0.849) | 0.637 (0.608–0.663) |
| 2K | **0.838** (0.829–0.847) | 0.682 (0.662–0.702) | 0.587 (0.568–0.607) |
| 8K | **0.840** (0.832–0.849) | 0.707 (0.687–0.724) | 0.602 (0.581–0.626) |
| 16K | **0.846** (0.839–0.852) | 0.713 (0.694–0.730) | 0.612 (0.587–0.636) |

**What it shows:** at every length the 27B answering from the 4B's state stays closer to what the
27B would have said than the 4B itself does, and further from it when the state comes from the
wrong prompt — the intervals separate in every row.

**Read these conditions before quoting a number.**

- **The reference is the native 27B's own greedy reply**, not held-out human text as in the paper,
  so native is favoured by construction; NCR measures closeness to the native 27B, not correctness.
- **The wrong-donor control is weak here.** All prompts share one question and one log style, and
  each reference begins by restating the question, so another prompt's state still predicts much of
  it (0.59–0.64, where the paper's wrong donor scored −0.36 on different documents). It still sits
  well below the right donor at every length.
- **Two cards, patched** (`scripts/patch_oracle_two_gpu.py`); Q4_K_M weights; thinking-mode replies.
- **Not the paper's pair or setting**: the product's 4B → 27B translator, chat-rendered prompts,
  64 scored tokens over 8 prompts per length.
