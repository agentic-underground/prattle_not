"""The fidelity panel beside the reproduced speed-up — docs/BRIDGE.md §1.7, adapted to the product pair.

For each prompt, the native 27B's own greedy reply (REF_TOKENS tokens) is the reference, scored
teacher-forced under five conditions, all through upstream's own API:

  native       the 27B reads the prompt itself          (target.capture → begin_decode)
  composite    4B reads, bridge translates, 27B scores   (source session → bridge.translate → begin_decode)
  empty        the 27B with an empty user turn
  wrong_donor  the composite state of ANOTHER prompt, scored on this prompt's reference
  source       the 4B reads the prompt itself, scoring the reference in ITS OWN tokens

Totals are nats; `reproduce fidelity` divides by the reference's UTF-8 bytes, so the 4B (another
tokenizer) is comparable and NCR is unchanged for the rest. Writes the RAW record.

usage: python oracle_fidelity.py <upstream ttft_demo.py> <prompt dir> <output dir>
env:   PRATTLE_LENGTHS, PRATTLE_PROMPTS, PRATTLE_QUESTION, PRATTLE_ORACLE_GPUS, PRATTLE_REF_TOKENS
"""
import json
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

if os.environ.get("PRATTLE_ORACLE_GPUS"):   # before importing latentport — see oracle_ttft.py
    os.environ["PRATTLE_TRANSLATOR_DEVICE"] = "cuda:" + os.environ["PRATTLE_ORACLE_GPUS"].split(",")[1]

import latentport
from latentport._native.engine import NativeOptions

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script's own folder off
from oracle_scoring import begin, composite_state, greedy, score

demo = runpy.run_path(sys.argv[1])
prompt_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)
lengths = [int(x) for x in os.environ.get("PRATTLE_LENGTHS", "128,2048,8192,16384").split(",")]
prompts = int(os.environ.get("PRATTLE_PROMPTS", "8"))
REF_TOKENS = int(os.environ.get("PRATTLE_REF_TOKENS", "64"))
question = os.environ["PRATTLE_QUESTION"]

record = {"protocol": "docs/BRIDGE.md §1.7 adapted: teacher-forced NLL of the native 27B's greedy reply",
          "started_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
          "gpus": os.environ.get("PRATTLE_ORACLE_GPUS") or "upstream single-GPU",
          "ref_tokens": REF_TOKENS, "question": question, "lengths": []}


def save():
    (out / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


with demo["quiet_native"](out / "native.log"):
    options = NativeOptions(context_size=max(lengths) + 1024, mtp_drafts=0)
    port = latentport.load(backend="gguf", native_options=options)
    port.warm()
    eos = set(port.target.backend.eos_ids(port.target.identity))
    for length in lengths:
        texts = [demo["make_prompt"](port.source.codec, length, prompt_dir / f"prompt-{v}.txt", question)[0]
                 for v in range(prompts)]
        entry = {"length": length, "documents": []}
        for p, text in enumerate(texts):
            native = port.target.capture((("user", text),), generation_prompt=True)
            ref, native_nll = greedy(begin(port.target, native), REF_TOKENS, eos)
            ref_text = port.target.codec.decode(ref, special=True)
            nll = {"native": native_nll,
                   "composite": score(begin(port.target, composite_state(port, text)), ref)}
            empty = port.target.capture((("user", " "),), generation_prompt=True)
            nll["empty"] = score(begin(port.target, empty), ref)
            nll["wrong_donor"] = score(begin(port.target, composite_state(port, texts[(p + 1) % prompts])), ref)
            source = port.source.capture((("user", text),), generation_prompt=True)
            nll["source"] = score(begin(port.source, source), list(port.source.codec.encode(ref_text)))
            entry["documents"].append({"prompt": p, "ref_tokens": len(ref), "ref_bytes": len(ref_text.encode()),
                                       "reference": ref_text[:200], "nll": nll})
        record["lengths"].append(entry)
        save()
    record["success"] = True
    save()
