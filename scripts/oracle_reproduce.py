"""E120's protocol through the oracle: 8 prompts x 3 paired repetitions per length, both models
loaded ONCE, every timed run through upstream's own ttft_demo.run_arm (its timer, its counter
checks). Writes the RAW record; `reproduce table` computes the statistic.

usage: python oracle_reproduce.py <upstream ttft_demo.py> <prompt dir> <output dir>
env:   PRATTLE_LENGTHS (default 128,2048,8192,16384), PRATTLE_PROMPTS (8), PRATTLE_REPS (3),
       PRATTLE_QUESTION, PRATTLE_ORACLE_GPUS (see patch_oracle_two_gpu.py)
"""
import hashlib
import json
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

# Set BEFORE importing latentport: some patched sites are evaluated at import.
if os.environ.get("PRATTLE_ORACLE_GPUS"):
    os.environ["PRATTLE_TRANSLATOR_DEVICE"] = "cuda:" + os.environ["PRATTLE_ORACLE_GPUS"].split(",")[1]

import latentport
from latentport._native.engine import NativeOptions

demo = runpy.run_path(sys.argv[1])          # upstream's functions, without running its main
prompt_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)
lengths = [int(x) for x in os.environ.get("PRATTLE_LENGTHS", "128,2048,8192,16384").split(",")]
prompts = int(os.environ.get("PRATTLE_PROMPTS", "8"))
reps = int(os.environ.get("PRATTLE_REPS", "3"))
question = os.environ["PRATTLE_QUESTION"]
PREVIEW = 8   # tokens per arm after the timed first token, as in the runs already recorded

record = {"protocol": "docs/BRIDGE.md §3.2 (E120 design) through upstream's ttft_demo.run_arm",
          "started_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
          "gpus": os.environ.get("PRATTLE_ORACLE_GPUS") or "upstream single-GPU",
          "native_options": {"context_size": max(lengths) + 1024, "mtp_drafts": 0},
          "question": question, "lengths": []}


def save():
    (out / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def slim(result):
    return {"observed_ttft_s": result["observed_ttft_s"], "first_token_id": result["first_token_id"],
            "prefill_tokens": result["prefill_tokens"], "preview": result["preview"][:120]}


with demo["quiet_native"](out / "native.log") as console:
    screen = demo["Screen"](console)
    options = NativeOptions(context_size=max(lengths) + 1024, mtp_drafts=0)
    port = latentport.load(backend="gguf", native_options=options)
    record["residency"] = port.warm()
    for length in lengths:
        entry = {"length": length, "prompts": [], "warmups": [], "runs": []}
        texts = []
        for v in range(prompts):
            path = prompt_dir / f"prompt-{v}.txt"
            text, count, _ = demo["make_prompt"](port.source.codec, length, path, question)
            texts.append(text)
            entry["prompts"].append({"index": v, "file": path.name, "context_tokens": count,
                                     "sha256": hashlib.sha256(text.encode()).hexdigest()})
        for name in ("native", "composite"):     # one discarded warm-up per arm per length
            entry["warmups"].append({"arm": name, **slim(demo["run_arm"](port, texts[0], name, 1, screen, "WARMUP"))})
        for p in range(prompts):
            for rep in range(reps):
                order = ("native", "composite") if (p + rep) % 2 == 0 else ("composite", "native")
                for name in order:
                    result = demo["run_arm"](port, texts[p], name, PREVIEW, screen, f"{length} p{p} r{rep}")
                    entry["runs"].append({"arm": name, "prompt": p, "rep": rep, **slim(result)})
        record["lengths"].append(entry)
        save()
    record["success"] = True
    save()
