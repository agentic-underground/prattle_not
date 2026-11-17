"""Dump the oracle's canonical persistent state for a fixed prompt — the reference every Rust port of
a piece is checked against (docs/BRIDGE.md slice 11e).

Prefills one model — the SOURCE (4B) or the TARGET (27B) — with the given text through upstream's own
backend.capture, in that model's own tokens, and writes every canonical field (K.i, V.i, C.i, R.i, H)
plus the token ids to <out>.safetensors / <out>.json.

usage: python oracle_dump_state.py <source|target> <output stem> <text>
"""
import json
import os
import sys

if os.environ.get("PRATTLE_ORACLE_GPUS"):   # before importing latentport — see oracle_ttft.py
    os.environ["PRATTLE_TRANSLATOR_DEVICE"] = "cuda:" + os.environ["PRATTLE_ORACLE_GPUS"].split(",")[1]

import latentport
from latentport._native.engine import NativeOptions
from safetensors.torch import save_file

side, stem, text = sys.argv[1], sys.argv[2], sys.argv[3]
port = latentport.load(backend="gguf", native_options=NativeOptions(context_size=1024, mtp_drafts=0))
model = port.source if side == "source" else port.target
ids = list(model.codec.encode(text))
values = model.backend.capture(model.identity, ids)
tensors = {k: v.detach().float().contiguous().cpu() for k, v in values.items() if hasattr(v, "detach")}
save_file(tensors, stem + ".safetensors")
with open(stem + ".json", "w", encoding="utf-8") as f:
    json.dump({"text": text, "ids": ids, "fields": sorted(tensors), "shapes": {k: list(v.shape) for k, v in tensors.items()}}, f, indent=1)
print(f"{len(ids)} tokens, {len(tensors)} fields -> {stem}.safetensors")
