"""Dump upstream's translator on one prompt: its exact inputs, every stage between, and its outputs —
the reference the Rust translator is checked against stage by stage (docs/BRIDGE.md slice 11e).

Runs upstream's own Bridge.translate on the 4B's state for <text> and records, through hooks on
upstream's own modules (nothing is recomputed here):

  in.<field>        the 4B's canonical state, as the translator received it
  meta.centers      the numeric alignment: one source-position center per 27B position
  meta.lengths      [source length, target length]
  parent.memory, parent.g          the persistent encoder's tokens and their mean
  parent.content, parent.routed    the chunk encoder's output and the position router's
  parent.q                         the query decoder's output (after its identity hook)
  parent.delta                     the head-basis correction to R
  kv.routed                        the K/V branch's routed features
  out.<field>       the translated 27B state the translator returned
  tokens.source, tokens.target     both models' token ids for the rendered prompt (the aligner's input)
  first.composite                  upstream's first token from the translated state (begin_decode)

usage: python oracle_dump_translator.py <output stem> <text>
"""
import json
import os
import sys

if os.environ.get("PRATTLE_ORACLE_GPUS"):   # before importing latentport — see oracle_ttft.py
    os.environ["PRATTLE_TRANSLATOR_DEVICE"] = "cuda:" + os.environ["PRATTLE_ORACLE_GPUS"].split(",")[1]

import numpy as np
import torch
import latentport
from latentport._native.engine import NativeOptions
from safetensors.torch import save_file

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves the script's own folder off
from oracle_scoring import begin

stem, text = sys.argv[1], sys.argv[2]
port = latentport.load(backend="gguf", native_options=NativeOptions(context_size=1024, mtp_drafts=0))
port.warm()
translator = port.source.backend._native().bridge   # the GGUFTranslator the engine calls
a, parent, kv, h = translator._load()
seen = {}

def keep(name, value):
    seen[name] = value.detach().float().contiguous().cpu()

base = parent.base
persistent, encode_chunk = base.persistent, base.encode_chunk
def persistent_hook(state):
    memory, g = persistent(state)
    keep("parent.memory", memory); keep("parent.g", g)
    return memory, g
chunks = []
def encode_hook(flat, scale, shift):
    out = encode_chunk(flat, scale, shift)
    chunks.append(out)
    return out
base.persistent, base.encode_chunk = persistent_hook, encode_hook
base.position_router.register_forward_hook(lambda m, args, out: keep("parent.routed", out[0]))
base.decoder_norm.register_forward_hook(lambda m, args, out: keep("parent.q", out))
parent.head_basis.register_forward_hook(lambda m, args, out: keep("parent.delta", out))
routed_features = a.routed_features
def routed_hook(model, source, metadata):
    out = routed_features(model, source, metadata)
    keep("kv.routed", out)
    return out
a.routed_features = routed_hook

inner = translator.translate
def translate_hook(components, source_clock, metadata):
    for k, v in components.items():
        keep("in." + k, v)
    keep("meta.centers", torch.tensor(metadata["centers"], dtype=torch.float32))
    keep("meta.lengths", torch.tensor([metadata["source_length"], metadata["target_length"]], dtype=torch.float32))
    values = inner(components, source_clock, metadata)
    for k, v in values.items():
        keep("out." + k, v)
    return values
translator.translate = translate_hook

state = port.source.session().prepare(text)
translated = port.bridge.translate(state, target=port.target)
keep("tokens.source", torch.tensor(state.rendered_token_ids, dtype=torch.float32))
keep("tokens.target", torch.tensor(translated.rendered_token_ids, dtype=torch.float32))
first = int(np.argmax(np.asarray(begin(port.target, translated).logits).reshape(-1)))
keep("first.composite", torch.tensor([first], dtype=torch.float32))
keep("parent.content", torch.cat(chunks[:len(chunks)], 1))
save_file(seen, stem + ".safetensors")
with open(stem + ".json", "w", encoding="utf-8") as f:
    json.dump({"text": text, "shapes": {k: list(v.shape) for k, v in sorted(seen.items())}}, f, indent=1)
print(f"{len(seen)} tensors -> {stem}.safetensors")
