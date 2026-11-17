"""prattle-server's oracle worker: upstream's 4B → 27B bridge, answering questions over JSON lines.

stdin:  {"id": 1, "question": "...", "mode": "bridge" | "native", "conversation": "c1"}
        {"id": 2, "closeness": "<the first turn's whole prompt>"}  — oracle_scoring.closeness
stdout: {"ready": true, ...} once, then per question
        {"id": 1, "reply": "...", "ttft_s": .., "total_s": .., "prefill": {"source": n, "target": m}, ...}
        or {"id": 1, "error": "..."}

A conversation is ONE upstream session, as in upstream's examples/chat.py: the first question is
read by the 4B and bridged; each follow-up is appended to the 27B's retained state and nothing is
replayed (policy "cached_handoff", upstream's default). Each turn is timed as
examples/ttft_demo.py::run_arm times an answer — the same cleanup, clock and per-token stamp — and
its counters are checked: a first bridged turn whose 27B read prompt tokens is an error, not a reply.
The native runtime writes to the process's stdout; quiet_native moves that to a log.

usage: python oracle_worker.py <upstream ttft_demo.py> <native log path>
env:   PRATTLE_ORACLE_GPUS (see patch_oracle_two_gpu.py), PRATTLE_ORACLE_CONTEXT (default 8192)
"""
import gc
import io
import json
import os
import runpy
import sys
from collections import OrderedDict
from time import perf_counter
from pathlib import Path

if os.environ.get("PRATTLE_ORACLE_GPUS"):   # before importing latentport — see oracle_ttft.py
    os.environ["PRATTLE_TRANSLATOR_DEVICE"] = "cuda:" + os.environ["PRATTLE_ORACLE_GPUS"].split(",")[1]

import latentport
from latentport._native.engine import NativeOptions

sys.dont_write_bytecode = True                              # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script's own folder off
from oracle_scoring import closeness

demo = runpy.run_path(sys.argv[1])
requests = sys.stdin

with demo["quiet_native"](Path(sys.argv[2])) as console:
    def say(obj):
        console.write(json.dumps(obj) + "\n")
        console.flush()

    screen = demo["Screen"](io.StringIO())
    options = NativeOptions(context_size=int(os.environ.get("PRATTLE_ORACLE_CONTEXT", "8192")), mtp_drafts=0)
    port = latentport.load(backend="gguf", native_options=options)
    port.warm()
    for name in ("native", "composite"):   # one discarded warm-up per arm, as E120 does
        demo["run_arm"](port, "Say hello.", name, 4, screen, "WARM-UP")
    say({"ready": True, "source": port.source.identity.repo_id if hasattr(port.source.identity, "repo_id") else "4B",
         "target": port.target.identity.repo_id if hasattr(port.target.identity, "repo_id") else "27B"})
    engine = port.source.backend._native()
    eos = set(port.target.backend.eos_ids(port.target.identity))
    REF_TOKENS = 64   # as the fidelity table
    conversations = OrderedDict()   # id → (mode, session); the oldest is released past KEEP
    KEEP = 4
    TURN_TOKENS, CHUNKS = 256, 8    # upstream's examples/chat.py budget

    def turn(session, question):
        import torch
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()   # as run_arm, before each turn
        before = {side: engine.trace(side)["prefill_tokens"] for side in ("source", "target")}
        stamp = {"first": None}
        def on_token(_):
            if stamp["first"] is None: stamp["first"] = perf_counter()
        started = perf_counter()
        answer = session.ask(question, max_new_tokens=TURN_TOKENS, mtp=False, on_token=on_token)
        tokens = list(answer.token_ids)
        for _ in range(CHUNKS - 1):
            if answer.finished:
                break
            answer = session.generate(max_new_tokens=TURN_TOKENS, mtp=False)
            tokens += list(answer.token_ids)
        ended = perf_counter()
        prefill = {side: engine.trace(side)["prefill_tokens"] - before[side] for side in ("source", "target")}
        return {"reply": port.target.codec.decode(tokens, special=True), "finished": bool(answer.finished),
                "ttft_s": stamp["first"] - started, "total_s": ended - started, "tokens": len(tokens), "prefill": prefill}

    for line in requests:
        if not line.strip():
            continue
        req = None
        try:
            req = json.loads(line)
            if "closeness" in req:
                say({"id": req.get("id"), "nll": closeness(port, req["closeness"], REF_TOKENS, eos)})
                continue
            mode = "native" if req.get("mode") == "native" else "bridge"
            key = str(req.get("conversation") or req.get("id"))
            first = key not in conversations or conversations[key][0] != mode
            if first:
                conversations.pop(key, None)
                session = port.target.session() if mode == "native" else port.composite.session()
                conversations[key] = (mode, session)
                while len(conversations) > KEEP:
                    conversations.popitem(last=False)
            conversations.move_to_end(key)
            r = turn(conversations[key][1], req["question"])
            if first and mode == "bridge" and r["prefill"]["target"] != 0:
                raise RuntimeError(f"the bridged 27B read {r['prefill']['target']} prompt tokens itself")
            if mode == "native" and r["prefill"]["source"] != 0:
                raise RuntimeError(f"the native turn used the 4B ({r['prefill']['source']} tokens)")
            if not r["finished"]:
                conversations.pop(key, None)   # upstream needs the turn's end before another question
            say({"id": req.get("id"), "conversation": key, "mode": mode, "turn": "first" if first else "follow-up", **r})
        except Exception as e:  # one bad question must not stop the worker
            say({"id": req.get("id") if isinstance(req, dict) else None, "error": f"{type(e).__name__}: {e}"})
