"""Teacher-forced scoring through upstream's own API — shared by the fidelity table
(oracle_fidelity.py) and the server chat's per-prompt closeness (oracle_worker.py), so the two are one
measurement. docs/BRIDGE.md §1.7.
"""
import numpy as np


def logprob(logits, token):
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    m = x.max()
    return float(x[token] - m - np.log(np.exp(x - m).sum()))


def begin(model, state):
    values = (state._read_tensors() if getattr(model.backend, "copy_on_write", False)
              else state._tensors(device=getattr(model.backend, "device", None)))
    return model.backend.begin_decode(model.identity, values, state.clock)


def score(handle, ids):
    total = 0.0
    for t in ids:
        total -= logprob(handle.logits, t)
        handle.step(t)
    return total


def greedy(handle, n, eos):
    ids, total = [], 0.0
    for _ in range(n):
        t = int(np.argmax(np.asarray(handle.logits).reshape(-1)))
        if t in eos:
            break
        total -= logprob(handle.logits, t)
        ids.append(t)
        handle.step(t)
    return ids, total


def composite_state(port, text):
    session = port.source.session()
    return port.bridge.translate(session.prepare(text), target=port.target)


def closeness(port, text, ref_tokens, eos):
    """The native 27B's own greedy reply to `text` (`ref_tokens` tokens) scored teacher-forced by the
    native 27B, the bridged 27B and an empty 27B: total nats each, the inputs of NCR."""
    native = port.target.capture((("user", text),), generation_prompt=True)
    ref, native_nll = greedy(begin(port.target, native), ref_tokens, eos)
    empty = port.target.capture((("user", " "),), generation_prompt=True)
    return {"native": native_nll,
            "bridged": score(begin(port.target, composite_state(port, text)), ref),
            "empty": score(begin(port.target, empty), ref),
            "ref_tokens": len(ref), "reference": port.target.codec.decode(ref, special=True)}
