"""Run upstream's examples/ttft_demo.py UNCHANGED, with the native options this machine needs.

The only difference from upstream's own run is the NativeOptions handed to latentport.load:
  * mtp_drafts=0  — the timed runs decode with multi-token prediction off (upstream's method says
                    so); 0 also skips allocating the draft buffers.
  * context_size  — sized to the prompt, from PRATTLE_ORACLE_CONTEXT, instead of the fixed 18432.
And, only when PRATTLE_ORACLE_GPUS is set, the translator's device (see patch_oracle_two_gpu.py).
Upstream qualified a 32 GB card; on 24 GB the default allocation does not fit beside both models.
Everything else — prompt, timer, counters, warm-ups, alternating order, statistic — is upstream's.

usage: python oracle_ttft.py <path to upstream ttft_demo.py> [ttft_demo arguments...]
"""
import functools
import os
import runpy
import sys

# Two GPUs (scripts/patch_oracle_two_gpu.py): the translator runs where the source model lives,
# named explicitly — torch's "current device" is per-thread and the native runtime moves it.
# SET BEFORE IMPORTING latentport: some patched sites are default arguments and class attributes,
# evaluated when the module is imported.
if os.environ.get("PRATTLE_ORACLE_GPUS"):
    os.environ["PRATTLE_TRANSLATOR_DEVICE"] = "cuda:" + os.environ["PRATTLE_ORACLE_GPUS"].split(",")[1]

import latentport
from latentport._native.engine import NativeOptions

options = NativeOptions(context_size=int(os.environ["PRATTLE_ORACLE_CONTEXT"]), mtp_drafts=0)

latentport.load = functools.partial(latentport.load, native_options=options)

demo = sys.argv[1]
sys.argv = [demo, *sys.argv[2:]]
runpy.run_path(demo, run_name="__main__")
