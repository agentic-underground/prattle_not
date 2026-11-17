"""Print the __metadata__ keys of a safetensors file, comma-joined and sorted; empty when it has none.

just hub-publish refuses a file whose keys differ from what its manifest line declares, because fit
tools can embed paths and hostnames there.
"""
import json
import struct
import sys

with open(sys.argv[1], "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(n))
print(",".join(sorted(header.get("__metadata__") or {})))
