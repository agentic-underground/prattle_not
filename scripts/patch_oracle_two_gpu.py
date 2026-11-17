"""Let upstream's runtime place the 27B and the 4B on DIFFERENT GPUs — a disclosed patch, not upstream.

Upstream pins both models and the translator to cuda:0; its qualified card has 32 GB. On two 24 GB
cards without peer-to-peer access, each model must live on its own device and every state buffer on
the device of the model it belongs to. One variable drives both halves so they cannot disagree:

    PRATTLE_ORACLE_GPUS="<target gpu>,<source gpu>"     unset = upstream's behaviour, both on 0

NATIVE (native/latentport.cpp): lp_open places the model, and its auxiliary tensors, on its side's
device; get() — the one
seam every entry point passes through — makes that device current, so upstream's layout kernels and
copies run where the model's memory is.
PYTHON (src/latentport/_native/engine.py): state buffers are allocated on their side's device;
the translator runs on the current device, which scripts/oracle_ttft.py sets to the source's.

usage: python patch_oracle_two_gpu.py <upstream checkout>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()


def edit(name, old, new):
    path = root / name
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) != 1:
        raise SystemExit(f"Patch anchor changed: {name}: {old[:70]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


cpp = "native/latentport.cpp"
edit(cpp, "struct port_context {", "#include <cuda_runtime_api.h>\n#include <cstdio>\n#include <cstdlib>\n"
     "static int prattle_device(int side){const char*v=std::getenv(\"PRATTLE_ORACLE_GPUS\");if(!v)return 0;"
     "int t=0,s=0;if(std::sscanf(v,\"%d,%d\",&t,&s)!=2)return 0;return side?t:s;}\n"
     "struct port_context {\n    int device=0;")
edit(cpp, "port_context & get(void * p){require(p,\"null native context\");return *(port_context*)p;}",
     "port_context & get(void * p){require(p,\"null native context\");auto&c=*(port_context*)p;"
     "cudaSetDevice(c.device);return c;}")
edit(cpp, "auto p=std::make_unique<port_context>();",
     "auto p=std::make_unique<port_context>();p->device=prattle_device(side);cudaSetDevice(p->device);")
edit(cpp, "mp.main_gpu=0;", "mp.main_gpu=p->device;")
edit(cpp, "aux_buf=ggml_backend_alloc_ctx_tensors_from_buft(aux,ggml_backend_cuda_buffer_type(0));",
     "aux_buf=ggml_backend_alloc_ctx_tensors_from_buft(aux,ggml_backend_cuda_buffer_type(device));")
edit(cpp, "p->gpu_weights={{{\".*\",ggml_backend_cuda_buffer_type(0)},{nullptr,nullptr}}};",
     "p->gpu_weights={{{\".*\",ggml_backend_cuda_buffer_type(p->device)},{nullptr,nullptr}}};")

# THE TRANSLATOR'S DEVICE, NAMED. Upstream builds and feeds it with bare "cuda" — the CURRENT device,
# which is per-thread and which the native get() moves. Every device literal below becomes
# PRATTLE_TRANSLATOR_DEVICE (default "cuda", i.e. unchanged). The two `.type == "cuda"` checks are
# comparisons of a device TYPE and are deliberately left alone.
DEV = '__import__("os").environ.get("PRATTLE_TRANSLATOR_DEVICE", "cuda")'
edit("src/latentport/_native/bridge.py", 'means = {k:v.to("cuda:0")', f'means = {{k:v.to({DEV})')
edit("src/latentport/_reference/native.py", 'translation_device = "cuda"', f'translation_device = {DEV}')
edit("src/latentport/gguf.py", '    translation_device = "cuda:0"', f'    translation_device = {DEV}')
edit("src/latentport/_reference/bridge.py", 'population_means.safetensors"), device="cuda")',
     f'population_means.safetensors"), device={DEV})')
edit("src/latentport/_reference/bridge.py", '            source = source.to("cuda")', f'            source = source.to({DEV})')
edit("src/latentport/_reference/bridge.py", 'h = a.RMSLinearHResidual().cuda()', f'h = a.RMSLinearHResidual().to({DEV})')
arch = "src/latentport/_reference/architecture.py"
text = (root / arch).read_text(encoding="utf-8")
if DEV not in text:
    for old, new in [("with torch.device('cuda'):", f"with torch.device({DEV}):"),
                     ("device='cuda', dtype=torch.float32", f"device={DEV}, dtype=torch.float32"),
                     ("def __init__(self, geometry, values, device='cuda'):", f"def __init__(self, geometry, values, device={DEV}):")]:
        if old not in text:
            raise SystemExit(f"Patch anchor changed: {arch}: {old!r}")
        text = text.replace(old, new)
    (root / arch).write_text(text, encoding="utf-8", newline="\n")

py = "src/latentport/_native/engine.py"
edit(py, "    def _fields(values):", "    def _fields(values, device=\"cuda:0\"):")
edit(py, "value=value.to(device=\"cuda:0\").contiguous()", "value=value.to(device=device).contiguous()")
edit(py, "    def _snapshot(self,side):",
     "    @staticmethod\n    def _dev(side):\n"
     "        spec=os.environ.get(\"PRATTLE_ORACLE_GPUS\")\n"
     "        if not spec: return \"cuda:0\"\n"
     "        target,source=(int(x) for x in spec.split(\",\"))\n"
     "        return f\"cuda:{target if side=='target' else source}\"\n\n"
     "    def _snapshot(self,side):")
edit(py, "device=\"cuda:0\") for name in names}", "device=self._dev(side)) for name in names}")
edit(py, "values[\"P\"]=torch.tensor(list(positions),device=\"cuda:0\",dtype=torch.long)",
     "values[\"P\"]=torch.tensor(list(positions),device=self._dev(side),dtype=torch.long)")
edit(py, "h=h.to(\"cuda:0\").contiguous()", "h=h.to(self._dev(side)).contiguous()")
text = (root / py).read_text(encoding="utf-8")
if text.count("fields,keep=self._fields(values)") == 3:
    (root / py).write_text(text.replace("fields,keep=self._fields(values)",
                                        "fields,keep=self._fields(values,self._dev(side))"),
                           encoding="utf-8", newline="\n")
elif "self._fields(values,self._dev(side))" not in text:
    raise SystemExit("Patch anchor changed: engine.py _fields call sites")
print(f"Patched {root} for two GPUs (PRATTLE_ORACLE_GPUS)")
