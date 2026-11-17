# oracle.just — RUNNING and MEASURING: the CUDA toolkit, LatentPort's runtime built as an oracle, its
# weights and the runs that regenerate results/, the page's models, a published bridge, and a bridge's
# measured speed-up. Imported by the justfile of BOTH repositories, so there is one definition of how
# a number is made. justfile_directory() is the importing repository's root.

# ---- the GPU toolchain ---------------------------------------------------------------------------
# CUDA 13.0, TOOLKIT ONLY, in user space: no sudo, no driver, no system change. Every GPU build here
# (the oracle's llama.cpp, candle's CUDA kernels) uses it. Remove by deleting the folder.

cuda_version := "13.0.2"
cuda_home := env_var("HOME") / ".cache/prattle/cuda-" + cuda_version

# Where the host tools run: the source's device, then the target's (crates/core/src/device.rs). The two GPUs by default; PRATTLE_DEVICES=cpu for the CPU.
devices := env_var_or_default("PRATTLE_DEVICES", "cuda:0,cuda:1")

# Every tool run holds this lock, so two runs QUEUE for the GPUs instead of one dying out of memory —
# and a timing never shares the cards. A run started by hand takes it too: flock /tmp/prattle-gpu.lock <cmd>.
gpu_lock := "/tmp/prattle-gpu.lock"

# Build one host tool and print its path: with candle's CUDA backend, in a build directory of its own
# beside this tree's, when the devices name a GPU — so the gate's CPU build is never rebuilt for it.
# The tool is found by its name in whichever workspace crate holds it.
_tool BIN:
    #!/usr/bin/env sh
    set -e
    target=$(cargo metadata -q --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
    case "{{ devices }}" in
        *cuda*)
            [ -x "{{ cuda_home }}/bin/nvcc" ] || just cuda >&2
            cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
            PATH="{{ cuda_home }}/bin:$PATH" CUDA_ROOT="{{ cuda_home }}" CUDA_PATH="{{ cuda_home }}" CUDA_COMPUTE_CAP="$cap" \
                CARGO_TARGET_DIR="$target/cuda" cargo build -q --release --features prattle-core/cuda --bin {{ BIN }} >&2
            echo "$target/cuda/release/{{ BIN }}" ;;
        *)
            cargo build -q --release --bin {{ BIN }} >&2
            echo "$target/release/{{ BIN }}" ;;
    esac

# Install the CUDA toolkit to ~/.cache/prattle (4.3 GB download, 7.1 GB installed, once).
cuda:
    #!/usr/bin/env sh
    set -e
    [ -x "{{ cuda_home }}/bin/nvcc" ] && { "{{ cuda_home }}/bin/nvcc" --version | tail -1; exit 0; }
    run="$HOME/.cache/prattle/cuda_{{ cuda_version }}.run"
    mkdir -p "$HOME/.cache/prattle"
    [ -s "$run" ] || curl -sSfL -o "$run" "https://developer.download.nvidia.com/compute/cuda/{{ cuda_version }}/local_installers/cuda_{{ cuda_version }}_580.95.05_linux.run"
    sh "$run" --silent --toolkit --toolkitpath="{{ cuda_home }}" --no-man-page --no-drm --override --tmpdir="$HOME/.cache/prattle"
    rm -f "$run"
    "{{ cuda_home }}/bin/nvcc" --version | tail -1

# ---- the oracle: upstream's runtime, built here -------------------------------------------------
# docs/BRIDGE.md slice 11. Upstream qualifies it on Windows + RTX 5090 only; this builds the SAME
# pinned sources for this machine's GPUs. Everything lands in ~/.cache/prattle/oracle — nothing in
# the repository, and upstream's source is fetched at its pin rather than copied in.

oracle_dir := env_var("HOME") / ".cache/prattle/oracle"
upstream_rev := "6be1956f6ef19528e16ea387ee39a00e9a5e733a"
llama_rev := "b19cbe925be361d229f0fe03435affe4a2717f37"

# Build the oracle runtime for the GPUs in this machine.
oracle:
    #!/usr/bin/env sh
    set -e
    just cuda
    d="{{ oracle_dir }}"; mkdir -p "$d"
    fetch() { [ -d "$2/.git" ] || git init -q "$2"; git -C "$2" fetch -q --depth 1 "$1" "$3" && git -C "$2" checkout -q FETCH_HEAD; }
    fetch https://github.com/LatentPort/latentport "$d/latentport" "{{ upstream_rev }}"
    fetch https://github.com/ggml-org/llama.cpp "$d/llama.cpp" "{{ llama_rev }}"
    python3 -I -B "$d/latentport/native/patch_llama.py" "$d/llama.cpp"
    python3 -I -B "{{ justfile_directory() }}/scripts/patch_oracle_two_gpu.py" "$d/latentport"
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
    export PATH="{{ cuda_home }}/bin:$PATH"
    cmake -S "$d/latentport/native" -B "$d/build" -DCMAKE_BUILD_TYPE=Release \
        -DLATENTPORT_LLAMA_SOURCE="$d/llama.cpp" -DCMAKE_CUDA_ARCHITECTURES="$arch" \
        -DCMAKE_CUDA_COMPILER="{{ cuda_home }}/bin/nvcc" -DCUDAToolkit_ROOT="{{ cuda_home }}" > "$d/configure.log"
    cmake --build "$d/build" --parallel "$(nproc)" --target llama latentport-layout-test > "$d/build.log"
    "$(find "$d/build" -name latentport-layout-test -type f | head -1)"
    echo "✅ oracle built for sm_$arch: $(find "$d/build" -name 'libllama.so' | head -1)"

# Upstream's Python package and torch, for the translator — upstream runs its 306M bridge as a torch
# graph, not in the native runtime. A venv in ~/.cache/prattle/oracle; upstream's own pins apply.
oracle-python:
    #!/usr/bin/env sh
    set -e
    d="{{ oracle_dir }}"
    [ -d "$d/latentport/.git" ] || just oracle
    [ -x "$d/venv/bin/python" ] || uv venv -q -p 3.13 "$d/venv"
    VIRTUAL_ENV="$d/venv" uv pip install -q "torch==2.11.0" -e "$d/latentport"
    "$d/venv/bin/python" -I -c 'import torch, latentport; print("✅ torch", torch.__version__, "cuda", torch.version.cuda, "· latentport", latentport.__version__ if hasattr(latentport, "__version__") else "")'

# Upstream's published 4B→27B GGUFs (~22 GB) into models/latentport-4-27b/, each verified against
# the SHA-256 Hugging Face records for it at the revision upstream's installation guide pins.
oracle_models_rev := "1415eb9591dfaf15355d8bba5ed38f64792026e0"

oracle-models:
    #!/usr/bin/env sh
    set -e
    d=$(just _weights gguf latentport/latentport-4-27b-research-preview-GGUF latentport-4-27b); mkdir -p "$d/models"
    base="https://huggingface.co/latentport/latentport-4-27b-research-preview-GGUF/resolve/{{ oracle_models_rev }}"
    tree="https://huggingface.co/api/models/latentport/latentport-4-27b-research-preview-GGUF/tree/{{ oracle_models_rev }}?recursive=1"
    curl -sSfL "$tree" | python3 -I -c 'import json,sys; [print(e["path"], e["lfs"]["oid"]) for e in json.load(sys.stdin) if e["path"].endswith(".gguf")]' |
    while read -r path sha; do
        f="$d/$path"
        if [ -s "$f" ] && [ -s "$f.ok" ]; then echo "✅ $path"; continue; fi
        curl -sSfL -C - -o "$f" "$base/$path"
        echo "$sha  $f" | sha256sum -c --quiet && : > "$f.ok" && echo "✅ $path verified"
    done
    # The tokenizers and chat templates upstream renders prompts with, at the revisions and SHA-256
    # upstream records for them (latentport.json in its Hugging Face bundle).
    while read -r side repo rev file sha; do
        f="$d/tokenizers/$side/$file"; mkdir -p "$(dirname "$f")"
        [ -s "$f" ] || curl -sSfL -o "$f" "https://huggingface.co/$repo/resolve/$rev/$file"
        echo "$sha  $f" | sha256sum -c --quiet && echo "✅ $side $file verified"
    done <<'PINS'
    source Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a tokenizer.json 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
    source Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a chat_template.jinja a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
    target Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 tokenizer.json 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
    target Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 chat_template.jinja c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041
    PINS

# The native arm of upstream's TTFT experiment: the 27B reads upstream's ~16K demo prompt itself and
# its first token is timed. CONTEXT sets the log length; GGUF picks another model (e.g. the 4B).
oracle-native CONTEXT="16384" GGUF="":
    PRATTLE_CUDA_HOME="{{ cuda_home }}" PRATTLE_CONTEXT_TOKENS="{{ CONTEXT }}" PRATTLE_NATIVE_GGUF="{{ GGUF }}" \
        cargo nextest run -p oracle --run-ignored only --no-capture -E 'binary(native)'

# Upstream's own code fetches its GGUFs through a Hugging Face cache under LATENTPORT_HOME. The blobs
# it asks for are byte-identical to the ones oracle-models verified (same SHA-256), so they are
# hard-linked in by hash — upstream's downloader then finds them and fetches only small metadata.
oracle_home := justfile_directory() / "models/latentport-home"

# Where a model's files live, printed. With PRATTLE_WEIGHTS set — a weights store outside the checkout
# — safetensors repos live at $PRATTLE_WEIGHTS/MODELS/<name> and GGUF repos at
# $PRATTLE_WEIGHTS/GGUF_MODELS/<org>/<repo>, and models/<name> is a link to them, so everything that
# reads models/ is unchanged. Unset, the files are in models/<name>. KIND is `safetensors` or `gguf`.
_weights KIND REPO NAME:
    #!/usr/bin/env sh
    set -e
    link="{{ justfile_directory() }}/models/{{ NAME }}"
    if [ -z "${PRATTLE_WEIGHTS:-}" ]; then mkdir -p "$link"; echo "$link"; exit 0; fi
    repo="{{ REPO }}"
    if [ "{{ KIND }}" = gguf ]; then dir="$PRATTLE_WEIGHTS/GGUF_MODELS/$repo"; else dir="$PRATTLE_WEIGHTS/MODELS/${repo#*/}"; fi
    mkdir -p "$dir" "$(dirname "$link")"
    if [ -L "$link" ]; then :
    elif [ -e "$link" ]; then echo "⛔ $link is a folder of its own — move its files into $dir, then rerun" >&2; exit 1
    else ln -s "$dir" "$link"; fi
    echo "$link"

oracle-home:
    #!/usr/bin/env sh
    set -e
    just oracle-models
    blobs="{{ oracle_home }}/huggingface/models--latentport--latentport-4-27b-research-preview-GGUF/blobs"
    mkdir -p "$blobs"
    curl -sSfL "https://huggingface.co/api/models/latentport/latentport-4-27b-research-preview-GGUF/tree/{{ oracle_models_rev }}?recursive=1" |
    python3 -I -c 'import json,sys; [print(e["path"], e["lfs"]["oid"]) for e in json.load(sys.stdin) if e["path"].endswith(".gguf")]' |
    while read -r path sha; do
        ln -sf "$(realpath "models/latentport-4-27b/$path")" "$blobs/$sha" && echo "✅ $path linked as $sha"
    done

# Upstream's own TTFT demo — native 27B against the composite 4B→bridge→27B — run unchanged except
# for the native options in scripts/oracle_ttft.py. GPUS="<target>,<source>" places the models
# (scripts/patch_oracle_two_gpu.py); GPUS="" is upstream's single-GPU layout. Writes target/oracle/.
oracle-ttft CONTEXT="16384" ROUNDS="3" GPUS="0,1":
    #!/usr/bin/env sh
    set -e
    o="{{ oracle_dir }}"; out="{{ justfile_directory() }}/target/oracle/ttft-{{ CONTEXT }}-$(date +%s)"
    mkdir -p "$(dirname "$out")"
    cd "$o/latentport"
    LATENTPORT_NATIVE_LIB="$o/build/bin/libllama.so" LATENTPORT_HOME="{{ oracle_home }}" \
    LD_LIBRARY_PATH="{{ cuda_home }}/lib64" PRATTLE_ORACLE_CONTEXT=$(( {{ CONTEXT }} + 1024 )) PRATTLE_ORACLE_GPUS="{{ GPUS }}" \
        "$o/venv/bin/python" -I "{{ justfile_directory() }}/scripts/oracle_ttft.py" examples/ttft_demo.py \
        --yes --rounds {{ ROUNDS }} --context-tokens {{ CONTEXT }} --preview-tokens 8 --output "$out" > "$out.log" 2>&1 || true
    grep -aE 'DEMO STOPPED|TTFT SPEEDUP|FIRST TOKEN in' "$out.log" | tail -4
    echo "record: $out/results.json"

# E120's protocol through the two-GPU oracle — 8 prompts × 3 paired repetitions at each length —
# and E120's statistic from prattle_core::measure, beside upstream's table. Writes target/oracle/.
reproduce LENGTHS="128,2048,8192,16384" PROMPTS="8" REPS="3" GPUS="0,1":
    #!/usr/bin/env sh
    set -e
    o="{{ oracle_dir }}"; root="{{ justfile_directory() }}"
    out="$root/target/oracle/reproduce-$(date +%s)"; prompts="$root/target/oracle/prompts"
    cargo run -q -p oracle --bin reproduce -- prompts "$prompts" "$root/models/latentport-4-27b/tokenizers/source/tokenizer.json" {{ PROMPTS }} 17000
    q=$(cat "$prompts/question.txt")
    cd "$o/latentport"
    LATENTPORT_NATIVE_LIB="$o/build/bin/libllama.so" LATENTPORT_HOME="{{ oracle_home }}" LD_LIBRARY_PATH="{{ cuda_home }}/lib64" \
    PRATTLE_ORACLE_GPUS="{{ GPUS }}" PRATTLE_LENGTHS="{{ LENGTHS }}" PRATTLE_PROMPTS="{{ PROMPTS }}" PRATTLE_REPS="{{ REPS }}" PRATTLE_QUESTION="$q" \
        "$o/venv/bin/python" -I "$root/scripts/oracle_reproduce.py" examples/ttft_demo.py "$prompts" "$out" > "$out.log" 2>&1
    cd "$root" && cargo run -q -p oracle --bin reproduce -- table "$out/results.json" | tee "$out/table.md"
    echo "record: $out/results.json"

# The fidelity panel beside the speed-up (docs/BRIDGE.md §1.7, adapted): the native 27B's own reply,
# scored teacher-forced under native, composite, 4B, empty and wrong-donor conditions.
fidelity LENGTHS="128,2048,8192,16384" PROMPTS="8" GPUS="0,1":
    #!/usr/bin/env sh
    set -e
    o="{{ oracle_dir }}"; root="{{ justfile_directory() }}"
    out="$root/target/oracle/fidelity-$(date +%s)"; prompts="$root/target/oracle/prompts"
    cargo run -q -p oracle --bin reproduce -- prompts "$prompts" "$root/models/latentport-4-27b/tokenizers/source/tokenizer.json" {{ PROMPTS }} 17000 > /dev/null
    q=$(cat "$prompts/question.txt")
    cd "$o/latentport"
    LATENTPORT_NATIVE_LIB="$o/build/bin/libllama.so" LATENTPORT_HOME="{{ oracle_home }}" LD_LIBRARY_PATH="{{ cuda_home }}/lib64" \
    PRATTLE_ORACLE_GPUS="{{ GPUS }}" PRATTLE_LENGTHS="{{ LENGTHS }}" PRATTLE_PROMPTS="{{ PROMPTS }}" PRATTLE_QUESTION="$q" \
        "$o/venv/bin/python" -I "$root/scripts/oracle_fidelity.py" examples/ttft_demo.py "$prompts" "$out" > "$out.log" 2>&1
    cd "$root" && cargo run -q -p oracle --bin reproduce -- fidelity "$out/results.json" | tee "$out/fidelity.md"
    echo "record: $out/results.json"

# The oracle's canonical 4B and 27B states for a fixed prompt, the references for the Rust ports (slice 11e).
oracle-state:
    #!/usr/bin/env sh
    set -e
    o="{{ oracle_dir }}"; root="{{ justfile_directory() }}"; mkdir -p "$root/target/oracle"
    cd "$o/latentport"
    text="Incident sample 1: catalog: cached price version 41 served after database advanced to version 42; invalidation event missing. Incident sample 2: checkout: hot key expired; 500 requests simultaneously queried the database."
    for pair in source:4b target:27b; do
        LATENTPORT_NATIVE_LIB="$o/build/bin/libllama.so" LATENTPORT_HOME="{{ oracle_home }}" LD_LIBRARY_PATH="{{ cuda_home }}/lib64" PRATTLE_ORACLE_GPUS="0,1" \
            "$o/venv/bin/python" -I "$root/scripts/oracle_dump_state.py" "${pair%%:*}" "$root/target/oracle/state-${pair#*:}" "$text" 2>&1 | tail -1
    done

# Upstream's translator on the same text: its inputs, every stage between, and its outputs — the
# reference for the Rust translator (crates/core/tests/translator_oracle.rs).
oracle-translator:
    #!/usr/bin/env sh
    set -e
    o="{{ oracle_dir }}"; root="{{ justfile_directory() }}"; mkdir -p "$root/target/oracle"
    cd "$o/latentport"
    LATENTPORT_NATIVE_LIB="$o/build/bin/libllama.so" LATENTPORT_HOME="{{ oracle_home }}" LD_LIBRARY_PATH="{{ cuda_home }}/lib64" PRATTLE_ORACLE_GPUS="0,1" \
        "$o/venv/bin/python" -I "$root/scripts/oracle_dump_translator.py" "$root/target/oracle/translator" \
        "Incident sample 1: catalog: cached price version 41 served after database advanced to version 42; invalidation event missing. Incident sample 2: checkout: hot key expired; 500 requests simultaneously queried the database." \
        2>&1 | tail -1

# Load the oracle behind Rust and ask its ABI.
oracle-doctor:
    PRATTLE_CUDA_HOME="{{ cuda_home }}" cargo nextest run -p oracle --run-ignored only --no-capture

# The server with LatentPort's own 4B → 27B pair behind /chat, through the oracle. Two 24 GB GPUs;
# `just oracle`, `just oracle-python` and `just oracle-home` first. The worker's own log is
# target/oracle/chat-native.log.
chat *ARGS:
    #!/usr/bin/env sh
    set -e
    just web models
    PRATTLE_ORACLE="$(just oracle-command)" cargo run -q -p server --release -- {{ ARGS }}

# The command prattle-server runs as its oracle worker (scripts/oracle_worker.py).
oracle-command GPUS="0,1":
    #!/usr/bin/env sh
    o="{{ oracle_dir }}"; log="{{ justfile_directory() }}/target/oracle/chat-native.log"
    mkdir -p "$(dirname "$log")"
    [ -f "$o/build/bin/libllama.so" ] || { echo "⛔ no oracle — just oracle" >&2; exit 1; }
    echo "cd '$o/latentport' && exec env LATENTPORT_NATIVE_LIB='$o/build/bin/libllama.so' LATENTPORT_HOME='{{ oracle_home }}' LD_LIBRARY_PATH='{{ cuda_home }}/lib64' PRATTLE_ORACLE_GPUS='{{ GPUS }}' '$o/venv/bin/python' -I '{{ justfile_directory() }}/scripts/oracle_worker.py' examples/ttft_demo.py '$log'"

# ---- the browser bridge: SmolLM2-135M → SmolLM2-360M ---------------------------------------------

# The models the page uses (the answering model and the bridge's reader), into models/ — which prattle-server serves at /models/.
models:
    #!/usr/bin/env sh
    set -e
    for name in SmolLM2-135M-Instruct SmolLM2-135M SmolLM2-360M-Instruct; do
        m=$(just _weights safetensors "HuggingFaceTB/$name" "$name")
        for f in config.json tokenizer.json model.safetensors; do
            [ -s "$m/$f" ] || curl -sSfL -o "$m/$f" "https://huggingface.co/HuggingFaceTB/$name/resolve/main/$f"
        done
    done

# Download the PUBLISHED 135M → 360M bridge into models/bridges/ instead of fitting it: the file the
# page loads, at the hub commit the page pins (BRIDGE_HUB in crates/web/www/index.html is the one
# definition). No token. A bridge already there is kept, and the command says whether it matches.
bridge-fetch:
    #!/usr/bin/env sh
    set -e
    url=$(sed -n "s/^const BRIDGE_HUB = '\(.*\)';$/\1/p" crates/web/www/index.html)
    [ -n "$url" ] || { echo "⛔ no BRIDGE_HUB in crates/web/www/index.html"; exit 1; }
    out=models/bridges/smollm2-135m-to-360m.safetensors
    mkdir -p models/bridges
    tmp=$(mktemp -p models/bridges)
    curl -sSfL -o "$tmp" "$url" || { rm -f "$tmp"; echo "⛔ $url did not download"; exit 1; }
    hub=$(sha256sum < "$tmp" | cut -d' ' -f1)
    if [ ! -e "$out" ]; then
        mv "$tmp" "$out"
        echo "✅ $out — the published bridge, sha256 $hub, from $url"
    elif [ "$(sha256sum < "$out" | cut -d' ' -f1)" = "$hub" ]; then
        rm -f "$tmp"
        echo "✅ $out already IS the published bridge (sha256 $hub)."
    else
        rm -f "$tmp"
        echo "▶ $out is a bridge fitted here and differs from the published one (sha256 $hub); kept."
        echo "  Move it aside and run this again to measure the published one."
    fi

# A bridge's measured speed-up to first token on this host (crates/core/src/bin/speedup.rs): the
# browser Measure's arms and token clock, E120's design (8 prompts x 3) and statistic, every run's
# counters checked. Run on a quiet machine — a timing beside a fit is a timing of contention.
speedup SRC TGT BRIDGE TOKENS="256" PROMPTS="8" REPS="3":
    #!/usr/bin/env sh
    set -e
    PRATTLE_DEVICES="{{ devices }}" LD_LIBRARY_PATH="{{ cuda_home }}/lib64" flock "{{ gpu_lock }}" "$(just _tool speedup)" "{{ SRC }}" "{{ TGT }}" "{{ BRIDGE }}" {{ TOKENS }} {{ PROMPTS }} {{ REPS }} 2>/dev/null
