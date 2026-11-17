# Prattle — the browser demo and the server that run LatentPort bridges and FAST_LLM files.
#
#     just web      # the page → target/site/
#     just run      # the server on 0.0.0.0:8192, serving the page, /compare and /chat
#     just test     # every test that needs no downloaded model
#     just models && just bridge-fetch && just speedup models/SmolLM2-135M-Instruct models/SmolLM2-360M-Instruct models/bridges/smollm2-135m-to-360m.safetensors
#     just cuda && just oracle && just oracle-python && just oracle-home && just reproduce   # LatentPort's own table

_default:
    @just --list

import 'oracle.just'

# Start the server, the page built first.
run *ARGS:
    #!/usr/bin/env sh
    set -e
    just web
    cargo run -p server --release -- {{ ARGS }}

test:
    cargo test --workspace

# Build the page into target/site/, which prattle-server serves.
#
# The wasm is PUBLISHED, and a Rust binary embeds the source path of every panic site — this
# checkout, the toolchain, and the crate registry's directory, whose name can carry a mirror's.
# Each prefix is rewritten to a neutral one at compile time.
web:
    #!/usr/bin/env sh
    set -e
    remap="--remap-path-prefix=$(pwd)=/prattle --remap-path-prefix=$HOME/.rustup=/rustup"
    for d in "$HOME"/.cargo/registry/src/*/ "$HOME"/.cargo/git/checkouts/*/; do
        [ -d "$d" ] && remap="$remap --remap-path-prefix=${d%/}=/cargo"
    done
    RUSTFLAGS="$remap" wasm-pack --quiet build crates/web --release --target web --no-pack --out-dir ../../target/site/pkg
    RUSTFLAGS="$remap" wasm-pack --quiet build crates/webgpu --release --target web --no-pack --out-dir ../../target/site/pkg-webgpu
    cp crates/web/www/* target/site/
    # The WebGPU twin is the same page, told by one meta tag to load the WebGPU engine: one source.
    sed 's|<meta charset="utf-8">|<meta charset="utf-8">\n<meta name="prattle-backend" content="webgpu">|' crates/web/www/index.html > target/site/webgpu.html
