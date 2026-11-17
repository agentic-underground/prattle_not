[prattle_not](../README.md) / [documents](README.md) / **FAST_LLM**

# FAST_LLM — one file holding a small model, a large model and the bridge between them

The small model reads the prompt, the bridge translates its reading, the large model answers. One
download carries all three. The reader is [`crates/core/src/fast_llm.rs`](../crates/core/src/fast_llm.rs);
the server's **/compare** and the page both load it. The bridge itself is LatentPort's idea —
Simon Villani's work ([`LatentPort/latentport`](https://github.com/LatentPort/latentport)).

## The file

**Plain safetensors.** Any safetensors reader can open it and list its tensors.

| tensor name | what it holds |
|---|---|
| `source.<name>` | the small model's tensors, each with its official bytes, unchanged |
| `target.<name>` | the large model's tensors, likewise |
| `bridge.<name>` | the bridge's tensors |
| `target.q.<name>` | a PRE-QUANTIZED file only: one of the large model's linear weights as raw GGML blocks (`U8`) |

A multimodal checkpoint contributes its **language model only**: when any name carries
`language_model.`, only those names are kept.

## The header's `__metadata__`

| key | value |
|---|---|
| `fast_llm` | the manifest, as JSON — below |
| `source.config.json` | the small model's `config.json`, verbatim |
| `target.config.json` | the large model's `config.json`, verbatim |
| `tokenizer.json` | the small model's `tokenizer.json` — the one the pair reads with |
| `target.tokenizer.json` | only when the large model's tokenizer differs from the small one's |

## The manifest

```json
{
  "format": "fast_llm/1",
  "source": { "name": "SmolLM2-135M-Instruct", "files": [ { "name": "model.safetensors", "bytes": …, "sha256": "…" } ] },
  "target": { "name": "SmolLM2-360M-Instruct", "files": [ … ] },
  "bridge": { "name": "smollm2-135m-to-360m.safetensors", "bytes": …, "sha256": "…" },
  "method": "LatentPort (Simon Villani), reproduced in Rust by PrattleNot"
}
```

- **`format`** is checked before anything else. A reader refuses a format it does not read, by name.
- **`source` / `target`** name each model and list every file it was packed from — `config.json`,
  `tokenizer.json` and each `.safetensors` — with its size and sha256, as its publisher's own listing
  shows them. No local path is ever written.
- **`bridge`** names the bridge file with its size and sha256.

### Three kinds of file, one format

| kind | what changes |
|---|---|
| **FULL** | everything above; both models' tensors are in the file |
| **THIN** | a model part carries `"ref": { "repo", "revision", "files": [ { "name", "bytes", "sha256" } ] }` in place of its `files`, and none of its tensors. A reader fetches each file from that Hugging Face repo at that pinned revision, and **refuses it unless its size and sha256 match** — before a byte of it is loaded. The configs and the tokenizer still travel in the header |
| **PRE-QUANTIZED** | the manifest carries `"quantized": { "ggml": "q4k" \| "q8_0", "shapes": { "<weight>": [dims…] } }`; each named weight is stored as `target.q.<weight>` blocks, exactly as quantizing at load would make them, and its `target.<weight>` is absent. A reader uses the blocks as stored and never quantizes again |

## Reading one

```rust
let parts = prattle_core::fast_llm::FastLlm::unpack(&bytes)?;             // FULL or PRE-QUANTIZED
let parts = parts.resolve(&prattle_core::fast_llm::hub_fetch)?;           // THIN: fetched and checked
let config = parts.get("target.config.json")?;
```

`unpack` returns each model's weights and the bridge as safetensors buffers of their own, prefixes
removed, beside the metadata. A file given as a TARGET is refused unless its `target.config.json` is
the same model as the folder it stands in for — a measurement against another model's weights would
report numbers about the wrong model.
