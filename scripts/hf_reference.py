"""A model's own reference logits from Hugging Face transformers — what the Rust decoder must
reproduce for a model family it has not run before.

Loads <model dir> (config.json, tokenizer.json, model.safetensors) in f32 on the CPU, renders <text>
as one ChatML user turn followed by the assistant header (crates/core/src/text.rs chat_prompt), and
writes the token ids and the last position's logits to <out>.safetensors.

A model too large for f32 in host memory takes a fourth argument, bf16: its reference is then
transformers' own bf16 logits.

usage: python hf_reference.py <model dir> <out stem> <text> [bf16]
"""
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

path, stem, text = sys.argv[1], sys.argv[2], sys.argv[3]
dtype = torch.bfloat16 if sys.argv[4:] == ["bf16"] else torch.float32
tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).eval()
with torch.no_grad():
    logits = model(ids).logits[0, -1].float().contiguous()
save_file({"ids": ids[0].to(torch.float32), "logits": logits}, stem + ".safetensors")
print(f"{ids.shape[1]} tokens; argmax {int(logits.argmax())} -> {stem}.safetensors")
