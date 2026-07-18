#!/usr/bin/env python3
"""Produce tests/data/hf_golden.json: HF transformers reference logits and
greedy continuation for the fixed e2e prompt, using the local models/gpt2
checkpoint so Forge and HF read identical weights."""

import json
import pathlib

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "Hello, my dog is cute"
GREEDY_TOKENS = 15

tok = GPT2TokenizerFast.from_pretrained(ROOT / "models/gpt2")
model = GPT2LMHeadModel.from_pretrained(ROOT / "models/gpt2", torch_dtype=torch.float32)
model.eval()

ids = tok(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits[0, -1, :].tolist()
    greedy = model.generate(
        ids, max_new_tokens=GREEDY_TOKENS, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )[0].tolist()

out = {
    "prompt": PROMPT,
    "prompt_ids": ids[0].tolist(),
    "logits_last": logits,
    "greedy_ids": greedy,
}
dest = ROOT / "tests/data/hf_golden.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out))
print(f"wrote {dest} ({dest.stat().st_size} bytes)")
print("greedy text:", tok.decode(greedy))
