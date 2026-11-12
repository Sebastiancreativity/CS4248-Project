"""Shared helpers for the pretrained Qwen candidate generators: model loading, prompt, output cleaning, greedy and epsilon-sampled decoding."""
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
}
PROMPT = "Translate the following Chinese text into English. Output only the English translation.\n\nChinese: {src}"
PREFIX = re.compile(r"^((here\s*(is|'s)\s*)?(the\s*)?(english\s*)?(translation|english)|英文|译文)\s*[:：]\s*", re.I)


def read(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def load(name):
    tok = AutoTokenizer.from_pretrained(MODELS[name], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.bfloat16).to("cuda")
    return tok, model.eval()


def build_prompt(tok, src):
    return tok.apply_chat_template([{"role": "user", "content": PROMPT.format(src=src)}],
                                   tokenize=False, add_generation_prompt=True)


def clean(raw, src):
    lines = [PREFIX.sub("", l.strip()) for l in raw.strip().split("\n")]
    lines = [l for l in lines if l]
    out = lines[0] if lines else ""
    if len(out) > 1 and out[0] == out[-1] and out[0] in "\"'“”" and src[:1] not in "\"“'‘「":
        out = out[1:-1].strip()
    return out


@torch.inference_mode()
def generate(tok, model, src, batch_size, max_new_tokens, n=1, eps=None):
    order = sorted(range(len(src)), key=lambda i: -len(src[i]))
    out = [None] * len(src)
    kw = dict(do_sample=True, epsilon_cutoff=eps, temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=n) \
        if eps else dict(do_sample=False, num_beams=1)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([build_prompt(tok, src[i]) for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, i in enumerate(idx):
            out[i] = dec[j * n:(j + 1) * n]
        print(f"  {min(b + batch_size, len(src))}/{len(src)}", end="\r", flush=True)
    print()
    return out
