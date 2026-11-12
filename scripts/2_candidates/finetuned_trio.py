"""Generates greedy translations from the fine-tuned ensemble (Hunyuan-MT-7B, Qwen2.5-7B, TowerInstruct-Mistral-7B with the team's LoRA adapters) on the sets the team did not cover.

Uses the team's prompts and cleaner from the ensemble repository (--team-dir); 4-bit NF4, greedy, at most 512 new tokens.
Usage (from official/): python scripts/2_candidates/finetuned_trio.py [--models ...] [--sets ...]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
TRIO = ["hunyuan-mt-7b", "qwen2.5-7b", "tower-mistral-7b"]
SETS = {"dev_short": "data/dev/dev_short.zh", "test_wmt22": "data/test/wmttest2022.zh",
        "test_tatoeba": "data/test/tatoeba.zh", "test_train10": "data/test/test_train10.zh"}


def load(name, adapters, models):
    cfg, adir = dict(models[name]), adapters / name
    rec = json.loads((adir / "training_complete.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(adir), padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], dtype=torch.bfloat16, quantization_config=q,
                                                 device_map="cuda", revision=rec["base_revision"])
    return cfg, tok, PeftModel.from_pretrained(model, str(adir), is_trainable=False).eval()


@torch.inference_mode()
def translate(tok, model, cfg, src, build_prompt, batch_size, max_new_tokens):
    order = sorted(range(len(src)), key=lambda i: -len(src[i]))
    raw = [None] * len(src)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([build_prompt(tok, cfg, src[i]) for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        enc.pop("token_type_ids", None)
        gen = model.generate(**enc, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             temperature=None, top_p=None, top_k=None)
        for i, g in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            raw[i] = g
        print(f"  {min(b + batch_size, len(src))}/{len(src)}", end="\r", flush=True)
    print()
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=str(ROOT.parent / "team2" / "best-ensemble-ft702010-v1"))
    ap.add_argument("--models", nargs="+", default=TRIO)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()
    team = Path(a.team_dir)
    sys.path.insert(0, str(team / "scripts"))
    from model_registry import MODELS, build_prompt
    from translation_cleaning import clean
    adapters = team / "artifacts" / "finetune" / "ft702010_v1" / "adapters"
    out_dir = ROOT / "results" / "candidates"
    for name in a.models:
        todo = [s for s in a.sets if not (out_dir / f"ft.{name}.{s}.txt").exists()]
        if not todo:
            continue
        cfg, tok, model = load(name, adapters, MODELS)
        for s in todo:
            src = (ROOT / SETS[s]).read_text(encoding="utf-8").splitlines()
            t0 = time.time()
            raw = translate(tok, model, cfg, src, build_prompt, a.batch_size, a.max_new_tokens)
            hyp = [clean(r, x) for r, x in zip(raw, src)]
            (out_dir / f"ft.{name}.{s}.txt").write_text("\n".join(hyp) + "\n", encoding="utf-8")
            print(json.dumps(dict(model=name, set=s, n=len(src), seconds=round(time.time() - t0, 1))), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
