"""Generates 1-best translations from pretrained NMT models (OPUS-MT zh-en, NLLB-200-distilled-600M) for the mixed pool and the guard fallback.

Usage (from official/):
  python scripts/2_candidates/nmt_pool.py --system nllb-600m --dev dev_wmt dev_alma
  python scripts/2_candidates/nmt_pool.py --system nllb-600m --src data/test/wmttest2022.zh --name test_wmt22
"""
import argparse
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=["opus-mt-zh-en", "nllb-600m"])
ap.add_argument("--dev", nargs="+", default=["dev_wmt", "dev_alma"])
ap.add_argument("--src", default=None)
ap.add_argument("--name", default=None)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=16)
a = ap.parse_args()
if not a.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = {
    "opus-mt-zh-en": dict(model="Helsinki-NLP/opus-mt-zh-en", tok={}, gen={}),
    "nllb-600m": dict(model="facebook/nllb-200-distilled-600M", tok={"src_lang": "zho_Hans"}, gen={"lang": "eng_Latn"}),
}


def main():
    torch.set_num_threads(a.threads)
    cfg = SYSTEMS[a.system]
    device = "cuda" if a.gpu and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model"], **cfg["tok"])
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"], dtype=dtype).to(device).eval()
    extra = {"forced_bos_token_id": tok.convert_tokens_to_ids(cfg["gen"]["lang"])} if "lang" in cfg["gen"] else {}
    out = ROOT / "results" / "candidates"
    jobs = [(Path(a.src), out / f"{a.system}.{a.name}.txt")] if a.src else [
        (ROOT / "data" / "dev" / f"{d}.zh", out / f"{a.system}.{d}.txt") for d in a.dev]
    for src_path, path in jobs:
        dev = src_path.stem
        src = src_path.read_text(encoding="utf-8").splitlines()
        order = sorted(range(len(src)), key=lambda i: -len(src[i]))
        out = [None] * len(src)
        with torch.inference_mode():
            for b in range(0, len(order), a.batch_size):
                idx = order[b:b + a.batch_size]
                enc = tok([src[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                          max_length=512).to(device)
                gen = model.generate(**enc, num_beams=5, max_new_tokens=256, **extra)
                for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                    out[i] = t.strip()
                print(f"  {dev} {min(b + a.batch_size, len(src))}/{len(src)}", end="\r", flush=True)
        print()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
