"""Flags misaligned pairs in the training corpus by scoring each Chinese line against its own and neighbouring English lines.

Usage (from official/): python scripts/1_data/audit_train_alignment.py [--gpu]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--margin", type=float, default=0.5)
ap.add_argument("--zh", default="data/train/train.zh-en.zh")
ap.add_argument("--en", default="data/train/train.zh-en.en")
a = ap.parse_args()
if not a.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from framework.scorers import build

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if a.gpu and torch.cuda.is_available() else "cpu"
FWD = {"type": "likelihood", "name": "opusmt.zh-en.fwd", "model": "Helsinki-NLP/opus-mt-zh-en", "batch_size": 64}
REV = {"type": "likelihood", "name": "opusmt.en-zh.rev", "model": "Helsinki-NLP/opus-mt-en-zh", "reverse": True,
       "prefix": ">>cmn_Hans<< ", "batch_size": 64}


def main():
    zh = (ROOT / a.zh).read_text(encoding="utf-8").split("\n")[:-1]
    en = (ROOT / a.en).read_text(encoding="utf-8").split("\n")[:-1]
    assert len(zh) == len(en)
    n = len(zh)
    pairs = [(zh[i], en[j]) for i in range(n) for j in (i - 1, i, i + 1) if 0 <= j < n]
    fwd, rev = build(FWD, DEVICE), build(REV, DEVICE)
    s = {p: f + r for p, f, r in zip(pairs, fwd.score(pairs), rev.score(pairs))}

    rows = []
    for i in range(n):
        own = s[(zh[i], en[i])]
        best_j, best = i, own
        for j in (i - 1, i + 1):
            if 0 <= j < n and s[(zh[i], en[j])] > best:
                best_j, best = j, s[(zh[i], en[j])]
        rows.append(dict(line=i + 1, own=round(own, 3), best_line=best_j + 1, best=round(best, 3),
                         gain=round(best - own, 3), flagged=best_j != i and best - own >= a.margin))

    flagged = [r for r in rows if r["flagged"]]
    blocks, cur = [], []
    for r in flagged:
        if cur and r["line"] - cur[-1]["line"] <= 2:
            cur.append(r)
        else:
            if cur:
                blocks.append(cur)
            cur = [r]
    if cur:
        blocks.append(cur)

    out = ROOT / "results" / "train_audit"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "flagged.tsv", "w", encoding="utf-8") as fh:
        fh.write("line\tbetter_matching_en_line\tgain\tzh\ten_own\ten_better\n")
        for r in flagged:
            i, j = r["line"] - 1, r["best_line"] - 1
            fh.write(f"{r['line']}\t{r['best_line']}\t{r['gain']}\t{zh[i]}\t{en[i]}\t{en[j]}\n")
    with open(out / "blocks.tsv", "w", encoding="utf-8") as fh:
        fh.write("first_line\tlast_line\tn_flagged\tdirection\n")
        for b in blocks:
            d = "en_shifted_up" if b[0]["best_line"] > b[0]["line"] else "en_shifted_down"
            fh.write(f"{b[0]['line']}\t{b[-1]['line']}\t{len(b)}\t{d}\n")
    summary = dict(lines=n, margin=a.margin, flagged=len(flagged), flagged_share=round(len(flagged) / n, 4),
                   blocks=len(blocks), lines_in_blocks_of_3_plus=sum(len(b) for b in blocks if len(b) >= 3),
                   largest_blocks=[(b[0]["line"], b[-1]["line"], len(b)) for b in sorted(blocks, key=len, reverse=True)[:15]])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
