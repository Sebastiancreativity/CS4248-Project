"""Builds a single-model candidate pool: one greedy and N epsilon-sampled Qwen2.5 translations per source sentence.

Usage (from official/):
  python scripts/2_candidates/sample_pool.py --model qwen2.5-1.5b --src data/dev/dev_wmt.zh --name dev_wmt --n 8
  python scripts/2_candidates/sample_pool.py --model qwen2.5-3b --src data/test/wmttest2022.zh --name test_wmt22 --n 4
"""
import argparse
import json
import time
from pathlib import Path

import torch

from llm import MODELS, clean, generate, load, read

OUT = Path(__file__).resolve().parents[2] / "results" / "candidates"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=4248)
    a = ap.parse_args()
    src = read(a.src)
    tok, model = load(a.model)
    t0 = time.time()
    greedy = generate(tok, model, src, 16, a.max_new_tokens)
    torch.manual_seed(a.seed)
    samples = generate(tok, model, src, a.batch_size, a.max_new_tokens, a.n, a.eps) if a.n else [[]] * len(src)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.model}.{a.name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i, (s, g, ss) in enumerate(zip(src, greedy, samples)):
            cands = [{"kind": "greedy", "text": clean(g[0], s)}] + \
                    [{"kind": f"sample{k}", "text": clean(r, s)} for k, r in enumerate(ss)]
            fh.write(json.dumps({"i": i, "src": s, "model": a.model, "cands": cands}, ensure_ascii=False) + "\n")
    meta = dict(model=a.model, src=a.src, n_samples=a.n, epsilon=a.eps, seed=a.seed, seconds=round(time.time() - t0, 1))
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
