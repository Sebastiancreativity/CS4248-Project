"""Diagnostics of the auditor signals: natural-error detection, repair-trigger precision and the controlled error-injection matrix.

Usage (from official/): python -m framework.probe --config framework/configs/qwen1.5b.json
"""
import argparse
import json
import os
import random
import re

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--dev", nargs="*")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
if not args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from .data import OFFICIAL, load_dev, load_pool, split
from .evaluate import OMISSION, classify, comet_scores
from .repair import uncovered_spans
from .scorers import build
from .taxonomy import PROPER, length_band as ref_band

torch.set_num_threads(args.threads)
NUM = re.compile(r"\d+")


def auroc(score, label):
    s, y = -np.asarray(score, float), np.asarray(label, bool)
    if y.all() or not y.any():
        return None
    r = s.argsort().argsort() + 1
    return round(float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())), 4)


def lift(score, label, frac=0.10):
    y = np.asarray(label, bool)
    k = max(1, int(frac * len(y)))
    p = float(y[np.argsort(score)[:k]].mean())
    return dict(base=round(float(y.mean()), 4), precision=round(p, 4), lift=round(p / max(y.mean(), 1e-9), 2))


def delete_span(t, frac=0.3):
    w = t.split()
    if len(w) < 8:
        return None
    k = max(1, round(frac * len(w)))
    s = (len(w) - k) // 2
    return " ".join(w[:s] + w[s + k:])


def swap_name(t, names, rng):
    found = [m for m in PROPER.findall(t) if m in t]
    if not found:
        return None
    old = found[0]
    choices = [n for n in names if n != old and n.split()[0] != old.split()[0]]
    return t.replace(old, rng.choice(choices), 1) if choices else None


def change_number(t, src, rng):
    cands = [m for m in NUM.finditer(t) if m.group() in src]
    if not cands:
        return None
    m = rng.choice(cands)
    v = int(m.group())
    new = str(v + rng.choice([1, 2, 3]) if v < 10 else v + rng.choice([-1, 1]) * max(1, v // 10))
    return t[:m.start()] + new + t[m.end():]


def insert_phrase(t, others, rng, k=5):
    w = t.split()
    donors = [o.split() for o in others if len(o.split()) >= 2 * k and o != t]
    if len(w) < 8 or not donors:
        return None
    d = rng.choice(donors)
    s = rng.randrange(0, len(d) - k)
    m = len(w) // 2
    return " ".join(w[:m] + d[s:s + k] + w[m:])


def pairwise(orig, damaged):
    o, d = np.asarray(orig, float), np.asarray(damaged, float)
    return round(float((o > d).mean() + 0.5 * (o == d).mean()), 4)


def main():
    cfg = json.loads(open(args.config, encoding="utf-8").read())
    scorers = {k: build(v) for k, v in cfg["scorers"].items()}
    lik = {f: scorers[s] for f, s in cfg["features"].items()}
    cov = scorers[cfg["repair"]["coverage"]]
    rng = random.Random(4248)
    for dev in args.dev or cfg["dev"]:
        sents = load_pool(load_dev(dev), dev, cfg.get("systems"))
        _, ev = split(sents)
        eband = ref_band([s.src for s in sents], [s.ref for s in sents])
        src = [sents[i].src for i in ev]
        hyp = [sents[i].cands[0].text for i in ev]
        ref = [sents[i].ref for i in ev]
        pairs = list(zip(src, hyp))

        tags = [classify(s, r, h, eband) for s, r, h in zip(src, ref, hyp)]
        seg = np.asarray(comet_scores(list(zip(src, hyp, ref))))
        labels = {"omission": [bool(t & OMISSION) for t in tags],
                  "entity_missing": ["entity_missing" in t for t in tags],
                  "number_dropped": ["number_dropped" in t for t in tags],
                  "low_comet_20pct": list(seg <= np.quantile(seg, 0.2))}
        cov_res = cov.score(pairs)
        signals = {f: sc.score(pairs) for f, sc in lik.items()}
        signals["coverage"] = [c["pen"] for c in cov_res]
        detection = {name: {lab: dict(auroc=auroc(v, y), **lift(v, y)) for lab, y in labels.items()}
                     for name, v in signals.items()}

        triggers = {}
        for tau in (0.05, 0.1, 0.2, 0.3):
            fired = [bool(uncovered_spans(c, tau)) for c in cov_res]
            n = sum(fired)
            triggers[str(tau)] = dict(rate=round(n / len(fired), 4),
                                      precision_omission=round(sum(f and y for f, y in zip(fired, labels["omission"])) / max(n, 1), 4),
                                      base_omission=round(float(np.mean(labels["omission"])), 4))

        names = sorted({m for h in hyp for m in PROPER.findall(h)})
        damage = {"span_deletion": [delete_span(h) for h in hyp],
                  "name_swap": [swap_name(h, names, rng) for h in hyp],
                  "number_change": [change_number(h, s, rng) for s, h in zip(src, hyp)]}
        rng_ins = random.Random(4249)
        damage["phrase_insertion"] = [insert_phrase(h, hyp, rng_ins) for h in hyp]
        matrix = {}
        for kind, dmg in damage.items():
            idx = [k for k, d in enumerate(dmg) if d and d != hyp[k]]
            orig_p = [pairs[k] for k in idx]
            dmg_p = [(src[k], dmg[k]) for k in idx]
            row = {"n": len(idx)}
            for f, sc in lik.items():
                row[f] = pairwise(sc.score(orig_p), sc.score(dmg_p))
            row["coverage"] = pairwise([c["pen"] for c in cov.score(orig_p)], [c["pen"] for c in cov.score(dmg_p)])
            matrix[kind] = row
            ex = idx[0] if idx else None
            row["example"] = dict(orig=hyp[ex], damaged=dmg[ex]) if ex is not None else None

        out = OFFICIAL / "results" / "framework" / cfg["name"] / dev
        out.mkdir(parents=True, exist_ok=True)
        res = dict(dev=dev, n_eval=len(ev), detection=detection, coverage_triggers=triggers, injection=matrix)
        (out / "probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
