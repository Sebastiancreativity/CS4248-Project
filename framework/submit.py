"""Tunes the selector on development data, applies it once to a test pool and writes the outputs and decision records.

Usage (from official/): python -m framework.submit --config framework/configs/submit_ensemble_framework.json [--gpu]
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
if not args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import sacrebleu
import torch

from . import filters
from .data import OFFICIAL, Sentence, load_dev, load_pool, read_lines
from .evaluate import comet_scores
from .features import calibrate, featurise
from .scorers import build
from .selector import choose, scores, tune

torch.set_num_threads(args.threads)
DEVICE = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"


def resolve(p):
    return p if os.path.isabs(p) else str(OFFICIAL / p)


def main():
    cfg = json.loads(open(args.config, encoding="utf-8").read())
    scorers = {k: build(v, DEVICE) for k, v in cfg["scorers"].items()}
    out_dir = OFFICIAL / "results" / "submission" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(name=cfg["name"], method=cfg["method"], targets={})
    for tgt in cfg["targets"]:
        t0 = time.time()
        print(f"== {tgt['name']}")
        sents = []
        for dev in tgt["tune_dev"]:
            sents += load_pool(load_dev(dev), dev, {k: v.format(dev=dev) for k, v in tgt["dev_systems"].items()})
        n_dev = len(sents)
        src = read_lines(resolve(tgt["test_src"]))
        test = load_pool([Sentence(i, s, "", {}) for i, s in enumerate(src)], "test",
                         {k: resolve(v) for k, v in tgt["test_systems"].items()})
        assert all(s.cands for s in test), "every test sentence needs at least one candidate"
        dev_ids = list(range(n_dev))
        band = filters.length_band(sents, dev_ids)
        n_fallback = 0
        if tgt.get("fallback_systems"):
            fb = load_pool([Sentence(i, s, "", {}) for i, s in enumerate(src)], "test",
                           {k: resolve(v) for k, v in tgt["fallback_systems"].items()})
            for s, f in zip(test, fb):
                if all(filters.tag(s.src, c.text, band)[0] & filters.HARD for c in s.cands):
                    s.cands += f.cands
                    n_fallback += 1
        sents += test
        for k, s in enumerate(sents):
            s.i = k
        test_ids = list(range(n_dev, len(sents)))
        F = featurise(sents, dev_ids + test_ids, scorers, cfg["features"], band)
        F = calibrate(F, dev_ids, cfg.get("calibrate"))
        for name in cfg.get("system_priors", []):
            for i in F:
                F[i]["sys_" + name] = (F[i]["sys"] == name).astype(float)
        for feat in tgt.get("indicators", {}):
            lines = []
            for dev in tgt["tune_dev"]:
                lines += read_lines(resolve(tgt["indicators"][feat]["dev"].format(dev=dev)))
            lines += read_lines(resolve(tgt["indicators"][feat]["test"]))
            for i in F:
                F[i][feat] = np.array([float(c.text == lines[i]) for c in sents[i].cands])
        texts ={i: [c.text for c in sents[i].cands] for i in F}
        refs = {i: sents[i].ref for i in dev_ids}
        tuned = tune(F, texts, refs, dev_ids, cfg["method"])
        w, guard, T, aux = tuned["weights"], tuned["guard"], tuned["T"], tuned["aux"]
        print(f"  tuned on {len(dev_ids)} dev sentences: {w} (dev chrF++ {tuned['tune_chrfpp']})")

        pick = {i: choose(F[i], w, guard, T, aux) for i in test_ids}
        hyps = [texts[i][pick[i]] for i in test_ids]
        (out_dir / tgt["out"]).write_text("\n".join(hyps) + "\n", encoding="utf-8")

        decisions, share, hard_left = [], {}, 0
        for i in test_ids:
            j = pick[i]
            c = sents[i].cands[j]
            tags, detail = filters.tag(sents[i].src, c.text, band)
            s = scores(F[i], w, T, aux)
            hard_left += bool(F[i]["hard"][j])
            key = f"{c.system}:{c.kind.rstrip('0123456789')}"
            share[key] = share.get(key, 0) + 1
            decisions.append(dict(i=i - n_dev, src=sents[i].src, chosen=dict(system=c.system, kind=c.kind, text=c.text),
                                  tags=sorted(tags), detail=detail, n_cands=len(texts[i]),
                                  excluded_by_guard=int(F[i]["hard"].sum()) if (~F[i]["hard"]).any() else 0,
                                  features={f: round(float(F[i][f][j]), 4) for f in w if f in F[i]} |
                                           {"mbr": round(float(F[i]["mbr"][j]), 4)},
                                  score=round(float(s[j]), 4)))
        (out_dir / f"{tgt['name']}.decisions.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in decisions) + "\n", encoding="utf-8")

        info = dict(dev=tgt["tune_dev"], n_dev=len(dev_ids), n_test=len(test_ids), weights=w,
                    dev_chrfpp=tuned["tune_chrfpp"], systems=list(tgt["test_systems"]), picked=dict(sorted(share.items())),
                    chosen_with_hard_tag=hard_left, fallback_added=n_fallback, out=str(out_dir / tgt["out"]))
        if tgt.get("test_ref"):
            ref = read_lines(resolve(tgt["test_ref"]))
            seg = comet_scores([(s, h, r) for s, h, r in zip(src, hyps, ref)])
            first = [texts[i][0] for i in test_ids]
            seg0 = comet_scores([(s, h, r) for s, h, r in zip(src, first, ref)])
            info["test_scored_once"] = dict(
                bleu=round(sacrebleu.corpus_bleu(hyps, [ref]).score, 2),
                chrfpp=round(sacrebleu.corpus_chrf(hyps, [ref], word_order=2).score, 2),
                comet22=round(100 * float(np.mean(seg)), 2),
                first_candidate=dict(bleu=round(sacrebleu.corpus_bleu(first, [ref]).score, 2),
                                     comet22=round(100 * float(np.mean(seg0)), 2)))
        info["seconds"] = round(time.time() - t0, 1)
        summary["targets"][tgt["name"]] = info
        print(json.dumps(info, ensure_ascii=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
