"""Runs a development experiment from a JSON config: tunes each method on the tuning half and reports it on the evaluation half.

Usage (from official/): python -m framework.run --config framework/configs/qwen1.5b.json [--gpu]
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--dev", nargs="*")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--skip-repair", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="first N sentences only (smoke tests)")
ap.add_argument("--name", default=None, help="override the output name")
args = ap.parse_args()
if not args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from sacrebleu.metrics import CHRF

from . import filters
from .data import Candidate, OFFICIAL, load_dev, load_pool, split
from .evaluate import bootstrap, evaluate
from .features import calibrate, featurise
from .repair import HFChatGenerator, problems, repair, uncovered_spans
from .scorers import build
from .selector import choose, tune
from .taxonomy import length_band as ref_band

torch.set_num_threads(args.threads)
DEVICE = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"


def run_methods(sents, F, tune_ids, eval_ids, methods, eband):
    texts = {i: [c.text for c in sents[i].cands] for i in F}
    refs = {i: sents[i].ref for i in F}
    chrf = CHRF(word_order=2)
    results, segs, picks = {}, {}, {}
    for spec in methods:
        t0 = time.time()
        if spec.get("type") == "first":
            tuned, pick = None, {i: 0 for i in eval_ids}
        elif spec.get("type") == "system":
            tuned = None
            pick = {i: next(k for k, c in enumerate(sents[i].cands) if c.system == spec["system"]) for i in eval_ids}
        elif spec.get("type") == "file":
            tuned, lines = None, spec["lines"]
            pick = {i: texts[i].index(lines[i]) for i in eval_ids}
        elif spec.get("type") == "fixed":
            tuned = spec["tuned"]
            pick = {i: choose(F[i], tuned["weights"], tuned["guard"], tuned["T"], tuned["aux"]) for i in eval_ids}
        elif spec.get("type") == "oracle":
            tuned = None
            pick = {i: int(np.argmax([chrf.sentence_score(t, [refs[i]]).score for t in texts[i]])) for i in eval_ids}
        else:
            tuned = tune(F, texts, refs, tune_ids, spec)
            pick = {i: choose(F[i], tuned["weights"], tuned["guard"], tuned["T"], tuned["aux"]) for i in eval_ids}
        hyps = [texts[i][pick[i]] for i in eval_ids]
        metrics, seg = evaluate(sents, eval_ids, hyps, eband)
        sysshare = {}
        for i in eval_ids:
            c = sents[i].cands[pick[i]]
            sysshare[f"{c.system}:{c.kind.rstrip('0123456789')}"] = sysshare.get(f"{c.system}:{c.kind.rstrip('0123456789')}", 0) + 1
        results[spec["name"]] = dict(metrics=metrics, tuned={k: v for k, v in (tuned or {}).items() if k != "grid"},
                                     picked=dict(sorted(sysshare.items())), seconds=round(time.time() - t0, 1))
        segs[spec["name"]], picks[spec["name"]] = seg, pick
        print(f"  {spec['name']:<34} {metrics['bleu']:6.2f} {metrics['chrfpp']:6.2f} {metrics['comet22']:6.2f}"
              f"  {(tuned or {}).get('weights', '')}")
    for name in results:
        for base in ("greedy", "mbr", "team_cometkiwi"):
            if base in segs and name != base:
                results[name].setdefault("beats", {})[base] = bootstrap(segs[base], segs[name])
    return results, segs, picks


def main():
    cfg = json.loads(open(args.config, encoding="utf-8").read())
    out_root = OFFICIAL / "results" / "framework" / (args.name or cfg["name"])
    scorers = {k: build(v, DEVICE) for k, v in cfg["scorers"].items()}
    for dev in args.dev or cfg["dev"]:
        print(f"== {cfg['name']} / {dev}")
        sents = load_pool(load_dev(dev), dev, cfg.get("systems"))
        if args.limit:
            sents = sents[:args.limit]
        tune_ids, eval_ids = split(sents)
        band = filters.length_band(sents, tune_ids)
        eband = ref_band([s.src for s in sents], [s.ref for s in sents])
        F = featurise(sents, tune_ids + eval_ids, scorers, cfg["features"], band)
        F = calibrate(F, tune_ids, cfg.get("calibrate"))
        for feat, path in cfg.get("indicators", {}).items():
            lines = (OFFICIAL / path.format(dev=dev)).read_text(encoding="utf-8").splitlines()
            for i in F:
                F[i][feat] = np.array([float(c.text == lines[i]) for c in sents[i].cands])
        for name in cfg.get("system_priors", []):
            for i in F:
                F[i]["sys_" + name] = (F[i]["sys"] == name).astype(float)
        for m in cfg["methods"]:
            if m.get("type") == "file":
                m["lines"] = (OFFICIAL / m["path"].format(dev=dev)).read_text(encoding="utf-8").splitlines()
        results, segs, picks = run_methods(sents, F, tune_ids, eval_ids, cfg["methods"], eband)

        rep = cfg.get("repair")
        decisions = []
        if rep and not args.skip_repair:
            base = next(m for m in cfg["methods"] if m["name"] == rep["base"])
            w = results[base["name"]]["tuned"]
            chosen = {i: picks[base["name"]][i] for i in eval_ids}
            cov = scorers[rep["coverage"]].score([(sents[i].src, sents[i].cands[chosen[i]].text) for i in chosen])
            todo = []
            for (i, j), cv in zip(chosen.items(), cov):
                draft = sents[i].cands[j].text
                tags, detail = filters.tag(sents[i].src, draft, band)
                spans = uncovered_spans(cv, rep.get("tau", 0.3)) if rep.get("use_coverage", True) else []
                probs = problems(tags, detail, spans)
                decisions.append(dict(i=i, src=sents[i].src,
                                      chosen=dict(system=sents[i].cands[j].system, kind=sents[i].cands[j].kind,
                                                  text=draft),
                                      tags=sorted(tags), uncovered=spans, problems=probs))
                if probs:
                    todo.append((i, draft, probs))
            print(f"  repair triggered on {len(todo)}/{len(chosen)} sentences")
            gen = HFChatGenerator(rep["generator"], device=DEVICE, batch_size=rep.get("batch_size", 8))
            fixed = repair(gen, [(sents[i].src, d, p) for i, d, p in todo])
            del gen
            for (i, _, _), t in zip(todo, fixed):
                sents[i].cands.append(Candidate("repair", "repair", t))
            touched = [i for i, _, _ in todo]
            F = calibrate(featurise(sents, touched, scorers, cfg["features"], band, F), tune_ids, cfg.get("calibrate"))
            spec = dict(name=base["name"] + "+repair", type="fixed", tuned=w)
            r2, s2, p2 = run_methods(sents, F, tune_ids, eval_ids, [spec], eband)
            name = spec["name"]
            r2[name]["beats"] = {"mbr": bootstrap(segs["mbr"], s2[name]),
                                 base["name"]: bootstrap(segs[base["name"]], s2[name])}
            trig = [k for k, i in enumerate(eval_ids) if i in set(touched)]
            adopted = [k for k in trig if sents[eval_ids[k]].cands[p2[name][eval_ids[k]]].system == "repair"]
            r2[name]["repair"] = dict(
                triggered_eval=len(trig), adopted_eval=len(adopted),
                comet_on_triggered=dict(before=round(100 * float(np.mean([segs[base['name']][k] for k in trig])), 2),
                                        after=round(100 * float(np.mean([s2[name][k] for k in trig])), 2)),
                comet_on_adopted=dict(before=round(100 * float(np.mean([segs[base['name']][k] for k in adopted])), 2),
                                      after=round(100 * float(np.mean([s2[name][k] for k in adopted])), 2))
                if adopted else None)
            results.update(r2)
            picks.update(p2)
            by_i = {d["i"]: d for d in decisions}
            for (i, _, _), t in zip(todo, fixed):
                by_i[i]["repair_output"] = t
            for k, i in enumerate(eval_ids):
                c = sents[i].cands[p2[name][i]]
                by_i[i]["final"] = dict(system=c.system, kind=c.kind, text=c.text)

        out = out_root / dev
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(dict(dev=dev, n_tune=len(tune_ids), n_eval=len(eval_ids),
                                                          systems=cfg.get("systems"), results=results),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
        for name, pick in picks.items():
            (out / f"{name}.eval.hyp").write_text("\n".join(sents[i].cands[pick[i]].text for i in eval_ids) + "\n",
                                                  encoding="utf-8")
        if decisions:
            (out / "decisions.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in decisions) + "\n",
                                                 encoding="utf-8")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
