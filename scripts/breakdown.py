"""Per-domain and per-length scores for one screened model, plus its lowest-COMET sentences.

Usage: python official/scripts/breakdown.py --model qwen2.5-1.5b
"""
import argparse
import json
import re
from pathlib import Path

import sacrebleu
from comet import download_model, load_from_checkpoint

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "official" / "data" / "dev"
RES = ROOT / "official" / "results" / "screening"


def read(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def length_bucket(zh):
    n = len(re.sub(r"\s", "", zh))
    return "<=15" if n <= 15 else "16-40" if n <= 40 else "41+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", nargs="+", default=["dev_wmt", "dev_short"])
    ap.add_argument("--worst", type=int, default=10)
    a = ap.parse_args()
    comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    report = {}
    for dev in a.dev:
        src, ref = read(DEV / f"{dev}.zh"), read(DEV / f"{dev}.en")
        hyp = read(RES / a.model / f"{dev}.hyp")
        meta = [json.loads(l) for l in read(DEV / f"{dev}.meta.jsonl")]
        seg = comet.predict([{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)],
                            batch_size=64, gpus=1, progress_bar=False).scores
        groups = {}
        for i, (s, m) in enumerate(zip(src, meta)):
            groups.setdefault(f"domain:{m.get('domain', 'tatoeba')}", []).append(i)
            groups.setdefault(f"length:{length_bucket(s)}", []).append(i)
        rows = {}
        for g, ix in sorted(groups.items()):
            h, r = [hyp[i] for i in ix], [ref[i] for i in ix]
            rows[g] = dict(n=len(ix), bleu=round(sacrebleu.corpus_bleu(h, [r]).score, 2),
                           chrfpp=round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 2),
                           comet22=round(100 * sum(seg[i] for i in ix) / len(ix), 2))
        worst = sorted(range(len(seg)), key=lambda i: seg[i])[:a.worst]
        report[dev] = dict(groups=rows, worst=[dict(i=i, comet=round(seg[i], 3), src=src[i], hyp=hyp[i], ref=ref[i])
                                               for i in worst])
        (RES / a.model / f"{dev}.comet_seg.json").write_text(json.dumps([round(x, 4) for x in seg]), encoding="utf-8")
    (RES / a.model / "breakdown.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for dev, d in report.items():
        for g, r in d["groups"].items():
            print(dev, g, r)


if __name__ == "__main__":
    main()
