"""Scores every system on the WMT22 test set and the internal test set (BLEU, chrF++, COMET-22, Chinese outputs, omission share) with paired bootstrap tests.

Usage (from official/): python scripts/4_evaluation/score_test_sets.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.data import Sentence
from framework.evaluate import bootstrap, evaluate
from framework.filters import length_band
from framework.text import HAN

SUB = "results/submission"
SETS = {
    "wmt22": ("data/test/wmttest2022.zh", "data/test/wmttest2022.AnnA.en", "test_wmt22", "wmt22.en"),
    "internal": ("data/test/test_train10.zh", "data/test/test_train10.en", "test_train10", "internal.en"),
}
SELECTORS = {"cometkiwi": "ensemble_cometkiwi", "consensus": "ensemble_consensus", "framework": "ensemble_framework"}


def read(p):
    return (ROOT / p).read_text(encoding="utf-8").splitlines()


def score(src_path, ref_path, systems):
    sents = [Sentence(i=i, src=s, ref=r, meta={}) for i, (s, r) in enumerate(zip(read(src_path), read(ref_path)))]
    ids = list(range(len(sents)))
    band = length_band(sents, ids)
    table, seg = {}, {}
    for name, path in systems.items():
        hyp = read(path)
        m, seg[name] = evaluate(sents, ids, hyp, band)
        table[name] = dict(bleu=m["bleu"], chrfpp=m["chrfpp"], comet22=m["comet22"], omission=m["omission"],
                           chinese=sum(bool(HAN.search(h)) for h in hyp))
        print(f"{name:18s} {m['bleu']:6.2f} {m['chrfpp']:6.2f} {m['comet22']:6.2f}")
    boot = {f"{a}>{b}": bootstrap(seg[b], seg[a]) for a in SELECTORS for b in systems if b != a}
    return dict(scores=table, bootstrap=boot)


def main():
    out = ROOT / "results" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    for name, (src, ref, tag, hyp) in SETS.items():
        systems = {m: f"results/candidates/ft.{m}.{tag}.txt" for m in ["hunyuan-mt-7b", "qwen2.5-7b", "tower-mistral-7b"]}
        systems |= {k: f"{SUB}/{v}/{hyp}" for k, v in SELECTORS.items()}
        if name == "wmt22":
            systems["qwen3b_framework"] = f"{SUB}/qwen3b_selfpool_test/wmt22.en"
            greedy = [json.loads(l)["cands"][0]["text"] for l in read("results/candidates/qwen2.5-3b.test_wmt22.jsonl")]
            tmp = out / "qwen3b_greedy.wmt22.en"
            tmp.write_text("\n".join(greedy) + "\n", encoding="utf-8")
            systems["qwen3b_greedy"] = str(tmp.relative_to(ROOT))
        print(f"== {name}")
        (out / f"{name}_table.json").write_text(json.dumps(score(src, ref, systems), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
