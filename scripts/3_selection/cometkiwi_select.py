"""COMETKiwi selection over the fine-tuned ensemble (the team's final selector): per sentence, the candidate with the highest wmt22-cometkiwi-da score wins.

Empty candidates score -inf and ties go to the first system. Writes the final submission files.
Usage (from official/): python scripts/3_selection/cometkiwi_select.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIO = ["hunyuan-mt-7b", "qwen2.5-7b", "tower-mistral-7b"]
SETS = {"dev_train20": "data/dev/dev_train20.zh", "dev_short": "data/dev/dev_short.zh",
        "test_wmt22": "data/test/wmttest2022.zh", "test_tatoeba": "data/test/tatoeba.zh",
        "test_train10": "data/test/test_train10.zh"}
SUBMIT = {"test_wmt22": "wmt22.en", "test_tatoeba": "tatoeba.en", "test_train10": "internal.en"}


def read(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def main():
    from comet import download_model, load_from_checkpoint
    model = None
    scores_dir = ROOT / "results" / "cometkiwi_scores"
    sel_dir = ROOT / "results" / "ensemble_selectors"
    sub = ROOT / "results" / "submission" / "ensemble_cometkiwi"
    for d in (scores_dir, sel_dir, sub):
        d.mkdir(parents=True, exist_ok=True)
    summary = {}
    for s, src_path in SETS.items():
        src = read(ROOT / src_path)
        hyp = {m: read(ROOT / "results" / "candidates" / f"ft.{m}.{s}.txt") for m in TRIO}
        qe = {}
        for m in TRIO:
            f = scores_dir / f"{m}.{s}.json"
            if not f.exists():
                model = model or load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
                pred = model.predict([{"src": x, "mt": h} for x, h in zip(src, hyp[m])], batch_size=32, gpus=1,
                                     progress_bar=False).scores
                f.write_text(json.dumps([float(v) for v in pred]), encoding="utf-8")
            qe[m] = json.loads(f.read_text(encoding="utf-8"))
        picks = [max(range(3), key=lambda k: qe[TRIO[k]][j] if hyp[TRIO[k]][j] else float("-inf"))
                 for j in range(len(src))]
        out = "\n".join(hyp[TRIO[k]][j] for j, k in enumerate(picks)) + "\n"
        (sel_dir / f"{s}.cometkiwi.txt").write_text(out, encoding="utf-8")
        if s in SUBMIT:
            (sub / SUBMIT[s]).write_text(out, encoding="utf-8")
        summary[s] = {m: picks.count(k) for k, m in enumerate(TRIO)}
        print(s, summary[s], flush=True)
    (sub / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
