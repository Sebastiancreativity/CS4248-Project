"""Generate missing model outputs, then score three-model translation ensembles.

Edit combinations.json and run: python scripts/run_ensembles.py --combinations combinations.json
By default both reference-free COMETKiwi and chrF++ agreement select outputs
from the same candidates. References are used only afterward for evaluation.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from cache_paths import COMET_CACHE
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "dev"
SCREEN = ROOT / "results" / "screening"
MODEL_OUTPUTS = SCREEN / "ensemble"
OUT_BASE = ROOT / "results" / "ensembles"
TEAM_MEMBERS = ["Ming Zhe", "Theodore", "Reet", "Shri"]


def read(path):
    return path.read_text(encoding="utf-8").splitlines()


def valid_cache(output_dir, dev, src):
    raw = output_dir / f"{dev}.raw.jsonl"
    hyp = output_dir / f"{dev}.hyp"
    if not raw.exists() or not hyp.exists():
        return False
    lines = read(hyp)
    if len(lines) != len(src):
        return False
    try:
        records = [json.loads(line) for line in read(raw)]
    except (ValueError, OSError):
        return False
    return len(records) == len(src) and all(r.get("src") == s for r, s in zip(records, src))


def choose_chrf(candidates):
    # A missing or empty translation cannot win unless all three are empty.
    if not any(candidates):
        return 0
    scores = []
    for i, text in enumerate(candidates):
        if not text:
            scores.append(float("-inf"))
            continue
        scores.append(sum(sacrebleu.sentence_chrf(text, [other], word_order=2).score
                          for j, other in enumerate(candidates) if i != j) / 2)
    return max(range(3), key=lambda i: scores[i])


def qe_scores(dev, model_name, src, hyp, qe_model):
    cache = OUT_BASE / ".cache" / "qe_cache" / model_name / f"{dev}.json"
    digest = hashlib.sha256(json.dumps([src, hyp], ensure_ascii=False).encode("utf-8")).hexdigest()
    if cache.exists():
        record = json.loads(cache.read_text(encoding="utf-8"))
        if record.get("digest") == digest and len(record.get("scores", [])) == len(src):
            return record["scores"]
    data = [{"src": s, "mt": h} for s, h in zip(src, hyp)]
    scores = [float(x) for x in qe_model.predict(data, batch_size=32, gpus=1,
                                                  progress_bar=False).scores]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"digest": digest, "scores": scores}), encoding="utf-8")
    return scores


def score(src, ref, hyp, comet_model):
    metrics = {
        "bleu": round(sacrebleu.corpus_bleu(hyp, [ref], tokenize="13a").score, 2),
        "chrfpp": round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 2),
    }
    if comet_model is not None:
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
        metrics["comet22"] = round(100 * comet_model.predict(data, batch_size=64, gpus=1,
                                                               progress_bar=False).system_score, 2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combinations", type=Path, default=ROOT / "combinations.json")
    parser.add_argument("--dev", nargs="+", choices=["dev_wmt", "dev_short"],
                        default=["dev_wmt", "dev_short"])
    parser.add_argument("--comet", action="store_true", help="also score with reference-based COMET")
    parser.add_argument("--selector", choices=["both", "cometkiwi", "chrf"], default="both",
                        help="candidate selection method; default runs both")
    parser.add_argument("--no-generate", action="store_true", help="require all model outputs to exist")
    parser.add_argument("--owner", choices=TEAM_MEMBERS, required=True,
                        help="required team member folder for this experiment run")
    args = parser.parse_args()

    from screen_model import MODELS
    combos = json.loads(args.combinations.read_text(encoding="utf-8"))
    if not isinstance(combos, list) or not combos:
        parser.error("combinations must be a nonempty array")
    for i, trio in enumerate(combos, 1):
        if (not isinstance(trio, list) or len(trio) != 3 or
                any(not isinstance(m, str) or m not in MODELS for m in trio) or
                len(set(trio)) != 3):
            parser.error(f"combination {i} must contain three distinct supported model names")
    if len({tuple(sorted(x)) for x in combos}) != len(combos):
        parser.error("duplicate combinations (order does not change an ensemble)")

    sources = {dev: read(DEV / f"{dev}.zh") for dev in args.dev}
    refs = {dev: read(DEV / f"{dev}.en") for dev in args.dev}
    for dev in args.dev:
        if len(sources[dev]) != len(refs[dev]):
            parser.error(f"source/reference line mismatch in {dev}")

    models = list(dict.fromkeys(m for trio in combos for m in trio))
    for model in models:
        output_dir = MODEL_OUTPUTS / model / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Migrate usable outputs from the original screening location on demand.
        legacy_dir = SCREEN / model
        for dev in args.dev:
            if not valid_cache(output_dir, dev, sources[dev]) and valid_cache(legacy_dir, dev, sources[dev]):
                shutil.copyfile(legacy_dir / f"{dev}.hyp", output_dir / f"{dev}.hyp")
                shutil.copyfile(legacy_dir / f"{dev}.raw.jsonl", output_dir / f"{dev}.raw.jsonl")
        missing = [dev for dev in args.dev if not valid_cache(output_dir, dev, sources[dev])]
        if missing:
            if args.no_generate:
                parser.error(f"missing or partial outputs for {model}: {', '.join(missing)}")
            print(f"Generating {model}: {', '.join(missing)}", flush=True)
            result = subprocess.run([sys.executable, "-X", "faulthandler", "-u",
                                     str(ROOT / "scripts" / "screen_model.py"),
                                     "--model", model, "--dev", *missing, "--no-comet",
                                     "--output-dir", str(output_dir)])
            if result.returncode:
                code = result.returncode & 0xFFFFFFFF
                detail = " (Windows access violation)" if code == 0xC0000005 else ""
                parser.exit(1, f"{model} failed with exit code 0x{code:08X}{detail}. "
                               f"Native diagnostics: {ROOT / 'logs' / (model + '-native-crash.log')}\n")

    OUT = OUT_BASE / args.owner
    OUT.mkdir(parents=True, exist_ok=True)
    all_hyp = {dev: {m: read(MODEL_OUTPUTS / m / "output" / f"{dev}.hyp") for m in models}
               for dev in args.dev}
    all_qe = None
    if args.selector in ("both", "cometkiwi"):
        from comet import download_model, load_from_checkpoint
        qe_model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da",
                                                       saving_directory=str(COMET_CACHE)))
        all_qe = {dev: {m: qe_scores(dev, m, sources[dev], all_hyp[dev][m], qe_model)
                        for m in models} for dev in args.dev}
        del qe_model
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    comet_model = None
    if args.comet:
        from comet import download_model, load_from_checkpoint
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da",
                                                          saving_directory=str(COMET_CACHE)))
    selectors = ["cometkiwi", "chrf"] if args.selector == "both" else [args.selector]
    all_rows = []
    for selector in selectors:
        rows = []
        for i, trio in enumerate(combos, 1):
            for dev in args.dev:
                if selector == "cometkiwi":
                    choices = [max(range(3), key=lambda k: all_qe[dev][trio[k]][j]
                                   if all_hyp[dev][trio[k]][j] else float("-inf"))
                               for j in range(len(sources[dev]))]
                else:
                    choices = [choose_chrf([all_hyp[dev][m][j] for m in trio])
                               for j in range(len(sources[dev]))]
                hyp = [all_hyp[dev][trio[pick]][j] for j, pick in enumerate(choices)]
                exp_dir = OUT / f"experiment_{i:02d}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                hyp_path = exp_dir / f"{dev}_{selector}.hyp"
                hyp_path.write_text("\n".join(hyp) + "\n", encoding="utf-8")
                counts = {m: choices.count(k) for k, m in enumerate(trio)}
                row = {"experiment": i, "dev": dev, "models": " + ".join(trio),
                       "selector": selector,
                       "n": len(hyp), **score(sources[dev], refs[dev], hyp, comet_model),
                       "selection_counts": json.dumps(counts), "hyp_file": str(hyp_path)}
                rows.append(row)
                print(f"{i:02d} {dev} {selector}: {row['models']} | BLEU {row['bleu']} | chrF++ {row['chrfpp']}" +
                      (f" | COMET {row['comet22']}" if args.comet else ""), flush=True)

        columns = list(rows[0])
        with (OUT / f"scores_{selector}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (OUT / f"scores_{selector}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        for i in range(1, len(combos) + 1):
            exp_dir = OUT / f"experiment_{i:02d}"
            experiment_rows = [r for r in all_rows + rows if r["experiment"] == i]
            (exp_dir / "scores.json").write_text(json.dumps(experiment_rows, indent=2), encoding="utf-8")
        all_rows.extend(rows)
        for dev in args.dev:
            top = max((r for r in rows if r["dev"] == dev),
                      key=lambda r: r.get("comet22", r["chrfpp"]))
            print(f"Best {selector} on {dev}: experiment {top['experiment']} ({top['models']})", flush=True)
    if args.selector == "both":
        with (OUT / "scores_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
