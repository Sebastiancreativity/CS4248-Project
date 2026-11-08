"""Score individual adapted models from verified, already-generated candidates."""
import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

from finetune_common import (adapter_record, fingerprint, generation_identity, load_run,
                             read_json, read_parallel, sha256, write_json)
from model_registry import TEAM_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["dev", "test"], required=True)
    parser.add_argument("--variant", choices=["pretrained", "finetuned"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    run = load_run(args.run_dir)
    settings = run["spec"]["evaluation"]
    from run_ensembles import score, valid_cache
    from bertscore_metric import EnglishBERTScore
    from cache_paths import COMET_CACHE
    from comet import download_model, load_from_checkpoint
    from translation_cleaning import clean

    output = args.run_dir / "evaluation" / args.stage / args.variant / "individual"
    output.mkdir(parents=True, exist_ok=True)
    rows, comet_model = [], None
    scorer = EnglishBERTScore(device="cuda", batch_size=settings["metric_batch_size"])
    software = {name: version(name) for name in ["torch", "transformers", "sacrebleu", "bert-score", "unbabel-comet"]}
    for model in TEAM_MODELS:
        adapter = args.run_dir / "adapters" / model
        record = adapter_record(adapter, model)
        if record["run_fingerprint"] != run["fingerprint"]:
            raise ValueError(f"Adapter belongs to another run: {model}")
        identity = generation_identity(model, adapter, settings["max_new_tokens"], args.variant == "pretrained")
        cached = args.run_dir / "candidates" / args.variant / model / "output"
        for dataset in args.datasets:
            src, ref = read_parallel(args.data_dir, dataset)
            if not valid_cache(cached, dataset, src, identity):
                raise ValueError(f"Missing or stale fine-tuned output: {model} {dataset}")
            raw = [json.loads(line) for line in (cached / f"{dataset}.raw.jsonl").read_text(encoding="utf-8").splitlines()]
            hyp = [clean(r["raw"], r["src"]) for r in raw]
            key = fingerprint([identity, src, ref, hyp, software, settings,
                               sha256(Path(__file__)), sha256(Path(__file__).with_name("run_ensembles.py"))])
            checkpoint = output / f"{model}-{dataset}.json"
            previous = read_json(checkpoint) if checkpoint.exists() else {}
            if previous.get("fingerprint") == key:
                row = previous["row"]
            else:
                if comet_model is None:
                    comet_model = load_from_checkpoint(download_model(
                        "Unbabel/wmt22-comet-da", saving_directory=str(COMET_CACHE)))
                row = {"variant": args.variant, "model": model, "dataset": dataset, "n": len(src),
                       **score(src, ref, hyp, comet_model, scorer)}
                write_json(checkpoint, {"fingerprint": key, "row": row})
            rows.append(row)
            print(f"Individual {model} {dataset}: {row}", flush=True)
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
