"""Publish the frozen ensemble and a dev-selected baseline before the full grid.

Evaluate both pretrained and fine-tuned variants of the same chosen ensemble
and individual model. This is four systems on three test sets, selected using
dev only. All generated candidates remain reusable by the existing full sweep.
"""
import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path
import sys

import finetune_pipeline as pipeline
from finetune_common import (ROOT, adapter_record, fingerprint, generation_identity, load_run,
                             read_json, read_parallel, sha256, write_json)
from model_registry import TEAM_MODELS

METRICS = ("comet22", "bleu", "chrfpp", "bertscore_f1")


def select_baseline(rows, n):
    expected = {(v, m, "dev_train20") for v in pipeline.VARIANTS for m in TEAM_MODELS}
    if (len(rows) != len(expected)
            or {(r["variant"], r["model"], r["dataset"]) for r in rows} != expected
            or any(int(r["n"]) != n or not all(math.isfinite(float(r[k])) for k in METRICS) for r in rows)):
        raise ValueError("Baseline selection requires every individual model's completed dev scores")
    return max(rows, key=lambda r: (float(r["comet22"]), float(r["bleu"]), float(r["chrfpp"]),
                                   r["variant"] == "pretrained", -TEAM_MODELS.index(r["model"])))


def prepare(run_dir):
    run = load_run(run_dir)
    settings = run["spec"]["evaluation"]
    identities = {}
    for model in TEAM_MODELS:
        adapter = run_dir / "adapters" / model
        if adapter_record(adapter, model)["run_fingerprint"] != run["fingerprint"]:
            raise ValueError(f"Adapter belongs to another run: {model}")
        identities[model] = generation_identity(model, adapter, settings["max_new_tokens"])
    choice = pipeline.verify_selection(run_dir, run, identities)
    source = run_dir / "evaluation/dev/individual_scores.csv"
    baseline = select_baseline(pipeline.read_csv(source), run["spec"]["counts"]["dev_train20"])
    trio = choice["primary"]["models"].split(" + ")
    if len(trio) != 3 or len(set(trio)) != 3 or any(m not in TEAM_MODELS for m in trio):
        raise ValueError("Invalid frozen ensemble")
    plan = {"run_fingerprint": run["fingerprint"], "adapters": identities,
            "ensemble": choice["primary"], "trio": trio, "individual": baseline,
            "dev_scores_sha256": choice["dev_scores_sha256"], "individual_dev_sha256": sha256(source),
            "driver_sha256": sha256(Path(__file__))}
    plan["fingerprint"] = fingerprint(plan)
    output = run_dir / "evaluation/priority_test"
    previous = output / "plan.json"
    if previous.exists() and read_json(previous) != plan:
        raise ValueError("The frozen priority test plan changed; do not mix different selections or recipes")
    write_json(previous, plan)
    # This never changes selection.json or uses test scores for selection.
    tests = pipeline.read_test_data("both", ROOT, Path(run["spec"]["split_dir"]))
    datasets = {name: data["n"] for name, data in tests.items()}
    if datasets != run["spec"]["tests"]:
        raise ValueError("Test datasets differ from the prepared run")
    data_dir = run_dir / "test_data"
    for name, data in tests.items():
        pipeline.write_text(data_dir / f"{name}.zh", data["source"])
        pipeline.write_text(data_dir / f"{name}.en", data["reference"])
    return run, plan, output, data_dir, datasets


def generate_needed(run_dir, run, plan, data_dir, datasets):
    from run_ensembles import valid_cache
    settings = run["spec"]["evaluation"]
    models = list(dict.fromkeys(plan["trio"] + [plan["individual"]["model"]]))
    for variant in pipeline.VARIANTS:
        for model in models:
            adapter = run_dir / "adapters" / model
            identity = generation_identity(model, adapter, settings["max_new_tokens"], variant == "pretrained")
            output = run_dir / "candidates" / variant / model / "output"
            missing = [name for name in datasets
                       if not valid_cache(output, name, read_parallel(data_dir, name)[0], identity)]
            if missing:
                pipeline.execute([sys.executable, ROOT / "scripts/screen_model.py", "--model", model,
                                  "--adapter-dir", adapter, "--data-dir", data_dir, "--dev", *missing,
                                  "--output-dir", output, "--no-comet", "--batch-size", settings["generation_batch_size"],
                                  "--max-new-tokens", settings["max_new_tokens"]]
                                 + (["--base-only"] if variant == "pretrained" else []))
            if any(not valid_cache(output, name, read_parallel(data_dir, name)[0], identity) for name in datasets):
                raise ValueError(f"Incomplete generation: {variant} {model}")


def score_individuals(run_dir, run, plan, data_dir, datasets):
    """Write checkpoints compatible with score_finetuned_models.py for reuse."""
    from run_ensembles import score, valid_cache
    from bertscore_metric import EnglishBERTScore
    from cache_paths import COMET_CACHE
    from comet import download_model, load_from_checkpoint
    from translation_cleaning import clean

    settings = run["spec"]["evaluation"]
    model = plan["individual"]["model"]
    software = {name: version(name) for name in ["torch", "transformers", "sacrebleu", "bert-score", "unbabel-comet"]}
    scorer = EnglishBERTScore(device="cuda", batch_size=settings["metric_batch_size"])
    rows, comet_model = [], None
    for variant in pipeline.VARIANTS:
        identity = generation_identity(model, run_dir / "adapters" / model,
                                       settings["max_new_tokens"], variant == "pretrained")
        cached = run_dir / "candidates" / variant / model / "output"
        for dataset in datasets:
            src, ref = read_parallel(data_dir, dataset)
            if not valid_cache(cached, dataset, src, identity):
                raise ValueError(f"Missing baseline candidates: {variant} {model} {dataset}")
            raw = [json.loads(line) for line in (cached / f"{dataset}.raw.jsonl").read_text(encoding="utf-8").splitlines()]
            hyp = [clean(r["raw"], r["src"]) for r in raw]
            key = fingerprint([identity, src, ref, hyp, software, settings,
                               sha256(ROOT / "scripts/score_finetuned_models.py"), sha256(ROOT / "scripts/run_ensembles.py")])
            checkpoint = run_dir / "evaluation/test" / variant / "individual" / f"{model}-{dataset}.json"
            previous = read_json(checkpoint) if checkpoint.exists() else {}
            if previous.get("fingerprint") == key:
                row = previous["row"]
            else:
                if comet_model is None:
                    comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da", saving_directory=str(COMET_CACHE)))
                row = {"variant": variant, "model": model, "dataset": dataset, "n": len(src),
                       **score(src, ref, hyp, comet_model, scorer)}
                write_json(checkpoint, {"fingerprint": key, "row": row})
            rows.append(row)
            print(f"Baseline complete: {variant} {model} {dataset}", flush=True)
    return rows


def report_row(kind, variant, models, selector, dataset, n, metrics, plan):
    if not all(math.isfinite(float(metrics[k])) for k in METRICS):
        raise ValueError("Invalid test metric")
    return {"system_type": kind, "variant": variant, "models": models, "selector": selector,
            "dataset": dataset, "n": n,
            "dev_selected_primary": kind == "ensemble" and variant == plan["ensemble"]["variant"],
            "dev_selected_individual": kind == "individual" and variant == plan["individual"]["variant"],
            **{k: metrics[k] for k in METRICS}}


def evaluate_locked(run_dir):
    run, plan, output, data_dir, datasets = prepare(run_dir)
    settings = run["spec"]["evaluation"]
    artifacts = ("report_scores.csv", "report_deltas.csv", "final_selected_scores.csv")
    status_path = output / "status.json"
    previous = read_json(status_path) if status_path.exists() else {}
    if (previous.get("status") == "complete" and previous.get("plan") == plan["fingerprint"]
            and all((output / name).exists() and sha256(output / name) == previous.get("output_sha256", {}).get(name)
                    for name in artifacts)):
        print(f"Verified completed report results: {output / 'report_scores.csv'}", flush=True)
        return
    write_json(output / "status.json", {"status": "running", "plan": plan["fingerprint"]})
    # A failed retry cannot masquerade as a completed aggregate report.
    for name in artifacts:
        (output / name).unlink(missing_ok=True)
    generate_needed(run_dir, run, plan, data_dir, datasets)
    combos = output / "frozen_combination.json"
    write_json(combos, [plan["trio"]])
    rows = []
    selector, owner = plan["ensemble"]["selector"], plan["ensemble"]["owner"]
    for variant in pipeline.VARIANTS:
        results = output / variant / "ensemble"
        pipeline.execute([sys.executable, ROOT / "scripts/run_ensembles.py", "--combinations", combos,
                          "--owner", owner, "--adapter-root", run_dir / "adapters",
                          "--model-outputs-dir", run_dir / "candidates" / variant,
                          "--results-dir", results, "--data-dir", data_dir, "--datasets", *datasets,
                          "--selector", selector, "--comet", "--bertscore", "--bertscore-device", "cuda",
                          "--bertscore-batch-size", settings["metric_batch_size"], "--max-new-tokens",
                          settings["max_new_tokens"], "--no-generate"]
                         + (["--base-only"] if variant == "pretrained" else []))
        scored = pipeline.read_csv(results / owner / f"scores_{selector}.csv")
        if (len(scored) != len(datasets) or {r["dev"] for r in scored} != set(datasets)
                or any(int(r["experiment"]) != 1 or r["models"] != " + ".join(plan["trio"])
                       or r["selector"] != selector or int(r["n"]) != datasets[r["dev"]] for r in scored)):
            raise ValueError(f"Missing or incorrect {variant} ensemble test results")
        rows.extend(report_row("ensemble", variant, r["models"], selector, r["dev"], int(r["n"]), r, plan) for r in scored)
    individuals = score_individuals(run_dir, run, plan, data_dir, datasets)
    expected = {(v, plan["individual"]["model"], d) for v in pipeline.VARIANTS for d in datasets}
    if (len(individuals) != len(expected)
            or {(r["variant"], r["model"], r["dataset"]) for r in individuals} != expected
            or any(int(r["n"]) != datasets[r["dataset"]] for r in individuals)):
        raise ValueError("Missing or incorrect individual baseline results")
    rows.extend(report_row("individual", r["variant"], r["model"], "none", r["dataset"], int(r["n"]), r, plan)
                for r in individuals)
    # Recheck integrity after computation before marking results complete.
    current = load_run(run_dir)
    pipeline.verify_selection(run_dir, current, plan["adapters"])
    if sha256(run_dir / "evaluation/dev/individual_scores.csv") != plan["individual_dev_sha256"]:
        raise ValueError("Individual dev results changed during testing")
    for model in dict.fromkeys(plan["trio"] + [plan["individual"]["model"]]):
        if generation_identity(model, run_dir / "adapters" / model, settings["max_new_tokens"]) != plan["adapters"][model]:
            raise ValueError(f"Adapter changed during testing: {model}")
    pipeline.save_csv(output / "report_scores.csv", rows)
    pipeline.save_csv(output / "report_deltas.csv",
                      pipeline.paired_deltas(rows, ("system_type", "models", "selector", "dataset")))
    pipeline.save_csv(output / "final_selected_scores.csv", [r for r in rows if r["dev_selected_primary"]])
    write_json(output / "status.json", {"status": "complete", "rows": len(rows), "datasets": datasets,
                                        "plan": plan["fingerprint"],
                                        "output_sha256": {name: sha256(output / name) for name in artifacts}})
    print(f"Report results ready: {output / 'report_scores.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status"])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "evaluation/priority_test"
    if args.action == "status":
        path = output / "status.json"
        print(read_json(path) if path.exists() else "Priority test results: pending")
        print(f"Report: {output / 'report_scores.csv'}")
        return
    from filelock import FileLock
    with FileLock(str(run_dir / "evaluation.lock"), timeout=0):
        try:
            evaluate_locked(run_dir)
        except Exception as exc:
            write_json(output / "status.json", {"status": "failed", "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
