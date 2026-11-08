"""Prepare data, compare fine-tuned systems on dev, freeze selection, then test."""
import argparse
import csv
import math
import shlex
import subprocess
import sys
from pathlib import Path

from build_train_splits import key
from finetune_common import (ROOT, adapter_record, fingerprint, generation_identity, load_run,
                             read_json, read_parallel, sha256, write_json)
from model_registry import TEAM_MODELS
from run_test_suite import read_assignments, read_test_data, write_text

SELECTORS = ("cometkiwi", "chrf", "bertscore")
VARIANTS = ("pretrained", "finetuned")
RECIPE_FILES = ["finetune_model.py", "finetune_data.py", "finetune_common.py",
                "finetune_pipeline.py", "model_registry.py", "score_finetuned_models.py",
                "screen_model.py", "run_ensembles.py", "run_test_suite.py", "build_train_splits.py",
                "bertscore_metric.py", "ensemble_cache.py", "translation_cleaning.py"]


def execute(command):
    print("+ " + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, cwd=ROOT)


def validate_disjoint(split_dir, official_dir):
    seen, counts = set(), {}
    official_sources = set()
    for filename in ("tatoeba.zh", "wmttest2022.zh"):
        official_sources.update(key(s) for s in (official_dir / filename).read_text(encoding="utf-8-sig").splitlines())
    for name in ("train", "dev_train20", "test_train10"):
        src, _ = read_parallel(split_dir, name)
        keys = [key(s) for s in src]
        if len(set(keys)) != len(keys) or set(keys) & (seen | official_sources):
            raise ValueError(f"Duplicate or overlapping source sentences in {name}")
        seen.update(keys)
        counts[name] = len(keys)
    return counts


def prepare(args):
    run_dir = args.run_dir.resolve()
    split_dir = ROOT / "data" / "splits_70_20_10"
    execute([sys.executable, ROOT / "scripts" / "build_train_splits.py"])
    counts = validate_disjoint(split_dir, ROOT)
    assignments = read_assignments(ROOT, None)
    assigned = {m for a in assignments for trio in a["combinations"] for m in trio}
    if assigned != set(TEAM_MODELS):
        raise ValueError("Assigned models differ from the eight supported training models")
    # Validate test file alignment now; no test metric is computed during preparation.
    tests = read_test_data("both", ROOT, split_dir)
    inputs = [split_dir / f"{split}.{lang}" for split in counts for lang in ("zh", "en")]
    inputs += [ROOT / name for name in ("train.zh-en.zh", "train.zh-en.en", "tatoeba.zh", "tatoeba.en",
                                         "wmttest2022.zh", "wmttest2022.AnnA.en", "combinations_team_manifest.json")]
    inputs += [a["path"] for a in assignments]
    spec = dict(
        split_dir=str(split_dir), models=list(TEAM_MODELS), counts=counts,
        tests={name: item["n"] for name, item in tests.items()},
        training=dict(epochs=args.epochs, learning_rate=args.learning_rate, lora_rank=16,
                      gradient_accumulation=16, seed=4248, max_source_length=512, max_target_length=512),
        evaluation=dict(generation_batch_size=4, max_new_tokens=512, metric_batch_size=16,
                        variants=list(VARIANTS), primary_metric="comet22",
                        tie_break="bleu then chrfpp then pretrained then assignment order"),
        input_sha256={str(p): sha256(p) for p in inputs},
        recipe_sha256={"scripts/" + name: sha256(ROOT / "scripts" / name) for name in RECIPE_FILES})
    spec["recipe_sha256"].update({name: sha256(ROOT / name) for name in
                                  ("requirements-cluster.txt", "requirements-finetune.txt")})
    manifest = {"spec": spec, "fingerprint": fingerprint(spec)}
    path = run_dir / "run.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Existing run has different data/settings/code. Choose a new run name.")
    write_json(path, manifest)
    print(f"Prepared {run_dir}: {counts}; 8 models, 40 combinations, 3 selectors, before/after comparison.")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def validated_results(results_dir, assignments, datasets):
    rows = []
    for assignment in assignments:
        owner = assignment["owner"]
        expected = {(i, dataset, selector) for i in range(1, len(assignment["combinations"]) + 1)
                    for dataset in datasets for selector in SELECTORS}
        seen = set()
        for row in read_csv(results_dir / owner / "scores_comparison.csv"):
            identity = (int(row["experiment"]), row["dev"], row["selector"])
            if identity not in expected or identity in seen:
                raise ValueError(f"Unexpected or duplicate result for {owner}: {identity}")
            if (int(row["n"]) != datasets[row["dev"]] or
                    row["models"] != " + ".join(assignment["combinations"][identity[0] - 1])):
                raise ValueError(f"Wrong data/model count in {owner}: {identity}")
            if not all(math.isfinite(float(row[m])) for m in ("comet22", "bleu", "chrfpp", "bertscore_f1")):
                raise ValueError(f"Nonfinite metric for {owner}: {identity}")
            seen.add(identity)
            rows.append({"owner": owner, **row})
        if seen != expected:
            raise ValueError(f"Incomplete evaluations for {owner}")
    return rows


def choose_on_dev(rows):
    if not rows or any(row["dev"] != "dev_train20" for row in rows):
        raise ValueError("Selection accepts dev_train20 results only; never test scores")
    rank = lambda r: tuple(float(r[m]) for m in ("comet22", "bleu", "chrfpp"))
    return {"primary": max(rows, key=rank),
            "per_selector": {s: max((r for r in rows if r["selector"] == s), key=rank) for s in SELECTORS}}


def paired_deltas(rows, keys):
    """Compare matched systems; positive deltas mean adaptation helped this metric."""
    grouped = {}
    for row in rows:
        group = tuple(row[k] for k in keys)
        variants = grouped.setdefault(group, {})
        if row["variant"] in variants:
            raise ValueError(f"Duplicate paired comparison: {group}")
        variants[row["variant"]] = row
    result = []
    for group, variants in grouped.items():
        if set(variants) != set(VARIANTS):
            raise ValueError(f"Missing pretrained/fine-tuned comparison: {group}")
        row = dict(zip(keys, group))
        for metric in ("comet22", "bleu", "chrfpp", "bertscore_f1"):
            before, after = [float(variants[v][metric]) for v in VARIANTS]
            row.update({f"pretrained_{metric}": before, f"finetuned_{metric}": after,
                        f"delta_{metric}": round(after - before, 4)})
        result.append(row)
    return result


def verify_selection(run_dir, run, identities):
    choice = read_json(run_dir / "selection.json")
    if (choice["run_fingerprint"] != run["fingerprint"] or choice["adapters"] != identities or
            choice["dev_scores_sha256"] != sha256(run_dir / "evaluation" / "dev" / "scores_comparison.csv")):
        raise ValueError("The frozen dev selection no longer matches this run")
    expected = choose_on_dev(read_csv(run_dir / "evaluation" / "dev" / "scores_comparison.csv"))
    if any(choice[name] != expected[name] for name in ("primary", "per_selector")):
        raise ValueError("Frozen selection was edited; do not select using test results")
    return choice


def evaluate(run_dir, stage):
    from filelock import FileLock
    with FileLock(str(run_dir / "evaluation.lock"), timeout=0):
        try:
            evaluate_locked(run_dir, stage)
        except Exception as exc:
            write_json(run_dir / "evaluation" / stage / "status.json", {"status": "failed", "error": str(exc)})
            raise


def evaluate_locked(run_dir, stage):
    run = load_run(run_dir)
    spec, split_dir = run["spec"], Path(run["spec"]["split_dir"])
    settings = spec["evaluation"]
    identities = {}
    for model in TEAM_MODELS:
        adapter = run_dir / "adapters" / model
        record = adapter_record(adapter, model)
        if record["run_fingerprint"] != run["fingerprint"]:
            raise ValueError(f"Wrong training run for {model}")
        identities[model] = generation_identity(model, adapter, settings["max_new_tokens"])
    if stage == "dev" and (run_dir / "selection.json").exists():
        verify_selection(run_dir, run, identities)
        print("Development evaluation and selection already complete; verified frozen choice.")
        return
    choice = verify_selection(run_dir, run, identities) if stage == "test" else None
    if stage == "dev":
        data_dir, datasets = split_dir, {"dev_train20": spec["counts"]["dev_train20"]}
    else:
        test_data = read_test_data("both", ROOT, split_dir)
        data_dir = run_dir / "test_data"
        datasets = {name: data["n"] for name, data in test_data.items()}
        for name, data in test_data.items():
            write_text(data_dir / f"{name}.zh", data["source"])
            write_text(data_dir / f"{name}.en", data["reference"])
    results_dir = run_dir / "evaluation" / stage
    results_dir.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a complete-looking combined result.
    (results_dir / "scores_comparison.csv").unlink(missing_ok=True)
    (results_dir / "final_selected_scores.csv").unlink(missing_ok=True)
    write_json(results_dir / "status.json", {"status": "running", "datasets": datasets})
    from run_ensembles import valid_cache
    assignments = read_assignments(ROOT, None)
    rows, individuals = [], []
    for variant in VARIANTS:
        base_only = variant == "pretrained"
        variant_args = ["--base-only"] if base_only else []
        candidates = run_dir / "candidates" / variant
        for model in TEAM_MODELS:
            adapter = run_dir / "adapters" / model
            identity = generation_identity(model, adapter, settings["max_new_tokens"], base_only)
            out = candidates / model / "output"
            missing = [name for name in datasets
                       if not valid_cache(out, name, read_parallel(data_dir, name)[0], identity)]
            if missing:
                execute([sys.executable, ROOT / "scripts" / "screen_model.py", "--model", model,
                         "--adapter-dir", adapter, "--data-dir", data_dir,
                         "--dev", *missing, "--output-dir", out, "--no-comet", "--batch-size",
                         settings["generation_batch_size"], "--max-new-tokens", settings["max_new_tokens"]] + variant_args)
        execute([sys.executable, ROOT / "scripts" / "score_finetuned_models.py", "--run-dir", run_dir,
                 "--stage", stage, "--variant", variant, "--data-dir", data_dir, "--datasets", *datasets])
        individuals.extend(read_csv(results_dir / variant / "individual" / "scores.csv"))
        for a in assignments:
            execute([sys.executable, ROOT / "scripts" / "run_ensembles.py", "--combinations", a["path"],
                     "--owner", a["owner"], "--adapter-root", run_dir / "adapters", "--model-outputs-dir",
                     candidates, "--results-dir", results_dir / variant, "--data-dir", data_dir,
                     "--datasets", *datasets, "--selector", "all", "--comet", "--bertscore",
                     "--bertscore-device", "cuda", "--bertscore-batch-size", settings["metric_batch_size"],
                     "--max-new-tokens", settings["max_new_tokens"], "--no-generate"] + variant_args)
        rows.extend({"variant": variant, **r} for r in validated_results(results_dir / variant, assignments, datasets))
    expected_individuals = {(v, m, d) for v in VARIANTS for m in TEAM_MODELS for d in datasets}
    if (len(individuals) != len(expected_individuals) or
            {(r["variant"], r["model"], r["dataset"]) for r in individuals} != expected_individuals or
            any(int(r["n"]) != datasets[r["dataset"]] for r in individuals) or
            any(not math.isfinite(float(r[m])) for r in individuals
                for m in ("comet22", "bleu", "chrfpp", "bertscore_f1"))):
        raise ValueError("Missing or invalid individual model evaluations")
    save_csv(results_dir / "individual_scores.csv", individuals)
    save_csv(results_dir / "individual_deltas.csv", paired_deltas(individuals, ("model", "dataset")))
    save_csv(results_dir / "ensemble_deltas.csv", paired_deltas(rows, ("owner", "experiment", "models", "dev", "selector")))
    if stage == "test":
        primary = choice["primary"]
        for row in rows:
            row["dev_selected_primary"] = all(row[k] == primary[k] for k in ("variant", "owner", "models", "selector"))
        primary_rows = [r for r in rows if r["dev_selected_primary"]]
        save_csv(results_dir / "final_selected_scores.csv", primary_rows)
    save_csv(results_dir / "scores_comparison.csv", rows)
    if stage == "dev":
        choice = {**choose_on_dev(rows), "run_fingerprint": run["fingerprint"], "adapters": identities,
                  "selection_rule": "highest dev COMET22; ties: BLEU, chrF++, pretrained, assignment order",
                  "dev_scores_sha256": sha256(results_dir / "scores_comparison.csv")}
        write_json(run_dir / "selection.json", choice)
        print(f"Frozen development choice: {choice['primary']['variant']} / "
              f"{choice['primary']['models']} / {choice['primary']['selector']}")
    write_json(results_dir / "status.json", {"status": "complete", "datasets": datasets, "rows": len(rows)})
    print(f"Completed {stage}: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "dev", "test", "status"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3, help="prepare only")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="prepare only")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.learning_rate < 1:
        parser.error("Epochs and learning rate must be positive")
    args.run_dir = args.run_dir.resolve()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "status":
        run = load_run(args.run_dir)
        for model in TEAM_MODELS:
            path = args.run_dir / "adapters" / model
            if (path / "training_complete.json").exists():
                record = adapter_record(path, model)
                print(f"{model}: trained, steps={record['global_step']}, best dev loss={record['best_dev_loss']:.5f}")
            else:
                print(f"{model}: pending/incomplete")
        for stage in ("dev", "test"):
            path = args.run_dir / "evaluation" / stage / "status.json"
            print(f"{stage}: {read_json(path) if path.exists() else 'pending'}")
    else:
        evaluate(args.run_dir, args.stage)


if __name__ == "__main__":
    main()
