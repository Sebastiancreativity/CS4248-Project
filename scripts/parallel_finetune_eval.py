"""Resume the existing dev/test recipe with independent Slurm GPU workers.

The original, fingerprinted recipe files stay unchanged. Generation workers
own one model/variant each. Ensemble workers use private copies of candidates
and metric caches because the original runner rewrites both. Only collection
can freeze the dev choice or publish complete combined results.
"""
import argparse
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import finetune_pipeline as pipeline
from bertscore_metric import DEFAULT_MODEL
from ensemble_cache import digest
from finetune_common import (ROOT, adapter_record, fingerprint, generation_identity,
                             load_run, read_json, read_parallel, sha256, write_json)
from model_registry import TEAM_MODELS


@contextmanager
def evaluation_lock(run_dir, shared=False, filename="evaluation.lock"):
    """Use the same flock as the original pipeline; workers may share it."""
    path = run_dir / filename
    if os.name == "nt":
        from filelock import FileLock
        with FileLock(str(path), timeout=0):
            yield
        return
    import fcntl
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluation is active. Stop the original job before switching launchers.") from exc
        # Inherit this lock into recipe children so killing only the coordinator
        # cannot allow another writer while its child still uses the candidates.
        os.set_inheritable(handle.fileno(), True)
        # Close rather than explicitly unlock: inherited descriptors keep the
        # lock alive if a recipe child outlives its parent process.
        yield handle.fileno()


def execute(command, lock_fd=None):
    import shlex
    import subprocess
    print("+ " + shlex.join(map(str, command)), flush=True)
    descriptors = lock_fd if isinstance(lock_fd, tuple) else (lock_fd,)
    kwargs = {"pass_fds": tuple(fd for fd in descriptors if fd is not None)} if os.name != "nt" else {}
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True, **kwargs)


def workspace(run_dir, stage):
    return run_dir / "parallel_evaluation" / stage


def verified_adapters(run_dir, run):
    identities = {}
    for model in TEAM_MODELS:
        adapter = run_dir / "adapters" / model
        if adapter_record(adapter, model)["run_fingerprint"] != run["fingerprint"]:
            raise ValueError(f"Adapter belongs to another run: {model}")
        identities[model] = generation_identity(model, adapter, run["spec"]["evaluation"]["max_new_tokens"])
    return identities


def seed_json(source, destination):
    """Copy usable checkpoints only; interrupted JSON writes are not results."""
    if not source.exists():
        return
    for path in source.rglob("*.json"):
        target = destination / path.relative_to(source)
        if target.exists():
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError):
            continue
        write_json(target, value)


def initialize(run_dir, stage):
    run = load_run(run_dir)
    identities = verified_adapters(run_dir, run)
    spec = run["spec"]
    if stage == "test":
        pipeline.verify_selection(run_dir, run, identities)
        tests = pipeline.read_test_data("both", ROOT, Path(spec["split_dir"]))
        data_dir = run_dir / "test_data"
        datasets = {name: data["n"] for name, data in tests.items()}
        for name, data in tests.items():
            pipeline.write_text(data_dir / f"{name}.zh", data["source"])
            pipeline.write_text(data_dir / f"{name}.en", data["reference"])
    else:
        if (run_dir / "selection.json").exists():
            pipeline.verify_selection(run_dir, run, identities)
            raise ValueError("Dev is already complete. Submit the parallel test stage instead.")
        data_dir = Path(spec["split_dir"])
        datasets = {"dev_train20": spec["counts"]["dev_train20"]}
    plan = {"run_fingerprint": run["fingerprint"], "adapters": identities,
            "stage": stage, "data_dir": str(data_dir), "datasets": datasets,
            "data_sha256": {str(data_dir / f"{name}.{lang}"): sha256(data_dir / f"{name}.{lang}")
                            for name in datasets for lang in ("zh", "en")},
            "driver_sha256": sha256(Path(__file__))}
    plan["fingerprint"] = fingerprint(plan)
    directory = workspace(run_dir, stage)
    path = directory / "plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Parallel plan changed; do not mix different adapters or driver versions.")
    assignments = pipeline.read_assignments(ROOT, None)
    if len(assignments) != 4:
        raise ValueError("This launcher requires the four team assignments")
    for variant in pipeline.VARIANTS:
        original = run_dir / "evaluation" / stage / variant
        for a in assignments:
            private = directory / variant / a["owner"] / "results"
            seed_json(original / ".cache", private / ".cache")
            seed_json(original / a["owner"], private / a["owner"])
    write_json(path, plan)
    write_json(run_dir / "evaluation" / stage / "status.json",
               {"status": "running", "mode": "parallel", "datasets": datasets})
    print(f"Prepared parallel {stage}; completed candidates and score checkpoints will be reused.", flush=True)


def warm_metric_downloads():
    """Populate shared downloads once before concurrent metric workers start."""
    from cache_paths import COMET_CACHE
    from comet import download_model
    from bertscore_metric import EnglishBERTScore
    for model in ("Unbabel/wmt22-comet-da", "Unbabel/wmt22-cometkiwi-da"):
        download_model(model, saving_directory=str(COMET_CACHE))
    EnglishBERTScore(device="cpu").get_scorer()


def load_plan(run_dir, stage):
    run = load_run(run_dir)
    plan = read_json(workspace(run_dir, stage) / "plan.json")
    if (plan["fingerprint"] != fingerprint({k: v for k, v in plan.items() if k != "fingerprint"})
            or plan["run_fingerprint"] != run["fingerprint"]
            or plan["driver_sha256"] != sha256(Path(__file__)) or plan["stage"] != stage):
        raise ValueError("Changed parallel evaluation plan, run or driver")
    if any(sha256(path) != expected for path, expected in plan["data_sha256"].items()):
        raise ValueError("Evaluation data changed after parallel preparation")
    if stage == "test":
        pipeline.verify_selection(run_dir, run, plan["adapters"])
    return run, plan


def generate(run_dir, stage, index, lock_fd=None):
    run, plan = load_plan(run_dir, stage)
    if not 0 <= index < 2 * len(TEAM_MODELS):
        raise ValueError("Generation index must be 0-15")
    variant = pipeline.VARIANTS[index // len(TEAM_MODELS)]
    model = TEAM_MODELS[index % len(TEAM_MODELS)]
    settings = run["spec"]["evaluation"]
    adapter = run_dir / "adapters" / model
    identity = generation_identity(model, adapter, settings["max_new_tokens"], variant == "pretrained")
    if generation_identity(model, adapter, settings["max_new_tokens"]) != plan["adapters"][model]:
        raise ValueError(f"Adapter changed: {model}")
    from run_ensembles import valid_cache
    output = run_dir / "candidates" / variant / model / "output"
    sources = {name: read_parallel(plan["data_dir"], name)[0] for name in plan["datasets"]}
    missing = [name for name, src in sources.items() if not valid_cache(output, name, src, identity)]
    if missing:
        execute([sys.executable, ROOT / "scripts/screen_model.py", "--model", model,
                 "--adapter-dir", adapter, "--data-dir", plan["data_dir"], "--dev", *missing,
                 "--output-dir", output, "--no-comet", "--batch-size", settings["generation_batch_size"],
                 "--max-new-tokens", settings["max_new_tokens"]]
                + (["--base-only"] if variant == "pretrained" else []), lock_fd)
    if any(not valid_cache(output, name, src, identity) for name, src in sources.items()):
        raise ValueError(f"Incomplete generation: {variant} {model}")
    print(f"Candidates ready: {variant} {model} ({'generated missing data' if missing else 'reused'})", flush=True)


def metric_fingerprint(settings):
    # Match run_ensembles.py exactly so already completed owners need no GPU work.
    return digest([
        {name: (ROOT / "scripts" / name).read_text(encoding="utf-8") for name in
         ["run_ensembles.py", "ensemble_cache.py", "bertscore_metric.py", "translation_cleaning.py"]},
        {name: version(name) for name in ["torch", "transformers", "sacrebleu", "bert-score", "unbabel-comet"]},
        True, True, DEFAULT_MODEL, "cuda", settings["metric_batch_size"]])


def completed_owner(results, assignment, plan, settings):
    """Require every CSV row and its content-addressed checkpoint to agree."""
    try:
        rows = pipeline.validated_results(results, [assignment], plan["datasets"])
        metric_id = metric_fingerprint(settings)
        data = {name: read_parallel(plan["data_dir"], name) for name in plan["datasets"]}
        for row in rows:
            i, name, selector = int(row["experiment"]), row["dev"], row["selector"]
            exp = results / assignment["owner"] / f"experiment_{i:02d}"
            hyp = (exp / f"{name}_{selector}.hyp").read_text(encoding="utf-8").splitlines()
            previous = read_json(exp / f"{name}_{selector}.checkpoint.json")
            counts = json.loads(row["selection_counts"])
            src, ref = data[name]
            trio = assignment["combinations"][i - 1]
            key = digest([metric_id, i, trio, name, selector, src, ref, hyp, counts])
            if (len(hyp) != len(src) or previous.get("digest") != key
                    or any(str(value) != row.get(k) for k, value in previous["row"].items())):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".copying")
    shutil.copyfile(source, temp)
    temp.replace(destination)


def score_worker(run_dir, stage, index, lock_fd=None):
    run, plan = load_plan(run_dir, stage)
    if not 0 <= index < 10:
        raise ValueError("Scoring index must be 0-9")
    settings = run["spec"]["evaluation"]
    assignments = pipeline.read_assignments(ROOT, None)
    if index >= 8:
        variant = pipeline.VARIANTS[index - 8]
        execute([sys.executable, ROOT / "scripts/score_finetuned_models.py", "--run-dir", run_dir,
                 "--stage", stage, "--variant", variant, "--data-dir", plan["data_dir"],
                 "--datasets", *plan["datasets"]], lock_fd)
        output = run_dir / "evaluation" / stage / variant / "individual/scores.csv"
    else:
        variant, a = pipeline.VARIANTS[index // 4], assignments[index % 4]
        canonical = run_dir / "evaluation" / stage / variant
        output = canonical / a["owner"] / "scores_comparison.csv"
        if completed_owner(canonical, a, plan, settings):
            print(f"Reused all completed ensemble scores: {variant} / {a['owner']}", flush=True)
        else:
            private = workspace(run_dir, stage) / variant / a["owner"]
            candidates = private / "candidates"
            # The old QE writer is not atomic. Keep damaged files for diagnosis
            # but let a retry regenerate the affected cache instead of crashing.
            for path in (private / "results").rglob("*.json"):
                try:
                    read_json(path)
                except ValueError:
                    path.replace(path.with_name(path.name + ".incomplete-" + sha256(path)[:12]))
            from run_ensembles import valid_cache
            for model in dict.fromkeys(m for trio in a["combinations"] for m in trio):
                source = run_dir / "candidates" / variant / model / "output"
                identity = generation_identity(model, run_dir / "adapters" / model,
                                               settings["max_new_tokens"], variant == "pretrained")
                for name in plan["datasets"]:
                    if not valid_cache(source, name, read_parallel(plan["data_dir"], name)[0], identity):
                        raise ValueError(f"Missing candidates: {variant} {model} {name}")
                    for suffix in ("hyp", "raw.jsonl", "generation.json"):
                        copy_file(source / f"{name}.{suffix}", candidates / model / "output" / f"{name}.{suffix}")
            execute([sys.executable, ROOT / "scripts/run_ensembles.py", "--combinations", a["path"],
                     "--owner", a["owner"], "--adapter-root", run_dir / "adapters", "--model-outputs-dir",
                     candidates, "--results-dir", private / "results", "--data-dir", plan["data_dir"],
                     "--datasets", *plan["datasets"], "--selector", "all", "--comet", "--bertscore",
                     "--bertscore-device", "cuda", "--bertscore-batch-size", settings["metric_batch_size"],
                     "--max-new-tokens", settings["max_new_tokens"], "--no-generate"]
                    + (["--base-only"] if variant == "pretrained" else []), lock_fd)
            pipeline.validated_results(private / "results", [a], plan["datasets"])
            # Each task exclusively owns this owner/variant destination.
            source = private / "results" / a["owner"]
            for path in source.rglob("*"):
                if path.is_file() and path.suffix in (".csv", ".json", ".hyp"):
                    copy_file(path, canonical / a["owner"] / path.relative_to(source))
    write_json(workspace(run_dir, stage) / "done" / f"score-{index}.json",
               {"plan": plan["fingerprint"], "output": str(output), "sha256": sha256(output)})


def collect(run_dir, stage):
    run, plan = load_plan(run_dir, stage)
    identities = verified_adapters(run_dir, run)
    if identities != plan["adapters"]:
        raise ValueError("Adapters changed during evaluation")
    for index in range(10):
        done = read_json(workspace(run_dir, stage) / "done" / f"score-{index}.json")
        if done["plan"] != plan["fingerprint"] or sha256(done["output"]) != done["sha256"]:
            raise ValueError(f"Incomplete or changed scoring task {index}")
    assignments = pipeline.read_assignments(ROOT, None)
    results = run_dir / "evaluation" / stage
    rows, individuals = [], []
    for variant in pipeline.VARIANTS:
        rows.extend({"variant": variant, **row} for row in
                    pipeline.validated_results(results / variant, assignments, plan["datasets"]))
        individuals.extend(pipeline.read_csv(results / variant / "individual/scores.csv"))
    expected = {(v, m, d) for v in pipeline.VARIANTS for m in TEAM_MODELS for d in plan["datasets"]}
    if (len(individuals) != len(expected)
            or {(r["variant"], r["model"], r["dataset"]) for r in individuals} != expected
            or any(int(r["n"]) != plan["datasets"][r["dataset"]] for r in individuals)
            or any(not math.isfinite(float(r[m])) for r in individuals
                   for m in ("comet22", "bleu", "chrfpp", "bertscore_f1"))):
        raise ValueError("Missing or invalid individual scores")
    if stage == "dev" and (run_dir / "selection.json").exists():
        pipeline.verify_selection(run_dir, run, identities)
        print("Dev selection already frozen and verified; leaving it unchanged.")
        return
    pipeline.save_csv(results / "individual_scores.csv", individuals)
    pipeline.save_csv(results / "individual_deltas.csv", pipeline.paired_deltas(individuals, ("model", "dataset")))
    pipeline.save_csv(results / "ensemble_deltas.csv",
                      pipeline.paired_deltas(rows, ("owner", "experiment", "models", "dev", "selector")))
    if stage == "test":
        primary = pipeline.verify_selection(run_dir, run, identities)["primary"]
        for row in rows:
            row["dev_selected_primary"] = all(row[k] == primary[k] for k in ("variant", "owner", "models", "selector"))
        selected = [r for r in rows if r["dev_selected_primary"]]
        if len(selected) != len(plan["datasets"]):
            raise ValueError("Missing frozen primary system on a test dataset")
        pipeline.save_csv(results / "final_selected_scores.csv", selected)
    pipeline.save_csv(results / "scores_comparison.csv", rows)
    if stage == "dev":
        write_json(run_dir / "selection.json", {**pipeline.choose_on_dev(rows),
                   "run_fingerprint": run["fingerprint"], "adapters": identities,
                   "selection_rule": "highest dev COMET22; ties: BLEU, chrF++, pretrained, assignment order",
                   "dev_scores_sha256": sha256(results / "scores_comparison.csv")})
        pipeline.verify_selection(run_dir, run, identities)
    write_json(results / "status.json", {"status": "complete", "mode": "parallel",
                                         "datasets": plan["datasets"], "rows": len(rows)})
    print(f"Completed {stage}: {len(rows)} ensemble evaluations; {results}", flush=True)


def progress(run_dir):
    """Read saved outputs while a job is running, without importing GPU libraries."""
    if not (run_dir / "run.json").exists():
        raise FileNotFoundError(f"No prepared run at {run_dir}")
    print("Saved progress (file counts; resume checks full data/adapter integrity):")
    trained = [m for m in TEAM_MODELS if (run_dir / "adapters" / m / "training_complete.json").exists()]
    print(f"Training completion records: {len(trained)}/8")
    for stage, names in (("dev", ["dev_train20"]), ("test", ["test_train10", "test_wmt22", "test_tatoeba"])):
        for variant in pipeline.VARIANTS:
            candidates = sum((run_dir / "candidates" / variant / m / "output" / f"{d}.generation.json").exists()
                             for m in TEAM_MODELS for d in names)
            scores = {}
            locations = [run_dir / "evaluation" / stage / variant,
                         workspace(run_dir, stage) / variant]
            for location in locations:
                for path in location.rglob("*.checkpoint.json"):
                    try:
                        row = read_json(path)["row"]
                        owner = path.parent.parent.name
                        scores[(owner, row["experiment"], row["dev"], row["selector"])] = row
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
            print(f"{stage} / {variant}: candidates {candidates}/{8 * len(names)}, "
                  f"ensemble scores {len(scores)}/{120 * len(names)}")
            for path in (run_dir / "evaluation" / stage / variant).glob("*/scores*.csv"):
                print(f"  CSV: {path}")
    choice = run_dir / "selection.json"
    if choice.exists():
        primary = read_json(choice)["primary"]
        print(f"Frozen dev choice: {primary['variant']} / {primary['models']} / {primary['selector']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "generate", "score", "collect", "status"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["dev", "test"], default="dev")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.action == "status":
        progress(run_dir)
        return
    if args.action in ("generate", "score") and args.index is None:
        parser.error("A worker requires --index")
    shared = args.action in ("generate", "score")
    with evaluation_lock(run_dir, shared=shared) as lock_fd:
        if args.action == "init":
            initialize(run_dir, args.stage)
            warm_metric_downloads()
        elif args.action == "collect":
            collect(run_dir, args.stage)
        else:
            directory = workspace(run_dir, args.stage)
            directory.mkdir(parents=True, exist_ok=True)
            with evaluation_lock(directory, filename=f"{args.action}-{args.index}.lock") as task_fd:
                worker = generate if args.action == "generate" else score_worker
                worker(run_dir, args.stage, args.index, (lock_fd, task_fd))


if __name__ == "__main__":
    main()
