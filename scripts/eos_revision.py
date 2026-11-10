"""Evaluate the Tower-Mistral EOS correction without modifying an active recipe.

The revision has its own candidates, scores and dev selection. Training remains
owned by the original run. Scoped, process-local bindings let the unchanged
evaluators use the revision's identities; every recipe subprocess re-enters this
driver. There is deliberately no run.json in the revision: invoking a legacy
launcher on it must fail rather than silently restore the old decoding policy.
"""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from finetune_common import (ROOT, adapter_record, fingerprint, generation_identity as original_identity,
                             load_run as load_training_run, read_json, read_parallel, sha256, write_json)
from model_registry import TEAM_MODELS
from translation_cleaning import clean

TARGET = "tower-mistral-7b"
POLICY = {"model": TARGET, "hf": "Unbabel/TowerInstruct-Mistral-7B-v0.2",
          "eos_token": "<|im_end|>", "eos_token_id": 32000, "revision": "tower-eos-v1"}
CODE = ("eos_revision.py", "parallel_finetune_eval.py", "priority_test.py", "run_eos_revision.slurm")
CHILDREN = {"screen_model.py": "screen", "run_ensembles.py": "ensemble",
            "score_finetuned_models.py": "individual"}
MODULES = {"screen": "screen_model", "ensemble": "run_ensembles", "individual": "score_finetuned_models"}


def revised_identity(model, adapter, max_tokens, base_only=False):
    original = original_identity(model, adapter, max_tokens, base_only)
    if model != TARGET:
        return original
    return fingerprint({"original": original, "decoding": POLICY, "implementation": sha256(Path(__file__))})


def corrected_load(loader, cfg, adapter_dir=None, base_only=False):
    tok, model = loader(cfg, adapter_dir, base_only)
    if cfg["hf"] == POLICY["hf"]:
        if tok.eos_token != POLICY["eos_token"] or tok.eos_token_id != POLICY["eos_token_id"]:
            raise ValueError("Unexpected Tower-Mistral tokenizer; refusing an unvalidated EOS correction")
        previous = model.generation_config.eos_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
        print(f"Tower-Mistral EOS correction: {previous} -> {tok.eos_token_id}; all other decoding settings unchanged",
              flush=True)
    return tok, model


def cleaned_candidates(output, dataset, sources, identity, verify_hyp=True):
    """Dependency-free full-cache check, including the cleaned text on disk."""
    rows = [json.loads(line) for line in (output / f"{dataset}.raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if (read_json(output / f"{dataset}.generation.json").get("identity") != identity
            or [r.get("src") for r in rows] != sources):
        raise ValueError(f"Missing, stale or misaligned candidates: {output} / {dataset}")
    hyp = [clean(r["raw"], r["src"]) for r in rows]
    if verify_hyp and (output / f"{dataset}.hyp").read_text(encoding="utf-8").splitlines() != hyp:
        raise ValueError(f"Cleaned candidates do not match raw outputs: {output} / {dataset}")
    return hyp


def parent_state(source, require_scores=False):
    import finetune_pipeline as pipeline
    run = load_training_run(source)
    if require_scores and read_json(source / "evaluation/dev/status.json").get("status") != "complete":
        raise ValueError("Wait for the original dev job to complete before preparing this revision")
    identities = {}
    for model in TEAM_MODELS:
        adapter = source / "adapters" / model
        if adapter_record(adapter, model)["run_fingerprint"] != run["fingerprint"]:
            raise ValueError(f"Wrong training provenance: {model}")
        identities[model] = original_identity(model, adapter, run["spec"]["evaluation"]["max_new_tokens"])
    if require_scores:
        pipeline.verify_selection(source, run, identities)
    return run


def checked_plan(revision):
    plan = read_json(revision / "revision.json")
    payload = {k: v for k, v in plan.items() if k != "fingerprint"}
    if plan.get("fingerprint") != fingerprint(payload) or plan.get("decoding") != POLICY:
        raise ValueError("The evaluation revision was modified; create a new revision")
    for name, expected in plan["code_sha256"].items():
        if sha256(ROOT / "scripts" / name) != expected:
            raise ValueError(f"Revision code changed: {name}; create a new revision")
    source = Path(plan["training_run"])
    run = load_training_run(source)
    if run["fingerprint"] != plan["training_fingerprint"]:
        raise ValueError("Training run changed")
    for name, expected in plan["source_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Source artifact changed after revision preparation: {name}")
    if (revision / "adapters").resolve() != (source / "adapters").resolve():
        raise ValueError("Revision adapters no longer point to the verified training run")
    for model, expected in plan["adapter_records"].items():
        adapter = source / "adapters" / model
        if sha256(adapter / "training_complete.json") != expected:
            raise ValueError(f"Training record changed: {model}")
        if adapter_record(adapter, model)["run_fingerprint"] != run["fingerprint"]:
            raise ValueError(f"Wrong adapter provenance: {model}")
    seeded = revision / "seed.json"
    if seeded.exists():
        snapshot = read_json(seeded)
        if snapshot.get("revision") != plan["fingerprint"]:
            raise ValueError("Metric snapshot belongs to another revision")
        for name, expected in snapshot["source_sha256"].items():
            if sha256(source / name) != expected:
                raise ValueError(f"Original dev scores changed after reuse: {name}")
    return run, plan


def copy_artifact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".copying")
    shutil.copyfile(source, temp)
    temp.replace(destination)


def link_adapters(source, destination):
    if os.name != "nt":
        destination.symlink_to(source, target_is_directory=True)
        return
    # Directory junctions need no Windows symlink privilege and have the same
    # read-only use here. No weights are copied or written by this driver.
    env = {**os.environ, "CS4248_ADAPTER_SOURCE": str(source), "CS4248_ADAPTER_LINK": str(destination)}
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                    "New-Item -ItemType Junction -Path $env:CS4248_ADAPTER_LINK -Target $env:CS4248_ADAPTER_SOURCE -ErrorAction Stop | Out-Null"],
                   env=env, check=True, capture_output=True, text=True)


def prepare(source, revision):
    source, revision = source.resolve(), revision.resolve()
    if source == revision or source in revision.parents or revision in source.parents:
        raise ValueError("Use a separate sibling directory for the evaluation revision")
    if (revision / "revision.json").exists():
        _, previous = checked_plan(revision)
        if Path(previous["training_run"]) != source:
            raise ValueError("Revision belongs to another training run")
        return
    run = parent_state(source)
    settings = run["spec"]["evaluation"]
    src, _ = read_parallel(run["spec"]["split_dir"], "dev_train20")
    files = [source / "run.json"]
    for variant in ("pretrained", "finetuned"):
        for model in TEAM_MODELS:
            output = source / "candidates" / variant / model / "output"
            identity = original_identity(model, source / "adapters" / model, settings["max_new_tokens"], variant == "pretrained")
            # The still-running original scorer rewrites .hyp files. Raw outputs
            # and generation records are final, so snapshot those and re-clean.
            cleaned_candidates(output, "dev_train20", src, identity, verify_hyp=False)
            files.extend(output / f"dev_train20.{suffix}" for suffix in ("raw.jsonl", "generation.json"))
    plan = {"training_run": str(source), "training_fingerprint": run["fingerprint"], "decoding": POLICY,
            "code_sha256": {name: sha256(ROOT / "scripts" / name) for name in CODE},
            "source_sha256": {str(p.relative_to(source)): sha256(p) for p in files},
            "adapter_records": {m: sha256(source / "adapters" / m / "training_complete.json") for m in TEAM_MODELS}}
    plan["fingerprint"] = fingerprint(plan)
    marker = revision / "preparing.json"
    if revision.exists() and any(revision.iterdir()):
        if not marker.exists() or read_json(marker) != plan:
            raise ValueError("Destination is not empty or contains another revision")
    revision.mkdir(parents=True, exist_ok=True)
    write_json(marker, plan)
    adapters = revision / "adapters"
    if not adapters.exists() and not adapters.is_symlink():
        link_adapters(source / "adapters", adapters)
    if adapters.resolve() != (source / "adapters").resolve():
        raise ValueError("Wrong adapter link")
    for path in files:
        relative = path.relative_to(source)
        if relative.parts[0] == "candidates":
            copy_artifact(path, revision / relative)
            if path.name.endswith(".raw.jsonl"):
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                (revision / relative.parent / "dev_train20.hyp").write_text(
                    "\n".join(clean(r["raw"], r["src"]) for r in rows) + "\n", encoding="utf-8")
    # Neither the old selection nor complete aggregate/status files are copied.
    for name, expected in plan["source_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Source changed while preparing: {name}")
    write_json(revision / "revision.json", plan)
    checked_plan(revision)
    print(f"Prepared EOS revision: {revision}; reusing all 8 trained adapters and 14 unaffected dev candidate sets", flush=True)


def seed_scores(revision):
    """Copy completed metrics after the old dev job finishes, never live writes."""
    run, plan = checked_plan(revision)
    source = Path(plan["training_run"])
    parent_state(source, require_scores=True)
    if (revision / "seed.json").exists():
        return
    if (revision / "parallel_evaluation/dev/done").exists():
        raise ValueError("Refusing to replace scores after revision scoring began")
    files = [source / name for name in ("selection.json", "evaluation/dev/scores_comparison.csv",
                                       "evaluation/dev/individual_scores.csv", "evaluation/dev/status.json")]
    for variant in ("pretrained", "finetuned"):
        files.extend(p for p in (source / "evaluation/dev" / variant).rglob("*")
                     if p.is_file() and not p.is_symlink() and p.suffix in (".csv", ".json", ".hyp"))
    checksums = {str(p.relative_to(source)): sha256(p) for p in files}
    for path in files:
        relative = path.relative_to(source)
        if len(relative.parts) > 2 and relative.parts[2] in ("pretrained", "finetuned"):
            copy_artifact(path, revision / relative)
    if any(sha256(source / name) != expected for name, expected in checksums.items()):
        raise ValueError("Source dev scores changed during copying")
    write_json(revision / "seed.json", {"revision": plan["fingerprint"], "source_sha256": checksums})


def same_dev_inputs(revision, plan, run, variant, models):
    source = Path(plan["training_run"])
    src, _ = read_parallel(run["spec"]["split_dir"], "dev_train20")
    cap = run["spec"]["evaluation"]["max_new_tokens"]
    for model in models:
        old = cleaned_candidates(source / "candidates" / variant / model / "output", "dev_train20", src,
                                 original_identity(model, source / "adapters" / model, cap, variant == "pretrained"), verify_hyp=False)
        new = cleaned_candidates(revision / "candidates" / variant / model / "output", "dev_train20", src,
                                 revised_identity(model, revision / "adapters" / model, cap, variant == "pretrained"))
        if old != new:
            return False
    return True


def redirected_command(command, revision):
    command = list(map(str, command))
    if (len(command) < 2 or Path(command[1]).name not in CHILDREN
            or Path(command[1]).resolve() != ROOT / "scripts" / Path(command[1]).name):
        raise ValueError(f"Unexpected recipe subprocess: {command}")
    task = CHILDREN[Path(command[1]).name]
    if task == "ensemble" and "--no-generate" not in command:
        raise ValueError("Ensemble workers may only score already verified candidates")
    return [command[0], "-u", str(Path(__file__).resolve()), "child", "--revision-dir", str(revision),
            "--task", task, "--", *command[2:]]


def validate_child_paths(task, argv, revision):
    flags = {"screen": ("--output-dir",), "ensemble": ("--model-outputs-dir", "--results-dir"),
             "individual": ("--run-dir",)}[task]
    for flag in flags:
        if flag not in argv:
            raise ValueError(f"Revision worker requires {flag}")
        path = Path(argv[argv.index(flag) + 1]).resolve()
        if path != revision and revision not in path.parents:
            raise ValueError(f"Refusing to write outside the evaluation revision: {path}")


@contextmanager
def runtime(revision, child=None):
    """Bind explicit compatibility adapters in this process, never edit recipe files."""
    run, plan = checked_plan(revision)
    import finetune_pipeline as pipeline
    import parallel_finetune_eval as parallel
    import priority_test as priority
    modules = [pipeline, parallel, priority]
    target = importlib.import_module(MODULES[child]) if child else None
    if target:
        modules.append(target)
    saved = []

    def bind(module, name, value):
        saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def load_run(path):
        if Path(path).resolve() != revision:
            raise ValueError("This worker may only write to its declared evaluation revision")
        return checked_plan(revision)[0]

    execute = parallel.execute
    original_completed = parallel.completed_owner

    def completed(results, assignment, stage_plan, settings):
        if stage_plan["stage"] == "dev":
            models = set(m for trio in assignment["combinations"] for m in trio)
            if not same_dev_inputs(revision, plan, run, results.name, models):
                return False
        return original_completed(results, assignment, stage_plan, settings)

    try:
        for module in modules:
            if hasattr(module, "load_run"):
                bind(module, "load_run", load_run)
            if hasattr(module, "generation_identity"):
                bind(module, "generation_identity", revised_identity)
        bind(pipeline, "execute", lambda command: execute(redirected_command(command, revision)))
        bind(parallel, "execute", lambda command, lock_fd=None: execute(redirected_command(command, revision), lock_fd))
        bind(parallel, "completed_owner", completed)
        if child == "screen":
            loader = target.load
            bind(target, "load", lambda cfg, adapter_dir=None, base_only=False: corrected_load(loader, cfg, adapter_dir, base_only))
        yield pipeline, parallel, priority, target
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def dev_comparison(revision):
    run, plan = checked_plan(revision)
    source = Path(plan["training_run"])
    src, _ = read_parallel(run["spec"]["split_dir"], "dev_train20")
    cap = run["spec"]["evaluation"]["max_new_tokens"]
    rows = []
    for variant in ("pretrained", "finetuned"):
        old = cleaned_candidates(source / "candidates" / variant / TARGET / "output", "dev_train20", src,
                                 original_identity(TARGET, source / "adapters" / TARGET, cap, variant == "pretrained"))
        new = cleaned_candidates(revision / "candidates" / variant / TARGET / "output", "dev_train20", src,
                                 revised_identity(TARGET, revision / "adapters" / TARGET, cap, variant == "pretrained"))
        rows.append({"variant": variant, "n": len(src), "changed_cleaned_translations": sum(a != b for a, b in zip(old, new))})
    write_json(revision / "dev_eos_comparison.json", {"revision": plan["fingerprint"], "comparisons": rows})
    print(json.dumps(rows, indent=2), flush=True)


def progress(revision):
    print(f"EOS evaluation revision: {revision}")
    for stage, datasets in (("dev", ["dev_train20"]), ("test", ["test_train10", "test_wmt22", "test_tatoeba"])):
        for variant in ("pretrained", "finetuned"):
            count = sum((revision / "candidates" / variant / m / "output" / f"{d}.generation.json").exists()
                        for m in TEAM_MODELS for d in datasets)
            print(f"{stage}/{variant}: {count}/{8 * len(datasets)} candidate markers (not a completion check)")
    for stage in ("dev", "priority_test", "test"):
        path = revision / "evaluation" / stage / "status.json"
        print(f"{stage}: {read_json(path) if path.exists() else 'pending'}")
    comparison = revision / "dev_eos_comparison.json"
    if comparison.exists():
        print(json.dumps(read_json(comparison), indent=2))
    print(f"First report: {revision / 'evaluation/priority_test/report_scores.csv'}")
    print(f"Full results: {revision / 'evaluation/test/scores_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "seed", "init", "generate", "score", "collect", "priority", "status", "child"])
    parser.add_argument("--revision-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--stage", choices=["dev", "test"], default="dev")
    parser.add_argument("--index", type=int)
    parser.add_argument("--task", choices=list(MODULES))
    # Child script arguments follow a literal -- and retain their exact order.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    child_args = argv[split + 1:]
    revision = args.revision_dir.resolve()
    if args.action == "status":
        return progress(revision)
    if args.action == "prepare":
        if args.source_run is None:
            parser.error("prepare requires --source-run")
        prepare(args.source_run, revision)
    if args.action == "child" and args.task is None:
        parser.error("child requires --task")
    if args.action in ("generate", "score") and args.index is None:
        parser.error("worker requires --index")
    if args.action == "child":
        validate_child_paths(args.task, child_args, revision)
    with runtime(revision, args.task if args.action == "child" else None) as (pipeline, parallel, priority, target):
        if args.action == "child":
            if args.task == "ensemble" and "--no-generate" not in child_args:
                parser.error("revision ensemble workers require --no-generate")
            sys.argv = [MODULES[args.task] + ".py", *child_args]
            target.main()
        else:
            shared = args.action in ("generate", "score")
            with parallel.evaluation_lock(revision, shared=shared) as lock_fd:
                if args.action in ("prepare", "seed", "init"):
                    if args.action == "seed":
                        seed_scores(revision)
                    if args.stage == "dev" and (revision / "selection.json").exists():
                        raise ValueError("Development is already complete; resume with the test stage")
                    parallel.initialize(revision, args.stage)
                elif args.action == "collect":
                    if args.stage == "dev":
                        dev_comparison(revision)
                    parallel.collect(revision, args.stage)
                elif args.action == "priority":
                    # Keep the revision lock in recipe children even if their
                    # coordinator is terminated by Slurm during a subprocess.
                    pipeline.execute = lambda command: parallel.execute(command, lock_fd)
                    priority.evaluate_locked(revision)
                else:
                    if args.action == "score" and args.stage == "dev" and not (revision / "seed.json").exists():
                        raise ValueError("Wait for completed original dev scores and run seed first")
                    directory = parallel.workspace(revision, args.stage)
                    directory.mkdir(parents=True, exist_ok=True)
                    with parallel.evaluation_lock(directory, filename=f"{args.action}-{args.index}.lock") as task_fd:
                        worker = parallel.generate if args.action == "generate" else parallel.score_worker
                        worker(revision, args.stage, args.index, (lock_fd, task_fd))
    checked_plan(revision)


if __name__ == "__main__":
    main()
