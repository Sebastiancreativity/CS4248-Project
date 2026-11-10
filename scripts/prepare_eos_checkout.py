"""Bind a fresh checkout to an existing training run without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from install_eos_additions import git, sha256, verify_run

MARKER = "eos_checkout.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(destination):
    plan = json.loads((destination / MARKER).read_text(encoding="utf-8"))
    if plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Checkout provenance was modified")
    source, source_run = Path(plan["source_repo"]), Path(plan["source_run"])
    verify_run(source, source_run)
    if sha256(source_run / "run.json") != plan["run_manifest_sha256"]:
        raise ValueError("Original training manifest changed")
    for name, expected in plan["files"].items():
        if sha256(source / name) != expected or sha256(destination / name) != expected:
            raise ValueError(f"Source/copy differs from the original experiment: {name}")
    if (destination / "artifacts/finetune" / plan["run_name"]).resolve() != source_run:
        raise ValueError("Training artifacts no longer point to the original run")
    return plan


def prepare(source, destination, run_name):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Use a separate sibling checkout; the active checkout must not be modified")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("Invalid run name")
    if Path(git(destination, "rev-parse", "--show-toplevel").decode().strip()).resolve() != destination:
        raise ValueError("Run from the new checkout root")
    if (destination / MARKER).exists():
        existing = verify(destination)
        if Path(existing["source_repo"]) != source or existing["run_name"] != run_name:
            raise ValueError("This checkout belongs to another run")
        print("Verified the existing separate checkout.")
        return
    source_run = (source / "artifacts/finetune" / run_name).resolve()
    verify_run(source, source_run)
    spec = json.loads((source_run / "run.json").read_text(encoding="utf-8"))["spec"]
    files = dict(spec["recipe_sha256"])
    for path, expected in spec["input_sha256"].items():
        relative = Path(path).resolve().relative_to(source)
        files[relative.as_posix()] = expected
    # Also preserve the loading/cache helpers used by the active Linux recipe.
    for name in ("cache_paths.py", "checkpoint_loading.py", "streaming_loading.py"):
        path = source / "scripts" / name
        if path.exists():
            files["scripts/" + name] = sha256(path)
    plan = {"source_repo": str(source), "source_run": str(source_run), "run_name": run_name,
            "run_manifest_sha256": sha256(source_run / "run.json"), "files": files}
    plan["fingerprint"] = digest(plan)
    preparing = destination / ".eos_checkout_preparing.json"
    link = destination / "artifacts/finetune" / run_name
    if preparing.exists():
        if json.loads(preparing.read_text(encoding="utf-8")) != plan:
            raise ValueError("An incomplete preparation belongs to another source")
    else:
        if git(destination, "status", "--porcelain", "--untracked-files=all").strip():
            raise ValueError("The destination must be a fresh, clean checkout")
        if link.exists() or link.is_symlink():
            raise ValueError("Destination already contains training artifacts")
        tracked = set(git(destination, "ls-files", "-z").decode().split("\0"))
        for name, expected in files.items():
            target = destination / name
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Destination path points outside the new checkout: {name}")
            if target.exists() and name not in tracked and sha256(target) != expected:
                raise ValueError(f"Refusing to overwrite an existing untracked file: {name}")
        save(preparing, plan)
    for name, expected in files.items():
        target = destination / name
        if sha256(source / name) != expected:
            raise ValueError(f"Source changed during preparation: {name}")
        if target.exists() and sha256(target) == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".eos-copying")
        shutil.copyfile(source / name, temporary)
        temporary.replace(target)
    if not link.exists() and not link.is_symlink():
        link.parent.mkdir(parents=True, exist_ok=True)
        from eos_revision import link_adapters
        link_adapters(source_run, link)
    save(destination / MARKER, plan)
    verify(destination)
    print(f"Separate checkout ready: {destination}")
    print(f"Reusing training, adapters and old dev results from: {source_run}")
    print("Original checkout unchanged. Original recipe and data verified; new EOS outputs stay separate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--run-name", default="ft702010_v1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    destination = Path.cwd().resolve()
    try:
        if args.verify:
            verify(destination)
        elif args.source_repo is None:
            parser.error("Preparation requires --source-repo")
        else:
            prepare(args.source_repo, destination, args.run_name)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Checkout preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
