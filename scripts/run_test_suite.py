"""Evaluate every assigned ensemble on test data using all three selectors.

Run from a CUDA Python environment. Use --dry-run to check inputs and commands
without loading models. Test scores describe held-out performance only while
the final configuration remains fixed from development-set selection.
"""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTORS = {"cometkiwi", "chrf", "bertscore"}


def read_test_data(test_set, official_dir, split_dir):
    paths = {}
    if test_set in ("official", "both"):
        paths["test_wmt22"] = (official_dir / "wmttest2022.zh",
                                official_dir / "wmttest2022.AnnA.en")
        paths["test_tatoeba"] = (official_dir / "tatoeba.zh", official_dir / "tatoeba.en")
    if test_set in ("internal", "both"):
        paths["test_train10"] = (split_dir / "test_train10.zh", split_dir / "test_train10.en")

    data = {}
    for name, (source_path, reference_path) in paths.items():
        source = source_path.read_text(encoding="utf-8-sig").splitlines()
        reference = reference_path.read_text(encoding="utf-8-sig").splitlines()
        if not source or len(source) != len(reference):
            raise ValueError(f"{name}: empty dataset or source/reference line-count mismatch")
        if any(not line.strip() for line in source + reference):
            raise ValueError(f"{name}: blank source or reference sentence")
        source_text = "\n".join(source) + "\n"
        reference_text = "\n".join(reference) + "\n"
        data[name] = {
            "source": source_text, "reference": reference_text, "n": len(source),
            "source_path": str(source_path.resolve()),
            "reference_path": str(reference_path.resolve()),
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
        }
    return data


def read_assignments(root, owners):
    manifest = json.loads((root / "combinations_team_manifest.json").read_text(encoding="utf-8"))
    available = {a["owner"]: a for a in manifest["assignments"]}
    selected = list(available) if owners is None else owners
    if len(selected) != len(set(selected)):
        raise ValueError("Specify each owner only once")
    assignments = []
    for owner in selected:
        if owner not in available:
            raise ValueError(f"Unknown owner {owner!r}; choose from {', '.join(available)}")
        assignment = available[owner]
        path = root / assignment["file"]
        combinations = json.loads(path.read_text(encoding="utf-8"))
        expected = [e["models"] for e in assignment["experiments"]]
        if combinations != expected:
            raise ValueError(f"{path.name} differs from the team assignment manifest")
        assignments.append({"owner": owner, "path": path, "combinations": combinations})
    return assignments


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def collect_results(root, assignments, datasets, run_name):
    combined = []
    for assignment in assignments:
        owner = assignment["owner"]
        path = root / "results" / "ensembles" / owner / run_name / "scores_comparison.csv"
        expected = {(i, dataset, selector)
                    for i in range(1, len(assignment["combinations"]) + 1)
                    for dataset in datasets for selector in SELECTORS}
        seen = set()
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["experiment"]), row["dev"], row["selector"])
                if key not in expected or key in seen:
                    raise ValueError(f"Unexpected or duplicate evaluation in {path}: {key}")
                if int(row["n"]) != datasets[row["dev"]]["n"]:
                    raise ValueError(f"Wrong sentence count in {path}: {key}")
                models = " + ".join(assignment["combinations"][key[0] - 1])
                if row["models"] != models:
                    raise ValueError(f"Wrong model combination in {path}: {key}")
                seen.add(key)
                combined.append({"owner": owner, **row})
        if seen != expected:
            raise ValueError(f"Missing {len(expected - seen)} evaluations in {path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", choices=["official", "internal", "both"], default="official")
    parser.add_argument("--owners", nargs="+", help="default: all four members in the assignment manifest")
    parser.add_argument("--official-dir", type=Path, default=ROOT)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits_70_20_10")
    parser.add_argument("--bertscore-device", default="cpu")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.bertscore_batch_size < 1:
        parser.error("--bertscore-batch-size must be positive")
    try:
        datasets = read_test_data(args.test_set, args.official_dir, args.split_dir)
        assignments = read_assignments(ROOT, args.owners)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    run_name = f"test_{args.test_set}_all"
    data_dir = ROOT / "data" / "test_suites" / args.test_set
    summary_dir = ROOT / "results" / "ensembles" / "test_suites" / args.test_set
    commands = []
    for assignment in assignments:
        command = [sys.executable, "-u", str(ROOT / "scripts" / "run_ensembles.py"),
                   "--combinations", str(assignment["path"]), "--owner", assignment["owner"],
                   "--run-name", run_name, "--selector", "all", "--comet", "--bertscore",
                   "--data-dir", str(data_dir), "--datasets", *datasets,
                   "--bertscore-device", args.bertscore_device,
                   "--bertscore-batch-size", str(args.bertscore_batch_size)]
        if args.no_generate:
            command.append("--no-generate")
        commands.append(command)
    expected = sum(len(a["combinations"]) for a in assignments) * len(SELECTORS) * len(datasets)
    print(f"Test suite: {len(assignments)} members, {expected} evaluation rows", flush=True)
    for name, data in datasets.items():
        print(f"  {name}: {data['n']} sentences; reference {data['reference_path']}", flush=True)
    print("Keep the final choice fixed from dev. Selecting from these results makes them validation results.", flush=True)
    if args.dry_run:
        for assignment, command in zip(assignments, commands):
            print(f"{assignment['owner']}: {json.dumps(command)}")
        return 0

    # filelock is installed with the Transformers inference dependencies.
    # Serialize test-suite runs sharing this checkout and its generation caches.
    from filelock import FileLock, Timeout
    lock_path = ROOT / "results" / "ensembles" / "test_suites" / "suite.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock_path), timeout=0):
            for name, data in datasets.items():
                write_text(data_dir / f"{name}.zh", data["source"])
                write_text(data_dir / f"{name}.en", data["reference"])
            manifest = {
                "test_set": args.test_set, "run_name": run_name, "expected_rows": expected,
                "datasets": {name: {k: v for k, v in data.items() if k not in ("source", "reference")}
                             for name, data in datasets.items()},
                "assignments": [{"owner": a["owner"], "combinations": a["combinations"]} for a in assignments],
                "status": "running",
            }
            manifest_path = summary_dir / "manifest.json"
            write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
            combined_path = summary_dir / "scores_comparison.csv"
            combined_path.unlink(missing_ok=True)
            failed = []
            for assignment, command in zip(assignments, commands):
                print(f"Running {assignment['owner']}", flush=True)
                if subprocess.run(command, cwd=ROOT).returncode:
                    failed.append(assignment["owner"])
            if failed:
                manifest.update(status="failed", failed_owners=failed)
                write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
                print(f"Incomplete suite. Failed owners: {', '.join(failed)}. "
                      "Rerun the same command to reuse completed evaluations.", file=sys.stderr)
                return 1
            try:
                rows = collect_results(ROOT, assignments, datasets, run_name)
            except (OSError, ValueError, KeyError) as error:
                manifest.update(status="failed", error=str(error))
                write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
                print(str(error), file=sys.stderr)
                return 1
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            write_text(combined_path, buffer.getvalue())
            manifest.update(status="complete", completed_rows=len(rows))
            write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
            print(f"Saved {len(rows)} evaluations to {combined_path}", flush=True)
    except Timeout:
        parser.error("Another test-suite process is using this checkout. Wait for it to finish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
