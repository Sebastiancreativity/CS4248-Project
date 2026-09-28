"""Run the complete NUCLE selective multi-pass experiment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from mpd_core import edit_distance, fit_logistic, oracle_label, predict_probability


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
DEFAULT_RUN = CODE / "pipeline_runs/nucle-full-grouped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_RUN / "model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RUN / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: use the first N dev/test rows")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[object]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n> {printable}", flush=True)
    subprocess.run([str(part) for part in command], check=True, cwd=CODE)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_lines(path: Path, limit: int = 0) -> list[str]:
    rows = path.read_text(encoding="utf-8").splitlines()
    return rows[:limit] if limit > 0 else rows


def write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def stable_choice(row: dict[str, object]) -> tuple[str, int]:
    if row["pass1"] == row["pass2"]:
        return str(row["pass1"]), 1
    if row["pass2"] == row["pass3"]:
        return str(row["pass2"]), 2
    return str(row["pass3"]), 3


def score_file(source: Path, reference: Path, prediction: Path, prefix: Path) -> None:
    run([
        sys.executable,
        CODE / "score_gec.py",
        "--source", source,
        "--reference", reference,
        "--predictions", prediction,
        "--output-prefix", prefix,
    ])


def parse_score(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\n\d+\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text)
    if not match:
        raise ValueError(f"Could not parse score file: {path}")
    return {"precision": float(match.group(1)), "recall": float(match.group(2)), "f0.5": float(match.group(3))}


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; choose another --output-dir or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for required in (args.model, args.data_dir / "dev.src", args.data_dir / "dev.tgt", args.data_dir / "test.src", args.data_dir / "test.tgt"):
        if not required.exists():
            raise FileNotFoundError(required)

    candidate_paths: dict[str, Path] = {}
    common = [
        "--model", args.model.resolve(),
        "--device", args.device,
        "--batch-size", args.batch_size,
        "--num-beams", args.num_beams,
    ]
    if args.fp16:
        common.append("--fp16")
    if args.limit > 0:
        common.extend(["--limit", args.limit])
    for split in ("dev", "test"):
        candidate_path = output / split / "candidates.jsonl"
        candidate_paths[split] = candidate_path
        run([
            sys.executable,
            HERE / "generate_candidates.py",
            *common,
            "--input", (args.data_dir / f"{split}.src").resolve(),
            "--output", candidate_path,
        ])

    dev_rows = read_jsonl(candidate_paths["dev"])
    dev_references = read_lines(args.data_dir / "dev.tgt", args.limit)
    labels = [oracle_label(str(row["pass1"]), str(row["pass2"]), reference) for row, reference in zip(dev_rows, dev_references)]
    selector = fit_logistic((row["features"] for row in dev_rows), labels)
    dev_probabilities = [predict_probability(selector, row["features"]) for row in dev_rows]
    dev_predictions = [int(value >= float(selector["threshold"])) for value in dev_probabilities]
    selector["development_examples"] = len(labels)
    selector["development_accuracy"] = sum(a == b for a, b in zip(labels, dev_predictions)) / len(labels)
    selector["development_positive_labels"] = sum(labels)
    (output / "selector.json").write_text(json.dumps(selector, indent=2) + "\n", encoding="utf-8")

    test_rows = read_jsonl(candidate_paths["test"])
    test_references = read_lines(args.data_dir / "test.tgt", args.limit)
    systems: dict[str, list[str]] = {
        "pass1": [str(row["pass1"]) for row in test_rows],
        "pass2": [str(row["pass2"]) for row in test_rows],
        "pass3": [str(row["pass3"]) for row in test_rows],
    }
    stable = [stable_choice(row) for row in test_rows]
    systems["stable"] = [prediction for prediction, _ in stable]
    probabilities = [predict_probability(selector, row["features"]) for row in test_rows]
    selected_passes = [2 if probability >= float(selector["threshold"]) else 1 for probability in probabilities]
    systems["selective"] = [str(row[f"pass{chosen}"]) for row, chosen in zip(test_rows, selected_passes)]
    oracle_labels = [oracle_label(str(row["pass1"]), str(row["pass2"]), reference) for row, reference in zip(test_rows, test_references)]
    systems["distance_oracle"] = [
        str(row["pass2"] if label else row["pass1"])
        for row, label in zip(test_rows, oracle_labels)
    ]

    test_source = args.data_dir / "test.src"
    test_reference = args.data_dir / "test.tgt"
    if args.limit > 0:
        test_source = output / "test" / "limited.src"
        test_reference = output / "test" / "limited.tgt"
        write_lines(test_source, read_lines(args.data_dir / "test.src", args.limit))
        write_lines(test_reference, test_references)

    results: dict[str, dict[str, float]] = {}
    for name, predictions in systems.items():
        prediction_path = output / "predictions" / f"{name}.txt"
        write_lines(prediction_path, predictions)
        score_prefix = output / "scores" / name
        score_file(test_source, test_reference, prediction_path, score_prefix)
        results[name] = parse_score(score_prefix.with_suffix(".score.txt"))

    second_pass_better = sum(oracle_labels)
    second_pass_worse = sum(
        edit_distance(str(row["pass2"]), reference) > edit_distance(str(row["pass1"]), reference)
        for row, reference in zip(test_rows, test_references)
    )
    analysis = {
        "test_examples": len(test_rows),
        "pass1_equals_pass2": sum(row["pass1"] == row["pass2"] for row in test_rows),
        "pass2_equals_pass3": sum(row["pass2"] == row["pass3"] for row in test_rows),
        "second_pass_better_by_token_distance": second_pass_better,
        "second_pass_worse_by_token_distance": second_pass_worse,
        "selector_chose_pass2": sum(chosen == 2 for chosen in selected_passes),
        "average_stable_passes": sum(passes for _, passes in stable) / len(stable),
    }
    summary = {"scores": results, "analysis": analysis, "selector": selector}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Selective Multi-Pass Results",
        "",
        "| System | Precision | Recall | F0.5 |",
        "|---|---:|---:|---:|",
    ]
    for name, score in results.items():
        lines.append(f"| {name} | {score['precision']:.4f} | {score['recall']:.4f} | {score['f0.5']:.4f} |")
    lines.extend(["", "## Analysis", ""])
    lines.extend(f"- {key.replace('_', ' ').title()}: {value}" for key, value in analysis.items())
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nExperiment complete. Summary: {output / 'summary.md'}")


if __name__ == "__main__":
    main()

