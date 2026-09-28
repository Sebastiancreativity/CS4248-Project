"""Prepare NUCLE splits, score base T5, and train/score a fine-tuned T5."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "baseline", "train-score", "all"])
    parser.add_argument("--source", type=Path, help="Source sentences; defaults to the selected NUCLE dataset")
    parser.add_argument("--target", type=Path, help="Corrected sentences; defaults to the selected NUCLE dataset")
    parser.add_argument("--work-dir", type=Path, help="Output folder; defaults to the selected NUCLE run")
    parser.add_argument(
        "--full-nucle", action="store_true",
        help="Use full NUCLE with unchanged sentences and a separate grouped run folder",
    )
    parser.add_argument("--test-size", type=int, default=200, help="Held-out scoring pairs (default: 200)")
    parser.add_argument("--dev-size", type=int, default=200, help="Development pairs (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed")
    parser.add_argument(
        "--group-by-source", action="store_true",
        help="Keep identical source sentences in the same split (recommended for full NUCLE)",
    )
    parser.add_argument("--base-model", default="t5-small", help="Base model, not yet fine-tuned for GEC")
    parser.add_argument("--prefix", default="grammar: ", help="Use the same prompt for both models")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true", help="Enable half precision for training on CUDA")
    parser.add_argument("--train-subset-size", type=int, default=1000, help="0 uses all remaining training pairs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--infer-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing model and predictions")
    args = parser.parse_args()
    if args.full_nucle:
        args.source = args.source or ROOT / "data/nucle-full/nucle.train.gold.bea19.src"
        args.target = args.target or ROOT / "data/nucle-full/nucle.train.gold.bea19.tgt"
        args.work_dir = args.work_dir or ROOT / "pipeline_runs/nucle-full-grouped"
        args.group_by_source = True
    else:
        args.source = args.source or ROOT / "data/nucle-error-only/nucle.train.gold.bea19.src"
        args.target = args.target or ROOT / "data/nucle-error-only/nucle.train.gold.bea19.tgt"
        args.work_dir = args.work_dir or ROOT / "pipeline_runs/nucle"
    return args


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_paths(work_dir: Path) -> dict[str, tuple[Path, Path]]:
    return {
        split: (work_dir / "data" / f"{split}.src", work_dir / "data" / f"{split}.tgt")
        for split in ("train", "dev", "test")
    }


def select_groups(groups: list[list[int]], size: int) -> tuple[list[int], list[list[int]]]:
    selected: list[int] = []
    remaining: list[list[int]] = []
    for group in groups:
        if len(selected) + len(group) <= size:
            selected.extend(group)
        else:
            remaining.append(group)
    if len(selected) != size:
        raise ValueError(f"Cannot make a {size}-pair split while keeping identical sources together")
    return sorted(selected), remaining


def prepare(args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    source, target = args.source.resolve(), args.target.resolve()
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError("NUCLE .src and .tgt files must both exist")
    if args.test_size < 1 or args.dev_size < 1:
        raise ValueError("--test-size and --dev-size must each be at least 1")
    sources = source.read_text(encoding="utf-8").splitlines()
    targets = target.read_text(encoding="utf-8").splitlines()
    if len(sources) != len(targets) or args.test_size + args.dev_size >= len(sources):
        raise ValueError(
            f"Need matching files and a nonempty training split: "
            f"source={len(sources)}, target={len(targets)}, "
            f"dev={args.dev_size}, test={args.test_size}"
        )

    work_dir = args.work_dir.resolve()
    paths = split_paths(work_dir)
    manifest_path = work_dir / "split.json"
    expected = {
        "source": str(source),
        "target": str(target),
        "source_sha256": digest(source),
        "target_sha256": digest(target),
        "seed": args.seed,
        "dev_size": args.dev_size,
        "test_size": args.test_size,
        "total_pairs": len(sources),
        "group_by_source": args.group_by_source,
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(
            saved.get(key, False if key == "group_by_source" else None) != value
            for key, value in expected.items()
        ):
            raise ValueError(
                f"Split settings or source files changed. Choose a new --work-dir; "
                f"existing split is at {work_dir}."
            )
        for split, (src, tgt) in paths.items():
            if not src.is_file() or not tgt.is_file() or (
                digest(src) != saved["split_sha256"][split]["src"]
                or digest(tgt) != saved["split_sha256"][split]["tgt"]
            ):
                raise ValueError(f"Existing {split} files changed or are missing: {work_dir}")
        print(f"Using existing split in {work_dir}", flush=True)
        return paths
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError(f"{work_dir} is not empty and has no split.json; choose a new --work-dir")

    if args.group_by_source:
        by_source: dict[str, list[int]] = {}
        for index, sentence in enumerate(sources):
            by_source.setdefault(sentence, []).append(index)
        source_groups = list(by_source.values())
        random.Random(args.seed).shuffle(source_groups)
        test_indices, source_groups = select_groups(source_groups, args.test_size)
        dev_indices, source_groups = select_groups(source_groups, args.dev_size)
        train_indices = sorted(index for group in source_groups for index in group)
    else:
        indices = list(range(len(sources)))
        random.Random(args.seed).shuffle(indices)
        test_indices = sorted(indices[: args.test_size])
        dev_indices = sorted(indices[args.test_size : args.test_size + args.dev_size])
        train_indices = sorted(indices[args.test_size + args.dev_size :])
    groups = {"train": train_indices, "dev": dev_indices, "test": test_indices}
    checksums = {}
    for split, selected in groups.items():
        src, tgt = paths[split]
        write_lines(src, [sources[index] for index in selected])
        write_lines(tgt, [targets[index] for index in selected])
        checksums[split] = {"src": digest(src), "tgt": digest(tgt)}
        print(f"{split}: {len(selected)} sentence pairs", flush=True)
    manifest_path.write_text(
        json.dumps({**expected, "split_sha256": checksums}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Split saved in {work_dir}", flush=True)
    return paths


def run_script(script: str, *options: object) -> None:
    command = [sys.executable, str(ROOT / script), *(str(option) for option in options)]
    print(f"Running {script} ...", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def assert_free(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")


def predict_and_score(
    args: argparse.Namespace,
    test: tuple[Path, Path],
    model: str | Path,
    name: str,
) -> None:
    source, target = test
    work_dir = args.work_dir.resolve()
    predictions = work_dir / "predictions" / f"{name}.txt"
    score_prefix = work_dir / "scores" / name
    assert_free(predictions, args.overwrite)
    if score_prefix.with_suffix(".score.txt").exists() and not args.overwrite:
        raise FileExistsError(f"Score for {name} already exists. Pass --overwrite to replace it.")
    run_script(
        "run_gec.py", "--model", model, "--input", source, "--output", predictions,
        "--prefix", args.prefix, "--batch-size", args.infer_batch_size, "--device", args.device,
    )
    run_script(
        "score_gec.py", "--source", source, "--reference", target,
        "--predictions", predictions, "--output-prefix", score_prefix,
    )


def train_and_score(args: argparse.Namespace, paths: dict[str, tuple[Path, Path]]) -> None:
    model_dir = args.work_dir.resolve() / "model"
    assert_free(model_dir, args.overwrite)
    train_src, train_tgt = paths["train"]
    dev_src, dev_tgt = paths["dev"]
    options: list[object] = [
        "--model", args.base_model,
        "--train-src", train_src, "--train-tgt", train_tgt,
        "--eval-src", dev_src, "--eval-tgt", dev_tgt,
        "--eval-subset-size", 0,
        "--output-dir", model_dir,
        "--prefix", args.prefix,
        "--subset-size", args.train_subset_size,
        "--epochs", args.epochs,
        "--batch-size", args.train_batch_size,
        "--gradient-accumulation-steps", args.gradient_accumulation_steps,
        "--seed", args.seed,
        "--device", args.device,
    ]
    if args.fp16:
        options.append("--fp16")
    run_script("train_gec.py", *options)
    predict_and_score(args, paths["test"], model_dir, "trained")


def main() -> None:
    args = parse_args()
    paths = prepare(args)
    if args.action == "prepare":
        return
    from score_gec import check_scoring_setup, errant_command

    check_scoring_setup()
    errant_command("errant_parallel")
    errant_command("errant_compare")
    if args.action in ("baseline", "all"):
        predict_and_score(args, paths["test"], args.base_model, "baseline")
    if args.action in ("train-score", "all"):
        train_and_score(args, paths)


if __name__ == "__main__":
    main()
