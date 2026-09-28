"""Score parallel GEC predictions with ERRANT (precision, recall, F0.5)."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original sentences, one per line")
    parser.add_argument("--reference", required=True, type=Path, help="Gold corrections, one per line")
    parser.add_argument("--predictions", required=True, type=Path, help="Model output, one per line")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Prefix for M2 and score files")
    parser.add_argument(
        "--pretokenized",
        action="store_true",
        help="Skip spaCy tokenization only when all three files already use identical tokenization",
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return len(path.read_text(encoding="utf-8").splitlines())


def errant_command(name: str) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    command = Path(sys.executable).parent / f"{name}{suffix}"
    if not command.is_file():
        raise RuntimeError(
            f"{name} is missing from this Python environment. Install scoring dependencies with "
            f"'{sys.executable} -m pip install -r requirements-scoring.txt'."
        )
    return command


def check_scoring_setup() -> None:
    if importlib.util.find_spec("errant") is None or importlib.util.find_spec("spacy") is None:
        raise RuntimeError(
            "ERRANT is not installed. Run 'python -m pip install -r requirements-scoring.txt' "
            "using this project's .venv Python."
        )
    import spacy

    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "The spaCy English model is missing. Run 'python -m spacy download en_core_web_sm' "
            "using this project's .venv Python."
        )


def main() -> None:
    args = parse_args()
    counts = {name: count_lines(path) for name, path in (
        ("source", args.source),
        ("reference", args.reference),
        ("predictions", args.predictions),
    )}
    if len(set(counts.values())) != 1 or counts["source"] == 0:
        raise ValueError(f"Source, reference and predictions must have the same nonzero line count: {counts}")

    check_scoring_setup()
    parallel = errant_command("errant_parallel")
    compare = errant_command("errant_compare")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    gold_m2 = Path(f"{args.output_prefix}.gold.m2")
    pred_m2 = Path(f"{args.output_prefix}.pred.m2")
    score_file = Path(f"{args.output_prefix}.score.txt")
    tokenization = [] if args.pretokenized else ["-tok"]

    print(f"Scoring {counts['source']} aligned sentences with ERRANT", flush=True)
    subprocess.run(
        [str(parallel), *tokenization, "-orig", str(args.source), "-cor", str(args.reference), "-out", str(gold_m2)],
        check=True,
    )
    subprocess.run(
        [str(parallel), *tokenization, "-orig", str(args.source), "-cor", str(args.predictions), "-out", str(pred_m2)],
        check=True,
    )
    result = subprocess.run(
        [str(compare), "-hyp", str(pred_m2), "-ref", str(gold_m2)],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    score_file.write_text(result.stdout, encoding="utf-8")
    print(f"Score saved to {score_file}")


if __name__ == "__main__":
    main()
