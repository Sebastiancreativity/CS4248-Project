"""Generate one GEC prediction per input line and optionally run an M2 scorer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small", help="Hub model ID or local checkpoint")
    parser.add_argument("--input", required=True, type=Path, help="One source sentence per line")
    parser.add_argument("--output", required=True, type=Path, help="Where predictions are written")
    parser.add_argument("--prefix", default="grammar: ", help="Training/inference prompt; use '' for none")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gold-m2", type=Path, help="Optional M2 gold file")
    parser.add_argument("--m2-scorer", type=Path, help="Path to the official m2scorer.py")
    parser.add_argument("--python", default=sys.executable, help="Python used to launch the M2 scorer")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is unavailable. "
            "Install a CUDA-enabled PyTorch build and check your NVIDIA driver."
        )
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if bool(args.gold_m2) != bool(args.m2_scorer):
        raise ValueError("Pass both --gold-m2 and --m2-scorer, or neither")

    sentences = args.input.read_text(encoding="utf-8").splitlines()
    device = choose_device(args.device)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Loading {args.model!r} on {device} ({device_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    predictions: list[str] = []
    for start in range(0, len(sentences), args.batch_size):
        batch = [args.prefix + text for text in sentences[start : start + args.batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                early_stopping=args.num_beams > 1,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        print(f"Generated {len(predictions)}/{len(sentences)}", end="\r")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    print(f"\nPredictions written to {args.output}")

    if args.gold_m2:
        # Official usage: python m2scorer.py SYSTEM_OUTPUT GOLD_M2
        command = [args.python, str(args.m2_scorer), str(args.output), str(args.gold_m2)]
        print("Running official M2 scorer ...")
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
