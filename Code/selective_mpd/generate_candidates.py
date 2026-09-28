"""Generate three iterative GEC passes and confidence/edit features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mpd_core import make_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default="grammar: ")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def generate_pass(model, tokenizer, device, sentences, args):
    predictions: list[str] = []
    confidence: list[float] = []
    for start in range(0, len(sentences), args.batch_size):
        batch = [args.prefix + sentence for sentence in sentences[start : start + args.batch_size]]
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
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
        predictions.extend(decoded)
        if generated.sequences_scores is not None:
            confidence.extend(float(value) for value in generated.sequences_scores.detach().cpu())
        else:
            confidence.extend([0.0] * len(decoded))
    return predictions, confidence


def main() -> None:
    args = parse_args()
    sentences = args.input.read_text(encoding="utf-8").splitlines()
    if args.limit > 0:
        sentences = sentences[: args.limit]
    if not sentences:
        raise ValueError("Input contains no sentences")
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else None
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    start_time = time.perf_counter()
    pass1, score1 = generate_pass(model, tokenizer, device, sentences, args)
    pass2, score2 = generate_pass(model, tokenizer, device, pass1, args)
    pass3, score3 = generate_pass(model, tokenizer, device, pass2, args)
    elapsed = time.perf_counter() - start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, source in enumerate(sentences):
            row = {
                "index": index,
                "source": source,
                "pass1": pass1[index],
                "pass2": pass2[index],
                "pass3": pass3[index],
                "score1": score1[index],
                "score2": score2[index],
                "score3": score3[index],
                "features": make_features(source, pass1[index], pass2[index], score1[index], score2[index]),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "model": args.model,
        "sentences": len(sentences),
        "device": str(device),
        "fp16": bool(args.fp16 and device.type == "cuda"),
        "num_beams": args.num_beams,
        "three_pass_seconds": elapsed,
        "seconds_per_sentence_pass": elapsed / (3 * len(sentences)),
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Generated three passes for {len(sentences)} sentences in {elapsed:.2f}s")


if __name__ == "__main__":
    main()

