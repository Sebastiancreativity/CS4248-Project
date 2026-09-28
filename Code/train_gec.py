"""Fine-tune a Hugging Face seq2seq model on a small parallel GEC subset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small", help="Hub model ID or local checkpoint")
    parser.add_argument("--train-src", required=True, type=Path)
    parser.add_argument("--train-tgt", required=True, type=Path)
    parser.add_argument("--eval-src", type=Path)
    parser.add_argument("--eval-tgt", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/t5-small-gec"))
    parser.add_argument("--prefix", default="grammar: ")
    parser.add_argument("--subset-size", type=int, default=1000, help="0 means use all examples")
    parser.add_argument("--eval-subset-size", type=int, default=500, help="0 means use all")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. 'auto' uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use on CUDA GPUs with fp16 support")
    parser.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows")
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


def read_parallel(src_path: Path, tgt_path: Path) -> list[tuple[str, str]]:
    sources = src_path.read_text(encoding="utf-8").splitlines()
    targets = tgt_path.read_text(encoding="utf-8").splitlines()
    if len(sources) != len(targets):
        raise ValueError(f"Line count mismatch: {src_path}={len(sources)}, {tgt_path}={len(targets)}")
    return list(zip(sources, targets))


def take_subset(examples: list[tuple[str, str]], size: int, seed: int) -> list[tuple[str, str]]:
    if size <= 0 or size >= len(examples):
        return examples
    rng = random.Random(seed)
    return rng.sample(examples, size)


class GECDataset(Dataset):
    def __init__(self, examples, tokenizer, prefix, max_source_length, max_target_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        source, target = self.examples[index]
        model_inputs = self.tokenizer(
            self.prefix + source, truncation=True, max_length=self.max_source_length
        )
        labels = self.tokenizer(text_target=target, truncation=True, max_length=self.max_target_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if bool(args.eval_src) != bool(args.eval_tgt):
        raise ValueError("Pass both --eval-src and --eval-tgt, or neither")
    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 requires CUDA; use --device cuda or --device auto with a CUDA GPU")

    set_seed(args.seed)
    train_examples = take_subset(
        read_parallel(args.train_src, args.train_tgt), args.subset_size, args.seed
    )
    eval_examples = None
    if args.eval_src:
        eval_examples = take_subset(
            read_parallel(args.eval_src, args.eval_tgt), args.eval_subset_size, args.seed
        )

    print(f"Training examples: {len(train_examples)}")
    print(f"Evaluation examples: {len(eval_examples) if eval_examples else 0}")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Training device: {device} ({device_name})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    train_dataset = GECDataset(
        train_examples, tokenizer, args.prefix, args.max_source_length, args.max_target_length
    )
    eval_dataset = (
        GECDataset(eval_examples, tokenizer, args.prefix, args.max_source_length, args.max_target_length)
        if eval_examples
        else None
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=False,
        fp16=args.fp16,
        use_cpu=device.type == "cpu",
        dataloader_num_workers=args.num_workers,
        report_to="none",
        save_total_limit=2,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in result.metrics.items()}
    (args.output_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "train_examples": len(train_examples), "metrics": metrics}, default=str, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
