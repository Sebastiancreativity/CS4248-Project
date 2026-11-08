"""Fine-tune one assigned model; invoked once per GPU by the Slurm array."""
import argparse
import math
from importlib.metadata import version
from pathlib import Path

from finetune_common import (adapter_record, load_run, read_json, read_parallel,
                             sha256, write_json)
from finetune_data import TranslationCollator, encode_dataset
from model_registry import MODELS, TEAM_MODELS


def train(run_dir, model_name):
    from filelock import FileLock
    output = run_dir / "training" / model_name
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "training.lock"), timeout=0):
        return train_locked(run_dir, model_name, output)


def train_locked(run_dir, model_name, output):
    run = load_run(run_dir)
    spec = run["spec"]
    settings = spec["training"]
    adapter_dir = run_dir / "adapters" / model_name
    if (adapter_dir / "training_complete.json").exists():
        record = adapter_record(adapter_dir, model_name)
        if record["run_fingerprint"] != run["fingerprint"]:
            raise ValueError("Adapter belongs to a different training run")
        print(f"Already trained: {model_name}; verified saved adapter.", flush=True)
        return

    # Cache configuration must precede Hugging Face imports.
    import cache_paths  # noqa: F401
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                              AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Fine-tuning requires a CUDA GPU with BF16 support (e.g. A100/H100)")
    set_seed(settings["seed"])
    cfg = MODELS[model_name]
    # Preserve the exact base revision across a resumed run and adapter inference.
    base_path = output / "base_revision.json"
    if base_path.exists():
        base = read_json(base_path)
        if base["run_fingerprint"] != run["fingerprint"]:
            raise ValueError("Training directory belongs to a different run")
        revision = base["revision"]
    else:
        revision = AutoConfig.from_pretrained(cfg["hf"])._commit_hash
        if not revision:
            raise ValueError("Could not resolve an immutable base-model revision")
        write_json(base_path, {"revision": revision, "run_fingerprint": run["fingerprint"]})
    tok_args = {"src_lang": cfg["src_lang"], "tgt_lang": cfg["tgt_lang"]} if cfg.get("src_lang") else {}
    tok = AutoTokenizer.from_pretrained(cfg["hf"], revision=revision, padding_side="right", **tok_args)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lengths = {name: settings[name] for name in ("max_source_length", "max_target_length")}
    data, counts = {}, {}
    for split in ("train", "dev_train20"):
        src, ref = read_parallel(spec["split_dir"], split)
        data[split], counts[split] = encode_dataset(tok, cfg, src, ref, **lengths)
    write_json(output / "tokenization.json", counts)
    print(f"{model_name}: training pairs={counts['train']['used']}, "
          f"validation pairs={counts['dev_train20']['used']}; overlength pairs recorded, not truncated.", flush=True)

    quantized = not cfg.get("seq2seq", False)
    model_class = AutoModelForCausalLM if quantized else AutoModelForSeq2SeqLM
    load_args = dict(revision=revision, dtype=torch.bfloat16, device_map={"": 0})
    if quantized:
        load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = model_class.from_pretrained(cfg["hf"], **load_args)
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM if quantized else TaskType.SEQ_2_SEQ_LM,
        r=settings["lora_rank"], lora_alpha=2 * settings["lora_rank"], lora_dropout=0.05,
        target_modules="all-linear", bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    trainable, total = model.get_nb_trainable_parameters()
    if not 0 < trainable < total:
        raise ValueError("Expected trainable LoRA parameters and frozen base weights")
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=settings["epochs"],
        learning_rate=settings["learning_rate"], per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=settings["gradient_accumulation"],
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.01, max_grad_norm=1.0, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=10, report_to="none", seed=settings["seed"],
        data_seed=settings["seed"], prediction_loss_only=True, label_names=["labels"],
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = Trainer(
        model=model, args=args, train_dataset=data["train"], eval_dataset=data["dev_train20"],
        processing_class=tok, data_collator=TranslationCollator(tok, model if not quantized else None))
    last = get_last_checkpoint(args.output_dir) if Path(args.output_dir).exists() else None
    if last:
        print(f"Resuming {model_name} from {last}", flush=True)
    result = trainer.train(resume_from_checkpoint=last)
    best_loss = trainer.state.best_metric
    if trainer.state.global_step < 1 or best_loss is None or not math.isfinite(best_loss):
        raise RuntimeError("Training did not produce a finite validation loss and a best checkpoint")
    # Trainer has restored the checkpoint with the lowest validation loss.
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(adapter_dir)
    trainer.save_state()
    write_json(output / "history.json", trainer.state.log_history)
    record = dict(
        model=model_name, base_model=cfg["hf"], base_revision=revision, quantized=quantized,
        method="QLoRA" if quantized else "LoRA", run_fingerprint=run["fingerprint"],
        global_step=trainer.state.global_step, best_checkpoint=trainer.state.best_model_checkpoint,
        best_dev_loss=best_loss, checkpoint_selection="minimum dev_train20 loss",
        trainable_parameters=trainable, total_parameters=total, tokenization=counts,
        training_metrics=result.metrics,
        versions={name: version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
        files={p.name: sha256(p) for p in adapter_dir.iterdir()
               if p.is_file() and p.name != "training_complete.json" and not p.name.endswith(".tmp")})
    write_json(adapter_dir / "training_complete.json", record)
    print(f"Saved fine-tuned {model_name}: {adapter_dir}; best dev loss={best_loss:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=TEAM_MODELS)
    group.add_argument("--model-index", type=int, choices=range(len(TEAM_MODELS)))
    args = parser.parse_args()
    train(args.run_dir.resolve(), args.model if args.model else TEAM_MODELS[args.model_index])


if __name__ == "__main__":
    main()
