"""Optional real-library smoke checks, without downloading translation weights.

python tests/smoke_finetune_models.py
python tests/smoke_finetune_models.py --tokenizers  # downloads tokenizer files only
"""
import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from model_registry import MODELS, TEAM_MODELS
from finetune_data import TranslationCollator, encode_pair


def check_tokenizers():
    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/finetune/tokenizer_validation"))
    from transformers import AutoTokenizer
    for name in TEAM_MODELS:
        cfg = MODELS[name]
        kwargs = {"src_lang": cfg["src_lang"], "tgt_lang": cfg["tgt_lang"]} if cfg.get("src_lang") else {}
        tok = AutoTokenizer.from_pretrained(cfg["hf"], padding_side="right", **kwargs)
        example = encode_pair(tok, cfg, "这是一个测试。", "This is a test.")
        assert example is not None
        supervised = [t for t in example["labels"] if t != -100]
        assert "This is a test." in tok.decode(supervised, skip_special_tokens=True)
        if cfg.get("tgt_lang"):
            assert tok.convert_tokens_to_ids("eng_Latn") in supervised
            assert tok.convert_tokens_to_ids("zho_Hans") in example["input_ids"]
        print(f"PASS real tokenizer/prompt/English labels: {name}", flush=True)


def check_models():
    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                              LlamaConfig, Qwen2Config, MistralConfig, HunYuanDenseV1Config,
                              T5Config, M2M100Config, PreTrainedTokenizerFast, Trainer, TrainingArguments)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    torch.set_num_threads(1)
    args = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False)
    configs = {
        "qwen2.5-1.5b": Qwen2Config(**args), "qwen2.5-3b": Qwen2Config(**args),
        "qwen2.5-7b": Qwen2Config(**args), "hunyuan-mt-7b": HunYuanDenseV1Config(**args),
        "tower-7b": LlamaConfig(**args), "tower-mistral-7b": MistralConfig(**args),
        "madlad400-3b": T5Config(vocab_size=64, d_model=32, d_ff=64, d_kv=8, num_layers=1,
                                  num_decoder_layers=1, num_heads=4, decoder_start_token_id=0,
                                  pad_token_id=0, eos_token_id=2, dropout_rate=0, use_cache=False),
        "nllb-1.3b": M2M100Config(vocab_size=64, d_model=32, encoder_layers=1, decoder_layers=1,
                                  encoder_attention_heads=4, decoder_attention_heads=4,
                                  encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=64,
                                  decoder_start_token_id=2, pad_token_id=0, eos_token_id=2,
                                  dropout=0, attention_dropout=0, activation_dropout=0, use_cache=False),
    }
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="3")),
                                 pad_token="0", eos_token="2", unk_token="3", padding_side="right")
    for name, config in configs.items():
        seq2seq = bool(MODELS[name].get("seq2seq"))
        cls = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM
        model = cls.from_config(config, attn_implementation="eager")
        pristine = copy.deepcopy(model)
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if seq2seq else TaskType.CAUSAL_LM,
            r=4, lora_alpha=8, lora_dropout=0, target_modules="all-linear", bias="none"))
        inputs = [10, 11, 2] if seq2seq else [10, 11, 5, 6, 2]
        labels = [5, 6, 2] if seq2seq else [-100, -100, 5, 6, 2]
        data = [{"input_ids": inputs, "attention_mask": [1] * len(inputs), "labels": labels}] * 2
        collator = TranslationCollator(tok, model if seq2seq else None)
        batch = collator(data)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as directory:
            training_args = TrainingArguments(
                output_dir=directory, use_cpu=True, num_train_epochs=2, learning_rate=0.01,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
                label_names=["labels"], remove_unused_columns=False, prediction_loss_only=True,
                report_to="none", disable_tqdm=True, dataloader_pin_memory=False)
            trainer = Trainer(model=model, args=training_args, train_dataset=data,
                              eval_dataset=data, processing_class=tok, data_collator=collator)
            trainer.train()
            assert trainer.state.global_step == 4
            assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)
            assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
            adapter = Path(directory) / "adapter"
            trainer.save_model(adapter)
            model.eval()
            with torch.no_grad():
                expected = model(**batch).logits
                restored = PeftModel.from_pretrained(pristine, adapter).eval()
                actual = restored(**batch).logits
            torch.testing.assert_close(expected, actual)
        print(f"PASS real Trainer, gradient update, best checkpoint and adapter reload: {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizers", action="store_true")
    options = parser.parse_args()
    check_tokenizers() if options.tokenizers else check_models()
