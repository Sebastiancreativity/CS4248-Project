"""Supervised Chinese-to-English examples. Gold text is used only as labels."""
from model_registry import PROMPT, build_prompt


def encode_pair(tok, cfg, source, target, max_source_length=512, max_target_length=512):
    prompt = build_prompt(tok, cfg, source)
    if cfg.get("seq2seq"):
        inputs = tok(prompt, add_special_tokens=True)["input_ids"]
        # NLLB's tokenizer is configured with src_lang=zho_Hans, tgt_lang=eng_Latn.
        labels = tok(text_target=target, add_special_tokens=True)["input_ids"]
    else:
        inputs = tok(prompt, add_special_tokens=False)["input_ids"]
        if cfg.get("chat", True) and tok.chat_template:
            full = tok.apply_chat_template([
                {"role": "user", "content": cfg.get("prompt", PROMPT).format(src=source)},
                {"role": "assistant", "content": target}],
                tokenize=False, add_generation_prompt=False)
            if not full.startswith(prompt):
                raise ValueError("Chat training prefix differs from generation prefix")
            completion = full[len(prompt):]
        else:
            completion = target + tok.eos_token
        labels = tok(completion, add_special_tokens=False)["input_ids"]
    # Never truncate one half of a translation pair into an inconsistent example.
    if len(inputs) > max_source_length or len(labels) > max_target_length:
        return None
    if not inputs or not labels:
        raise ValueError("Tokenizer produced an empty source or target")
    if not cfg.get("seq2seq"):
        labels = [-100] * len(inputs) + labels
        inputs = inputs + labels[len(inputs):]
    return {"input_ids": inputs, "attention_mask": [1] * len(inputs), "labels": labels}


def encode_dataset(tok, cfg, sources, targets, **lengths):
    if len(sources) != len(targets):
        raise ValueError("Source/target counts differ")
    examples, skipped = [], []
    for row, (source, target) in enumerate(zip(sources, targets), 1):
        encoded = encode_pair(tok, cfg, source, target, **lengths)
        if encoded is None:
            skipped.append(row)
        else:
            examples.append(encoded)
    if not examples:
        raise ValueError("No usable examples remain after the token-length limit")
    return examples, {"available": len(sources), "used": len(examples), "overlength_rows": skipped}


class TranslationCollator:
    """Right-pad labels by length; preserve EOS even when EOS is also PAD."""
    def __init__(self, tokenizer, model=None):
        self.tokenizer, self.model = tokenizer, model

    def __call__(self, features):
        import torch
        batch = self.tokenizer.pad(
            [{k: v for k, v in f.items() if k != "labels"} for f in features],
            padding=True, pad_to_multiple_of=8, return_tensors="pt")
        width = ((max(len(f["labels"]) for f in features) + 7) // 8) * 8
        batch["labels"] = torch.tensor(
            [f["labels"] + [-100] * (width - len(f["labels"])) for f in features], dtype=torch.long)
        # T5 exposes this helper; M2M100/NLLB shifts labels inside forward().
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(batch["labels"])
        return batch
