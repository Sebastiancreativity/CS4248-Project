"""Encoder-decoder auditors: forward and reverse forced-decoding likelihood and cross-attention coverage."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .cache import Cache, key

_LOADED = {}


def _model(model_id, device, dtype):
    k = (model_id, device, dtype)
    if k not in _LOADED:
        m = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=getattr(torch, dtype), attn_implementation="eager")
        _LOADED[k] = m.to(device).eval()
    return _LOADED[k]


class Scorer:
    def __init__(self, name, model_id, src_lang=None, tgt_lang=None, prefix="", device="cpu", dtype="bfloat16",
                 batch_size=16, cache=None):
        self.name, self.model_id, self.prefix = name, model_id, prefix
        self.device, self.dtype, self.batch_size = device, dtype, batch_size
        kw = {"src_lang": src_lang, "tgt_lang": tgt_lang} if src_lang else {}
        self.tok = AutoTokenizer.from_pretrained(model_id, **kw)
        self.cache = cache or Cache()

    def score(self, pairs):
        keys = [key(s, h) for s, h in pairs]
        have = self.cache.get_many(self.name, set(keys))
        todo = {k: p for k, p in zip(keys, pairs) if k not in have}
        if todo:
            items = sorted(todo.items(), key=lambda kv: -len(kv[1][0]) - len(kv[1][1]))
            model = _model(self.model_id, self.device, self.dtype)
            for b in range(0, len(items), self.batch_size):
                chunk = items[b:b + self.batch_size]
                with torch.inference_mode():
                    vals = self._compute(model, [p for _, p in chunk])
                done = {k: v for (k, _), v in zip(chunk, vals)}
                self.cache.put_many(self.name, done)
                have.update(done)
                print(f"  [{self.name}] {min(b + self.batch_size, len(items))}/{len(items)}", end="\r", flush=True)
            print()
        return [have[k] for k in keys]

    def _encode(self, inputs, targets):
        enc = self.tok([self.prefix + x for x in inputs], text_target=[t if t.strip() else "." for t in targets],
                       return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        labels = enc["labels"].masked_fill(enc["labels"].eq(self.tok.pad_token_id), -100)
        return enc, labels


class Likelihood(Scorer):

    def __init__(self, name, model_id, reverse=False, **kw):
        super().__init__(name, model_id, **kw)
        self.reverse = reverse

    def _compute(self, model, pairs):
        inputs = [h for s, h in pairs] if self.reverse else [s for s, h in pairs]
        targets = [s for s, h in pairs] if self.reverse else [h for s, h in pairs]
        enc, labels = self._encode(inputs, targets)
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).logits
        lp = logits.float().log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        ok = labels.ne(-100)
        return [round(float(lp[b][ok[b]].mean()), 4) for b in range(len(pairs))]


class Coverage(Scorer):

    def __init__(self, name, model_id, layer=-2, **kw):
        super().__init__(name, model_id, **kw)
        self.layer = layer
        self.special = set(self.tok.all_special_ids)

    def _compute(self, model, pairs):
        enc, labels = self._encode([s for s, h in pairs], [h for s, h in pairs])
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels,
                    output_attentions=True)
        A = out.cross_attentions[self.layer].float().mean(1)
        res = []
        for b in range(len(pairs)):
            s_ids, t_ids = enc["input_ids"][b], enc["labels"][b]
            s_keep = enc["attention_mask"][b].bool() & torch.tensor(
                [int(t) not in self.special for t in s_ids.tolist()], device=s_ids.device)
            t_keep = labels[b].ne(-100) & torch.tensor(
                [int(t) not in self.special for t in t_ids.tolist()], device=t_ids.device)
            a = A[b][t_keep][:, s_keep]
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
            cov = a.sum(0)
            toks = self.tok.convert_ids_to_tokens(s_ids[s_keep])
            if self.prefix:
                toks = toks[len(self.tok(self.prefix, add_special_tokens=False)["input_ids"]):]
                cov = cov[-len(toks):] if toks else cov[:0]
            res.append(dict(pen=round(float(cov.clamp(max=1).clamp_min(1e-6).log().mean()), 4) if len(cov) else 0.0,
                            tokens=toks, cov=[round(float(x), 3) for x in cov]))
        return res


def build(cfg, device="cpu"):
    kw = {k: cfg[k] for k in ("src_lang", "tgt_lang", "prefix", "batch_size", "dtype") if k in cfg}
    if cfg["type"] == "coverage":
        return Coverage(cfg["name"], cfg["model"], layer=cfg.get("layer", -2), device=device, **kw)
    return Likelihood(cfg["name"], cfg["model"], reverse=cfg.get("reverse", False), device=device, **kw)

