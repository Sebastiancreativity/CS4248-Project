"""Targeted repair: a flagged selection is re-translated with its problems named in the prompt and added to the pool as one more candidate."""
import torch

from .text import HAN, clean_output

PROMPT = ("Translate the following Chinese text into English. Output only the English translation.\n\n"
          "Chinese: {src}\n\nA previous translation was:\n{draft}\n\n"
          "Problems with it: {problems}.\nWrite a corrected, complete English translation.")


def uncovered_spans(cov, tau=0.3, min_han=2, top=2):
    spans, cur = [], []
    for tok, c in zip(cov["tokens"], cov["cov"]):
        piece = tok.replace("▁", "")
        if c < tau and HAN.search(piece):
            cur.append((piece, c))
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    spans = [("".join(p for p, _ in sp), sum(c for _, c in sp) / len(sp)) for sp in spans]
    spans = [s for s in spans if len(HAN.findall(s[0])) >= min_han]
    return [t for t, _ in sorted(spans, key=lambda s: s[1])[:top]]


def problems(tags, detail, spans):
    msgs = []
    if "untranslated_han" in tags:
        msgs.append("it leaves some Chinese untranslated")
    if "number_missing" in tags:
        msgs.append("it omits the number(s) " + ", ".join(detail["numbers"]))
    if "latin_missing" in tags:
        msgs.append("it omits " + ", ".join(detail["latin"]))
    if "too_short" in tags:
        msgs.append("it is much shorter than the source and probably leaves content out")
    if tags & {"empty", "meta_text", "repetition"}:
        msgs.append("it is not a clean translation")
    msgs += [f"it may leave out the part “{s}”" for s in spans]
    return msgs


class HFChatGenerator:

    def __init__(self, model_id, device="cpu", dtype="bfloat16", batch_size=8, max_new_tokens=256):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
        self.device, self.batch_size, self.max_new_tokens = device, batch_size, max_new_tokens

    @torch.inference_mode()
    def generate(self, prompts):
        out = []
        for b in range(0, len(prompts), self.batch_size):
            chat = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                 add_generation_prompt=True) for p in prompts[b:b + self.batch_size]]
            enc = self.tok(chat, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            gen = self.model.generate(**enc, do_sample=False, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  [repair] {len(out)}/{len(prompts)}", end="\r", flush=True)
        print()
        return out


def repair(generator, items):
    prompts = [PROMPT.format(src=s, draft=d, problems="; ".join(p)) for s, d, p in items]
    return [clean_output(r, s) for r, (s, _, _) in zip(generator.generate(prompts), items)]
