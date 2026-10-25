"""Screen one open-source model on the dev sets for the MBR candidate pool.

Usage:
  python official/scripts/screen_model.py --model qwen2.5-1.5b
  python official/scripts/screen_model.py --model qwen2.5-7b --dev dev_wmt --limit 200
"""
import argparse
import gc
import json
import re
import time
from datetime import datetime
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "dev"
RES = ROOT / "results" / "screening"

MODELS = {
    "qwen2.5-1.5b": dict(hf="Qwen/Qwen2.5-1.5B-Instruct"),
    "qwen2.5-3b": dict(hf="Qwen/Qwen2.5-3B-Instruct"),
    "qwen2.5-7b": dict(hf="Qwen/Qwen2.5-7B-Instruct", four_bit=True),
    "hunyuan-mt-7b": dict(hf="tencent/Hunyuan-MT-7B", four_bit=True),
    "tower-7b": dict(hf="Unbabel/TowerInstruct-7B-v0.2", four_bit=True),
    "alma-7b": dict(hf="haoranxu/ALMA-7B", four_bit=True, chat=False),
    "madlad400-3b": dict(hf="google/madlad400-3b-mt", seq2seq=True, target_prefix="<2en> "),
    "nllb-1.3b": dict(hf="facebook/nllb-200-distilled-1.3B", seq2seq=True, src_lang="zho_Hans", tgt_lang="eng_Latn"),
}

PROMPT = "Translate the following Chinese text into English. Output only the English translation.\n\nChinese: {src}"
PLAIN_PROMPT = "Translate this from Chinese into English:\nChinese: {src}\nEnglish:"
CJK = re.compile(r"[一-鿿]")
PREFIX = re.compile(r"^((here\s*(is|'s)\s*)?(the\s*)?(english\s*)?(translation|english)|英文|译文)\s*[:：]\s*", re.I)


def read(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def load(cfg):
    tok_kwargs = {"padding_side": "right" if cfg.get("seq2seq") else "left"}
    if cfg.get("src_lang"):
        tok_kwargs["src_lang"] = cfg["src_lang"]
    tok = AutoTokenizer.from_pretrained(cfg["hf"], **tok_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if cfg.get("four_bit"):
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        cls = AutoModelForSeq2SeqLM if cfg.get("seq2seq") else AutoModelForCausalLM
        model = cls.from_pretrained(cfg["hf"], quantization_config=q, device_map="cuda")
    else:
        cls = AutoModelForSeq2SeqLM if cfg.get("seq2seq") else AutoModelForCausalLM
        model = cls.from_pretrained(cfg["hf"], dtype=torch.bfloat16).to("cuda")
    return tok, model.eval()


def build_prompt(tok, cfg, src):
    if cfg.get("seq2seq"):
        return cfg.get("target_prefix", "") + src
    if cfg.get("chat", True) and tok.chat_template:
        return tok.apply_chat_template([{"role": "user", "content": PROMPT.format(src=src)}],
                                       tokenize=False, add_generation_prompt=True)
    return PLAIN_PROMPT.format(src=src)


def clean(raw, src):
    lines = [PREFIX.sub("", l.strip()) for l in raw.strip().split("\n")]
    lines = [l for l in lines if l]
    out = lines[0] if lines else ""
    if len(out) > 1 and out[0] == out[-1] and out[0] in "\"'“”" and src[:1] not in "\"“'‘「":
        out = out[1:-1].strip()
    return out


def repeats(text, n=3, times=3):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return any(grams.count(g) >= times for g in set(grams))


@torch.inference_mode()
def translate(tok, model, cfg, src, batch_size, max_new_tokens):
    # Longest-first batching keeps padding low; order is restored afterwards.
    order = sorted(range(len(src)), key=lambda i: -len(src[i]))
    raw = [None] * len(src)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([build_prompt(tok, cfg, src[i]) for i in idx], return_tensors="pt",
                  padding=True, add_special_tokens=bool(cfg.get("seq2seq"))).to("cuda")
        kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": max_new_tokens,
                  "pad_token_id": tok.pad_token_id}
        if cfg.get("tgt_lang"):
            kwargs["forced_bos_token_id"] = tok.convert_tokens_to_ids(cfg["tgt_lang"])
        gen = model.generate(**enc, **kwargs)
        output = gen if cfg.get("seq2seq") else gen[:, enc["input_ids"].shape[1]:]
        for i, g in zip(idx, tok.batch_decode(output, skip_special_tokens=True)):
            raw[i] = g
        print(f"  {min(b + batch_size, len(src))}/{len(src)}", end="\r", flush=True)
    print()
    return raw


def gpu_power_limit_w():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else None


def comet22(src, hyp, ref):
    from comet import download_model, load_from_checkpoint
    m = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
    return round(100 * m.predict(data, batch_size=64, gpus=1, progress_bar=False).system_score, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dev", nargs="+", default=["dev_wmt", "dev_short"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="re-clean saved raw outputs and re-score; no generation")
    a = ap.parse_args()
    cfg = MODELS[a.model]
    out_dir = RES / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.rescore:
        return rescore(a, cfg, out_dir)

    power_w = gpu_power_limit_w()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    tok, model = load(cfg)
    load_s = time.time() - t_load

    # Generation for all dev sets first, then the LLM is freed before COMET loads.
    runs = []
    for dev in a.dev:
        src, ref = read(DEV / f"{dev}.zh"), read(DEV / f"{dev}.en")
        if a.limit:
            src, ref = src[:a.limit], ref[:a.limit]
        print(f"[{a.model}] {dev}: {len(src)} sentences")
        t0 = time.time()
        raw = translate(tok, model, cfg, src, a.batch_size, a.max_new_tokens)
        secs = time.time() - t0
        hyp = [clean(r, s) for r, s in zip(raw, src)]
        (out_dir / f"{dev}.raw.jsonl").write_text(
            "\n".join(json.dumps({"src": s, "raw": r}, ensure_ascii=False) for s, r in zip(src, raw)) + "\n",
            encoding="utf-8")
        (out_dir / f"{dev}.hyp").write_text("\n".join(hyp) + "\n", encoding="utf-8")
        runs.append(dict(dev=dev, src=src, ref=ref, hyp=hyp, secs=secs))
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    del model
    gc.collect()
    torch.cuda.empty_cache()

    for r in runs:
        run_info = dict(sent_per_s=round(len(r["hyp"]) / r["secs"], 2), gen_seconds=round(r["secs"], 1),
                        load_seconds=round(load_s, 1), peak_gpu_gb=round(peak_gb, 2),
                        gpu_power_limit_w=power_w, batch_size=a.batch_size, limit=a.limit)
        log_row(a, cfg, r["dev"], r["src"], r["ref"], r["hyp"], run_info)


def log_row(a, cfg, dev, src, ref, hyp, run_info):
    n = len(hyp)
    row = dict(
        model=a.model, hf=cfg["hf"], dev=dev, n=n,
        bleu=round(sacrebleu.corpus_bleu(hyp, [ref], tokenize="13a").score, 2),
        chrfpp=round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 2),
        comet22=None if a.no_comet else comet22(src, hyp, ref),
        chinese_rate=round(sum(bool(CJK.search(h)) for h in hyp) / n, 4),
        empty_rate=round(sum(not h for h in hyp) / n, 4),
        repeat_rate=round(sum(repeats(h) for h in hyp) / n, 4),
        **run_info,
        prompt=(cfg.get("target_prefix", "source language token") if cfg.get("seq2seq") else
                PROMPT if cfg.get("chat", True) else PLAIN_PROMPT),
        time=datetime.now().isoformat(timespec="seconds"),
    )
    with open(RES / "screening.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "prompt"}, ensure_ascii=False))


def rescore(a, cfg, out_dir):
    # Timing fields are carried over from the latest generation run of the same model and dev set.
    log = [json.loads(l) for l in open(RES / "screening.jsonl", encoding="utf-8")]
    keys = ["sent_per_s", "gen_seconds", "load_seconds", "peak_gpu_gb", "gpu_power_limit_w", "batch_size", "limit"]
    for dev in a.dev:
        recs = [json.loads(l) for l in open(out_dir / f"{dev}.raw.jsonl", encoding="utf-8")]
        src = [x["src"] for x in recs]
        ref = read(DEV / f"{dev}.en")[:len(src)]
        hyp = [clean(x["raw"], x["src"]) for x in recs]
        (out_dir / f"{dev}.hyp").write_text("\n".join(hyp) + "\n", encoding="utf-8")
        prev = [r for r in log if r["model"] == a.model and r["dev"] == dev and not r.get("rescored")][-1]
        log_row(a, cfg, dev, src, ref, hyp, {**{k: prev.get(k) for k in keys}, "rescored": True})


if __name__ == "__main__":
    main()
