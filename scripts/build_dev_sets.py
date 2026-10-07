"""Build dev sets for MBR candidate screening (see official/README.md).

Usage: python official/scripts/build_dev_sets.py
"""
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import opencc
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "official"
OUT = OFF / "data" / "dev"
SEED = 4248
WMT_PER_DOMAIN = 500
WMT_DOMAINS = ["news", "user_review"]

t2s = opencc.OpenCC("t2s")


def read(p):
    return [l.rstrip("\n") for l in open(p, encoding="utf-8")]


def key(zh):
    return re.sub(r"\s+", "", t2s.convert(zh))


def write_set(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.zh", "w", encoding="utf-8") as fz, \
         open(OUT / f"{name}.en", "w", encoding="utf-8") as fe, \
         open(OUT / f"{name}.meta.jsonl", "w", encoding="utf-8") as fm:
        for i, r in enumerate(rows):
            fz.write(r["zh"] + "\n")
            fe.write(r["en"] + "\n")
            meta = {k: v for k, v in r.items() if k not in ("zh", "en")}
            fm.write(json.dumps({"id": f"{name}-{i:05d}", **meta}, ensure_ascii=False) + "\n")


def describe(rows):
    lens = sorted(len(re.sub(r"\s", "", r["zh"])) for r in rows)
    return {"n": len(rows), "mean_chars": round(sum(lens) / len(lens), 1),
            "median_chars": lens[len(lens) // 2], "max_chars": lens[-1],
            "share_le15_chars": round(sum(l <= 15 for l in lens) / len(lens), 3)}


def sacrebleu_field(field):
    exe = Path(sys.executable).with_name("sacrebleu.exe")
    exe = str(exe) if exe.exists() else "sacrebleu"
    out = subprocess.run([exe, "-t", "wmt23", "-l", "zh-en", "--echo", field],
                         capture_output=True, encoding="utf-8", check=True).stdout
    return out.split("\n")[:-1]


def main():
    # Every dev sentence is checked against both official test sets and the training data.
    excl = {
        "tatoeba_test": {key(x) for x in read(OFF / "tatoeba.zh")},
        "wmt22_test": {key(x) for x in read(OFF / "wmttest2022.zh")},
        "train": {key(x) for x in read(OFF / "train.zh-en.zh")},
    }

    def blocked(zh):
        return [k for k, s in excl.items() if key(zh) in s]

    stats = {"seed": SEED, "removed": {}}

    # dev_wmt: WMT23 zh-en, WMT22-like domains only, sampled by whole documents.
    f = {k: sacrebleu_field(k) for k in ["src", "ref", "docid", "domain"]}
    assert len({len(v) for v in f.values()}) == 1, "WMT23 fields misaligned"
    docs, removed = defaultdict(list), Counter()
    for zh, en, doc, dom in zip(f["src"], f["ref"], f["docid"], f["domain"]):
        zh, en = zh.strip(), en.strip()
        if dom not in WMT_DOMAINS:
            removed[f"domain:{dom}"] += 1
        elif not zh or not en:
            removed["empty"] += 1
        elif blocked(zh):
            removed["overlap:" + "+".join(blocked(zh))] += 1
        else:
            docs[(dom, doc)].append({"zh": zh, "en": en, "source": "wmt23", "domain": dom, "docid": doc})
    rng = random.Random(SEED)
    wmt_rows = []
    for dom in WMT_DOMAINS:
        ids = sorted(d for d in docs if d[0] == dom)
        rng.shuffle(ids)
        taken = 0
        for d in ids:
            if taken >= WMT_PER_DOMAIN:
                break
            wmt_rows += docs[d]
            taken += len(docs[d])
    write_set("dev_wmt", wmt_rows)
    stats["dev_wmt"] = {**describe(wmt_rows), "domains": Counter(r["domain"] for r in wmt_rows),
                        "documents": len({r["docid"] for r in wmt_rows})}
    stats["removed"]["dev_wmt"] = removed

    # dev_short: Tatoeba cmn-eng, converted to simplified to match the official Tatoeba test.
    d = load_dataset("mteb/tatoeba-bitext-mining", "cmn-eng")["test"]
    removed, seen, short_rows = Counter(), set(), []
    for zh0, en in zip(d["sentence1"], d["sentence2"]):
        zh0, en = zh0.strip(), en.strip()
        zh = t2s.convert(zh0)
        if not zh or not en:
            removed["empty"] += 1
        elif blocked(zh):
            removed["overlap:" + "+".join(blocked(zh))] += 1
        elif key(zh) in seen:
            removed["duplicate"] += 1
        else:
            seen.add(key(zh))
            short_rows.append({"zh": zh, "en": en, "source": "tatoeba-cmn-eng(mteb)",
                               "converted_t2s": zh != zh0})
    write_set("dev_short", short_rows)
    stats["dev_short"] = {**describe(short_rows),
                          "converted_t2s": sum(r["converted_t2s"] for r in short_rows)}
    stats["removed"]["dev_short"] = removed

    # dev_alma: ALMA validation split, for LoRA early stopping only.
    a = load_dataset("haoranxu/ALMA-Human-Parallel", "zh-en")["validation"]
    removed, alma_rows = Counter(), []
    for ex in a["translation"]:
        zh, en = ex["zh"].strip(), ex["en"].strip()
        if blocked(zh):
            removed["overlap:" + "+".join(blocked(zh))] += 1
        else:
            alma_rows.append({"zh": zh, "en": en, "source": "alma-human-parallel/validation"})
    write_set("dev_alma", alma_rows)
    stats["dev_alma"] = describe(alma_rows)
    stats["removed"]["dev_alma"] = removed

    stats["test_tatoeba"] = describe([{"zh": x} for x in read(OFF / "tatoeba.zh")])
    stats["test_wmt22"] = describe([{"zh": x} for x in read(OFF / "wmttest2022.zh")])
    with open(OUT / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2, default=dict)
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
