"""Rebuilds the ensemble's 70/20/10 split (seed 4248) from the revised training corpus: writes dev_train20 and the internal test set test_train10.

Usage (from official/): python scripts/1_data/rebuild_split.py
"""
import json
import random
import unicodedata
from pathlib import Path

OFF = Path(__file__).resolve().parents[2]
DELETED = set(range(953, 978)) | {11513, 11514}
MERGED = [(2595, 2596), (2619, 2621), (2626, 2627), (2628, 2629), (4050, 4051), (4265, 4266)]


def read(p):
    return Path(p).read_text(encoding="utf-8-sig").splitlines()


def key(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def revise(zh, en):
    first = dict(MERGED)
    inner = {i for a, b in MERGED for i in range(a + 1, b + 1)}
    out_zh, out_en = [], []
    for i in range(1, len(zh) + 1):
        if i in DELETED or i in inner:
            continue
        span = range(i, first.get(i, i) + 1)
        out_zh.append("".join(zh[k - 1].strip() for k in span))
        out_en.append(" ".join(en[k - 1].strip() for k in span))
    return out_zh, out_en


def split(zh, en, excluded, seed=4248):
    blocked, seen, rows = {key(s) for s in excluded}, set(), []
    for i, (s, r) in enumerate(zip(zh, en), 1):
        s, r = s.strip(), r.strip()
        k = key(s)
        if k and r and k not in blocked and k not in seen:
            seen.add(k)
            rows.append((s, r, i))
    random.Random(seed).shuffle(rows)
    n_train, n_dev = len(rows) * 70 // 100, len(rows) * 20 // 100
    return {"dev": rows[n_train:n_train + n_dev], "test": rows[n_train + n_dev:]}


def write(rows, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.zh").write_text("".join(r[0] + "\n" for r in rows), encoding="utf-8", newline="\n")
    (folder / f"{name}.en").write_text("".join(r[1] + "\n" for r in rows), encoding="utf-8", newline="\n")
    (folder / f"{name}.meta.jsonl").write_text(
        "".join(json.dumps({"id": f"{name}-{k:05d}", "line": r[2]}) + "\n" for k, r in enumerate(rows)),
        encoding="utf-8", newline="\n")


def main():
    zh, en = revise(read(OFF / "data/train/train.zh-en.zh"), read(OFF / "data/train/train.zh-en.en"))
    excluded = read(OFF / "data/test/tatoeba.zh") + read(OFF / "data/test/wmttest2022.zh")
    parts = split(zh, en, excluded)
    write(parts["dev"], OFF / "data" / "dev", "dev_train20")
    write(parts["test"], OFF / "data" / "test", "test_train10")
    print(json.dumps({"revised_pairs": len(zh), "dev_train20": len(parts["dev"]), "test_train10": len(parts["test"])}))


if __name__ == "__main__":
    main()
