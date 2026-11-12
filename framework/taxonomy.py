"""Rule-based error labels for evaluation (omission, number and name errors); uses the reference."""
import re
from collections import Counter

HAN = re.compile(r"[一-鿿]")
ARAB = re.compile(r"\d+(?:[.,]\d+)*")
CJK_NUM = re.compile("[零〇一二三四五六七八九十百千万亿两]+")
PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
OMISSION = {"entity_missing", "number_dropped", "too_short"}


def numbers(s):
    return Counter(n.replace(",", "") for n in ARAB.findall(s))


def rep_ratio(s, n=4):
    w = s.split()
    if len(w) < 2 * n:
        return 0.0
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(g)) / len(g)


def length_band(src, ref):
    rs = sorted(len(r.split()) / len(HAN.findall(s)) for s, r in zip(src, ref) if HAN.findall(s))
    return rs[int(.01 * len(rs))], rs[int(.99 * len(rs))]


def classify(src, ref, hyp, band):
    if not hyp.strip():
        return {"empty"}
    tags = set()
    if HAN.search(hyp):
        tags.add("untranslated_han")
    if rep_ratio(hyp) > 0.30:
        tags.add("repetition")
    ns, nh = numbers(src), numbers(hyp)
    if ns - nh:
        tags.add("number_dropped")
    if (nh - ns) and not CJK_NUM.search(src):
        tags.add("number_added")
    hl = hyp.lower()
    for m in PROPER.findall(ref):
        toks = [t for t in m.lower().split() if len(t) > 2]
        if toks and sum(t in hl for t in toks) / len(toks) < 0.5:
            tags.add("entity_missing")
            break
    nhan = len(HAN.findall(src))
    if nhan:
        r = len(hyp.split()) / nhan
        if r < band[0]:
            tags.add("too_short")
        elif r > band[1]:
            tags.add("too_long")
    return tags or {"clean"}
