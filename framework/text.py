"""Text utilities shared by the guard: Chinese, number and Latin-string detection and normalisation."""
import re
import unicodedata
from collections import Counter

HAN = re.compile(r"[一-鿿]")
ARAB = re.compile(r"\d+(?:[.,]\d+)*")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+\-.]*[A-Za-z0-9]|[A-Za-z]{2,}")
PREFIX = re.compile(r"^((here\s*(is|'s)\s*)?(the\s*|a\s*)?(corrected\s*)?(and\s*complete\s*|,\s*complete\s*)?"
                    r"(english\s*)?(translation|english)|英文|译文)\s*[:：]\s*", re.I)
META = re.compile(r"^(here\s*(is|'s)|translation\s*[:：]|note\s*[:：]|\(note)", re.I)


def clean_output(raw, src):
    lines = [PREFIX.sub("", l.strip()) for l in raw.strip().split("\n")]
    lines = [l for l in lines if l]
    out = lines[0] if lines else ""
    if len(out) > 1 and out[0] == out[-1] and out[0] in "\"'“”" and src[:1] not in "\"“'‘「":
        out = out[1:-1].strip()
    return out


WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen twenty").split()
ORDINALS = ("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
            "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()
TENS = {30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety", 100: "hundred"}


def numbers(s):
    return Counter(n.replace(",", "") for n in ARAB.findall(unicodedata.normalize("NFKC", s)))


def source_numbers(src):
    src = unicodedata.normalize("NFKC", src)
    keep = Counter()
    for m in ARAB.finditer(src):
        nxt = src[m.end():].lstrip()[:1]
        if nxt in ("月", "万", "亿") or (m.start() == 0 and nxt in "、.)．"):
            continue
        keep[m.group().replace(",", "")] += 1
    return keep


def missing_numbers(src, hyp):
    words = set(re.findall(r"[a-z]+", hyp.lower()))
    out = []
    for n in (source_numbers(src) - numbers(hyp)).elements():
        v = int(n) if n.isdigit() else None
        spelled = v is not None and ((v <= 20 and (WORDS[v] in words or ORDINALS[v] in words)) or TENS.get(v) in words)
        if not spelled:
            out.append(n)
    return out


def latin_strings(src):
    return [m for m in LATIN.findall(src) if not m.isdigit()]


def hanzi_count(s):
    return len(HAN.findall(s))


def rep_ratio(s, n=4):
    w = s.split()
    if len(w) < 2 * n:
        return 0.0
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(g)) / len(g)
