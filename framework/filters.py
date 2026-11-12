"""Reference-free rule guard: hard tags exclude a candidate, soft tags add a penalty."""
from .text import HAN, META, hanzi_count, latin_strings, missing_numbers, rep_ratio

HARD = {"empty", "untranslated_han", "repetition", "meta_text"}
SOFT = {"number_missing", "latin_missing", "too_short", "too_long"}


def length_band(sents, ids):
    rs = sorted(len(sents[i].ref.split()) / hanzi_count(sents[i].src) for i in ids if hanzi_count(sents[i].src))
    return rs[int(.01 * len(rs))], rs[int(.99 * len(rs))]


def tag(src, hyp, band):
    if not hyp.strip():
        return {"empty"}, {}
    tags, detail = set(), {}
    if HAN.search(hyp):
        tags.add("untranslated_han")
    if rep_ratio(hyp) > 0.30:
        tags.add("repetition")
    if META.search(hyp):
        tags.add("meta_text")
    missing = missing_numbers(src, hyp)
    if missing:
        tags.add("number_missing")
        detail["numbers"] = missing
    lat = [x for x in latin_strings(src) if x.lower() not in hyp.lower()]
    if lat:
        tags.add("latin_missing")
        detail["latin"] = lat
    n = hanzi_count(src)
    if n >= 4:
        r = len(hyp.split()) / n
        if r < band[0]:
            tags.add("too_short")
        elif r > band[1]:
            tags.add("too_long")
    return tags, detail
