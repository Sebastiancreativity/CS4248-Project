"""Linear selection over candidate features with the guard, and grid-search tuning of the weights by corpus chrF++."""
import itertools

import numpy as np
import sacrebleu

from .mbr import expected_utility, softmax


def scores(F, w, T=None, aux=()):
    feats = dict(F)
    if "wmbr" in w:
        feats["wmbr"] = expected_utility(F["U"], softmax(sum(F[a] for a in aux), T))
    return sum(v * feats[f] for f, v in w.items() if v)


def choose(F, w, guard=True, T=None, aux=()):
    s = scores(F, w, T, aux)
    if guard and (~F["hard"]).any():
        s = np.where(F["hard"], -np.inf, s)
    return int(np.argmax(s))


def tune(feats, texts, refs, ids, spec):
    grid = spec["grid"]
    names = list(grid)
    Ts = spec.get("wmbr", {}).get("T", [None])
    aux = spec.get("wmbr", {}).get("aux", ())
    best, table = None, []
    for T in Ts:
        for combo in itertools.product(*(grid[n] for n in names)):
            w = dict(zip(names, combo))
            hyp = [texts[i][choose(feats[i], w, spec.get("guard", True), T, aux)] for i in ids]
            score = sacrebleu.corpus_chrf(hyp, [[refs[i] for i in ids]], word_order=2).score
            table.append(dict(w=w, T=T, chrfpp=round(score, 3)))
            if best is None or score > best[0]:
                best = (score, w, T)
    return dict(weights=best[1], T=best[2], aux=list(aux), guard=spec.get("guard", True),
                tune_chrfpp=round(best[0], 3), grid=table)
