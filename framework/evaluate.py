"""Corpus BLEU, chrF++ and COMET-22 scoring, omission share and paired bootstrap resampling."""
import numpy as np
import sacrebleu

from .cache import Cache, key
from .taxonomy import OMISSION, classify

_comet = None


def comet_scores(triples, cache=None, batch_size=32):
    global _comet
    cache = cache or Cache()
    keys = [key(*t) for t in triples]
    have = cache.get_many("comet22", set(keys))
    todo = {k: t for k, t in zip(keys, triples) if k not in have}
    if todo:
        if _comet is None:
            import torch
            from comet import download_model, load_from_checkpoint
            _comet = (load_from_checkpoint(download_model("Unbabel/wmt22-comet-da")), torch.cuda.is_available())
        model, gpu = _comet
        items = list(todo.items())
        pred = model.predict([{"src": s, "mt": m, "ref": r} for _, (s, m, r) in items], batch_size=batch_size,
                             gpus=1 if gpu else 0, progress_bar=False).scores
        done = {k: round(float(v), 5) for (k, _), v in zip(items, pred)}
        cache.put_many("comet22", done)
        have.update(done)
    return [have[k] for k in keys]


def evaluate(sents, ids, hyps, band):
    src, ref = [sents[i].src for i in ids], [sents[i].ref for i in ids]
    seg = comet_scores(list(zip(src, hyps, ref)))
    tags = [classify(s, r, h, band) for s, r, h in zip(src, ref, hyps)]
    rate = lambda t: round(sum(t in x for x in tags) / len(tags), 4)
    return dict(bleu=round(sacrebleu.corpus_bleu(hyps, [ref]).score, 2),
                chrfpp=round(sacrebleu.corpus_chrf(hyps, [ref], word_order=2).score, 2),
                comet22=round(100 * float(np.mean(seg)), 2),
                omission=round(sum(bool(x & OMISSION) for x in tags) / len(tags), 4),
                entity_missing=rate("entity_missing"), number_dropped=rate("number_dropped"),
                too_short=rate("too_short"), too_long=rate("too_long"), untranslated_han=rate("untranslated_han"),
                repetition=rate("repetition"), clean=rate("clean")), seg


def bootstrap(a, b, n=1000, seed=4248):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return round(float((d > 0).mean() + 0.5 * (d == 0).mean()), 4)
