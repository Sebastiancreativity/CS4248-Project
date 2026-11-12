"""chrF++ consensus: pairwise utility matrix and expected utility (MBR)."""
import numpy as np
from sacrebleu.metrics import CHRF

_chrf = CHRF(word_order=2)


def utility_matrix(texts):
    n = len(texts)
    U = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                U[a, b] = _chrf.sentence_score(texts[a], [texts[b]]).score / 100
    return U


def expected_utility(U, weights=None):
    n = len(U)
    if n == 1:
        return np.ones(1)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    W = np.tile(w, (n, 1))
    np.fill_diagonal(W, 0)
    return (U * W).sum(1) / W.sum(1).clip(1e-12)


def softmax(x, t):
    z = np.asarray(x, float) / t
    z = np.exp(z - z.max())
    return z / z.sum()
