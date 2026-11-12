"""Per-candidate features (consensus, guard tags, auditor scores) and per-system calibration."""
import numpy as np

from . import filters
from .mbr import expected_utility, utility_matrix


def featurise(sents, ids, scorers, feature_map, band, F=None):
    F = F if F is not None else {}
    pairs = [(sents[i].src, c.text) for i in ids for c in sents[i].cands]
    feat_vals = {f: scorers[s].score(pairs) for f, s in feature_map.items()}
    pos = 0
    for i in ids:
        texts = [c.text for c in sents[i].cands]
        tags = [filters.tag(sents[i].src, t, band)[0] for t in texts]
        U = utility_matrix(texts)
        F[i] = dict(U=U, mbr=expected_utility(U), hard=np.array([bool(t & filters.HARD) for t in tags]),
                    rule=-np.array([len(t & filters.SOFT) for t in tags], float))
        for f, vals in feat_vals.items():
            F[i][f] = np.asarray(vals[pos:pos + len(texts)], float)
        F[i]["sys"] = np.array([c.system for c in sents[i].cands])
        pos += len(texts)
    return F


def calibrate(F, tune_ids, spec):
    for new, base in (spec or {}).items():
        vals = {}
        for i in tune_ids:
            for s, v in zip(F[i]["sys"], F[i][base]):
                vals.setdefault(s, []).append(v)
        mean = {s: float(np.mean(v)) for s, v in vals.items()}
        overall = float(np.mean([v for vs in vals.values() for v in vs]))
        for i in F:
            F[i][new] = F[i][base] - np.array([mean.get(s, overall) for s in F[i]["sys"]])
    return F
