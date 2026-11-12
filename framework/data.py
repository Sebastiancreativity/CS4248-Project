"""Loading of development sets and candidate files, and the seeded tuning/evaluation split."""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parents[1]
DEV_DIR = OFFICIAL / "data" / "dev"


@dataclass
class Candidate:
    system: str
    kind: str
    text: str


@dataclass
class Sentence:
    i: int
    src: str
    ref: str
    meta: dict
    cands: list = field(default_factory=list)


def read_lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def load_dev(name):
    src, ref = read_lines(DEV_DIR / f"{name}.zh"), read_lines(DEV_DIR / f"{name}.en")
    meta = [json.loads(l) for l in read_lines(DEV_DIR / f"{name}.meta.jsonl")]
    return [Sentence(i, s, r, m) for i, (s, r, m) in enumerate(zip(src, ref, meta))]


def split(sents, seed=4248, block=20):
    key = lambda s: s.meta.get("docid", f"block{s.i // block}")
    docs = sorted({key(s) for s in sents})
    random.Random(seed).shuffle(docs)
    tune_docs = set(docs[: len(docs) // 2])
    return [s.i for s in sents if key(s) in tune_docs], [s.i for s in sents if key(s) not in tune_docs]


def _read_candidate_file(path, system, n):
    path = Path(path)
    out = [[] for _ in range(n)]
    if path.suffix == ".jsonl":
        for line in read_lines(path):
            r = json.loads(line)
            if "cands" in r:
                out[r["i"]] += [Candidate(system, c.get("kind", "cand"), c["text"]) for c in r["cands"]]
            else:
                out[r["i"]].append(Candidate(system, r.get("kind", "1best"), r["text"]))
    else:
        lines = read_lines(path)
        assert len(lines) == n, f"{path}: {len(lines)} lines, dev set has {n}"
        out = [[Candidate(system, "1best", t)] for t in lines]
    return out


def load_pool(sents, dev, systems):
    for name, pattern in systems.items():
        path = Path(str(pattern).format(dev=dev))
        path = path if path.is_absolute() else OFFICIAL / path
        for s, cands in zip(sents, _read_candidate_file(path, name, len(sents))):
            s.cands += cands
    return sents
