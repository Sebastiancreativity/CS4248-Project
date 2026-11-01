"""Content-addressed pair scores and atomic experiment checkpoints."""
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def cached_pair(path, candidates, peers, scorer):
    key = digest([scorer.get_scorer().hash, candidates, peers])
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('digest') == key:
            return record['scores']
    scores = scorer.pair_f1(candidates, peers)
    if len(scores) != len(candidates) or not all(math.isfinite(v) for v in scores):
        raise ValueError('Invalid pairwise BERTScore output')
    save(path, {'digest': key, 'scores': scores})
    return scores


def choose_bert(trio, outputs, pair_scores):
    """Mean outgoing F1 against the other two peers; ties keep input order."""
    choices = []
    for j in range(len(outputs[trio[0]])):
        scores = [sum(pair_scores[(a, b)][j] for b in trio if b != a) / 2
                  if outputs[a][j].strip() else float('-inf') for a in trio]
        choices.append(max(range(3), key=lambda k: scores[k]))
    return choices
