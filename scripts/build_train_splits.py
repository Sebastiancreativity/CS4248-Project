"""Create reproducible, source-disjoint 70/20/10 splits from parallel training files."""
import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def key(text):
    return ''.join(unicodedata.normalize('NFKC', text).casefold().split())


def split_rows(sources, references, excluded=(), seed=4248):
    if len(sources) != len(references):
        raise ValueError('Source/reference line counts differ')
    blocked = {key(s) for s in excluded}
    seen, rows = set(), []
    removed = dict(empty=0, test_overlap=0, duplicate_source=0)
    for i, (src, ref) in enumerate(zip(sources, references), 1):
        src, ref = src.strip(), ref.strip()
        k = key(src)
        if not k or not ref:
            removed['empty'] += 1
        elif k in blocked:
            removed['test_overlap'] += 1
        elif k in seen:
            removed['duplicate_source'] += 1
        else:
            seen.add(k)
            rows.append((src, ref, i))
    if len(rows) < 10:
        raise ValueError('Need at least 10 eligible unique source sentences')
    random.Random(seed).shuffle(rows)
    ntrain, ndev = len(rows) * 70 // 100, len(rows) * 20 // 100
    return {'train': rows[:ntrain], 'dev_train20': rows[ntrain:ntrain+ndev],
            'test_train10': rows[ntrain+ndev:]}, removed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'train.zh-en.zh')
    p.add_argument('--reference', type=Path, default=ROOT / 'train.zh-en.en')
    p.add_argument('--exclude-source', nargs='+', type=Path,
                   default=[ROOT / 'tatoeba.zh', ROOT / 'wmttest2022.zh'])
    p.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'splits_70_20_10')
    p.add_argument('--seed', type=int, default=4248)
    a = p.parse_args()
    read = lambda path: path.read_text(encoding='utf-8-sig').splitlines()
    excluded = [line for path in a.exclude_source for line in read(path)]
    splits, removed = split_rows(read(a.source), read(a.reference), excluded, a.seed)
    files = {}
    for name, rows in splits.items():
        for col, lang in [(0, 'zh'), (1, 'en')]:
            files[f'{name}.{lang}'] = ''.join(r[col] + '\n' for r in rows)
        files[f'{name}.meta.jsonl'] = ''.join(json.dumps({'original_line': r[2]}) + '\n' for r in rows)
    manifest = {'seed': a.seed, 'ratios': [70, 20, 10],
                'counts': {n: len(r) for n, r in splits.items()}, 'removed': removed,
                'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [a.source, a.reference, *a.exclude_source]},
                'normalization': 'NFKC, casefold, remove whitespace; keep first source occurrence',
                'rounding': 'floor train/dev counts; test receives remainder'}
    files['manifest.json'] = json.dumps(manifest, indent=2) + '\n'
    # Never silently replace an established split used by an earlier run.
    for name, content in files.items():
        path = a.output_dir / name
        if path.exists() and path.read_text(encoding='utf-8') != content:
            p.error(f'{path} differs; choose a new --output-dir for a new split')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (a.output_dir / name).write_text(content, encoding='utf-8', newline='\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
