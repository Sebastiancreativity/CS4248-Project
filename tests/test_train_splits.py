import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_train_splits import key, split_rows


class SplitTests(unittest.TestCase):
    def test_ratios_alignment_disjoint_and_deterministic(self):
        src = [f'source {i}' for i in range(100)]
        ref = [f'reference {i}' for i in range(100)]
        splits, _ = split_rows(src, ref)
        self.assertEqual([len(r) for r in splits.values()], [70, 20, 10])
        self.assertEqual(splits, split_rows(src, ref)[0])
        self.assertNotEqual(splits, split_rows(src, ref, seed=2)[0])
        seen = set()
        for rows in splits.values():
            for s, r, line in rows:
                self.assertNotIn(key(s), seen)
                seen.add(key(s))
                self.assertEqual((s, r), (src[line-1], ref[line-1]))

    def test_normalized_duplicates_and_external_test_exclusion(self):
        src = [f'source {i}' for i in range(100)] + ['ＳＯＵＲＣＥ 0', '']
        ref = ['translation'] * len(src)
        splits, removed = split_rows(src, ref, excluded=['SOURCE 1'])
        self.assertEqual(removed, dict(empty=1, test_overlap=1, duplicate_source=1))
        self.assertEqual(sum(map(len, splits.values())), 99)
        self.assertNotIn(key('source 1'), {key(r[0]) for rows in splits.values() for r in rows})
        with self.assertRaises(ValueError):
            split_rows(['a'], [])

    def test_assignment_files_match_manifest(self):
        assignments = json.loads((ROOT / 'combinations_team_manifest.json').read_text())['assignments']
        all_trios = []
        for owner in ['Ming Zhe', 'Theodore', 'Shri', 'Reet']:
            actual = json.loads((ROOT / f"combinations_{owner.lower().replace(' ', '_')}.json").read_text())
            expected = next(a for a in assignments if a['owner'] == owner)
            self.assertEqual(actual, [e['models'] for e in expected['experiments']])
            self.assertEqual(len(actual), 10)
            all_trios.extend(tuple(sorted(t)) for t in actual)
        self.assertEqual(len(set(all_trios)), 40)
