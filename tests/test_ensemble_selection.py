import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ensemble_cache import cached_pair, choose_bert
from bertscore_metric import EnglishBERTScore


class SelectionTests(unittest.TestCase):
    def test_average_peer_score_and_empty_ties(self):
        trio = ['a', 'b', 'c']
        outputs = {'a': ['a', '', 'a'], 'b': ['b', '', 'b'], 'c': ['c', '', '']}
        pairs = {(a,b): [0.1, 0.0, 0.5] for a in trio for b in trio if a != b}
        pairs['b','a'][0] = .8
        pairs['b','c'][0] = .9
        self.assertEqual(choose_bert(trio, outputs, pairs), [1, 0, 0])

    def test_pair_cache_invalidates_text_and_encoder(self):
        scorer = Mock()
        scorer.get_scorer.return_value.hash = 'model1'
        scorer.pair_f1.return_value = [.5]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pair.json'
            cached_pair(path, ['a'], ['b'], scorer)
            cached_pair(path, ['a'], ['b'], scorer)
            self.assertEqual(scorer.pair_f1.call_count, 1)
            cached_pair(path, ['changed'], ['b'], scorer)
            scorer.get_scorer.return_value.hash = 'model2'
            cached_pair(path, ['changed'], ['b'], scorer)
            self.assertEqual(scorer.pair_f1.call_count, 3)

    def test_empty_peers_score_zero_without_dropping_alignment(self):
        scorer = EnglishBERTScore()
        backend = Mock()
        backend.score.return_value = (None, None, SimpleNamespace(tolist=lambda: [.8]))
        scorer._scorer = backend
        self.assertEqual(scorer.pair_f1(['a', '', 'c'], ['b', 'b', '']), [.8, 0., 0.])
        backend.score.assert_called_once_with(['a'], ['b'], batch_size=16)
