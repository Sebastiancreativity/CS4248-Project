"""Exercise metric aggregation and validation without downloading encoder weights."""
import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bertscore_metric import EnglishBERTScore, add_arguments, from_args


def values(items):
    return SimpleNamespace(mean=lambda: SimpleNamespace(item=lambda: sum(items) / len(items)))


class BERTScoreTests(unittest.TestCase):
    def test_sentence_f1_is_averaged_and_empty_hypotheses_count(self):
        encoder = Mock(hash="test-settings")
        # Opposing P/R values distinguish mean sentence F1 from F1 of mean P/R.
        encoder.score.return_value = tuple(values(v) for v in
                                           ([1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [2/3, 2/3, 0.0]))
        scorer = EnglishBERTScore(batch_size=8)
        scorer._scorer = encoder
        hyp = ["one", "two", ""]
        ref = ["first", "second", "third"]
        result = scorer.score(hyp, ref)
        self.assertEqual(result["bertscore_precision"], 50.0)
        self.assertEqual(result["bertscore_recall"], 50.0)
        self.assertEqual(result["bertscore_f1"], 44.44)
        self.assertEqual(result["bertscore_hash"], "test-settings")
        encoder.score.assert_called_once_with(hyp, ref, batch_size=8)

    def test_invalid_inputs_fail_before_model_loading(self):
        scorer = EnglishBERTScore()
        for hyp, ref in [([], []), (["a"], []), (["a"], [" "])]:
            with self.subTest(hyp=hyp, ref=ref), self.assertRaises(ValueError):
                scorer.score(hyp, ref)
        self.assertIsNone(scorer._scorer)

    def test_batch_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            EnglishBERTScore(batch_size=0)

    def test_encoder_is_loaded_once_and_settings_are_fixed(self):
        encoder = Mock(hash="test-settings")
        encoder.score.return_value = tuple(values([1.0]) for _ in range(3))
        factory = Mock(return_value=encoder)
        with patch.dict(sys.modules, {"cache_paths": Mock(), "bert_score": SimpleNamespace(BERTScorer=factory)}):
            scorer = EnglishBERTScore()
            scorer.score(["hello"], ["hello"])
            scorer.score(["world"], ["world"])
        self.assertEqual(factory.call_count, 1)
        self.assertFalse(factory.call_args.kwargs["idf"])
        self.assertFalse(factory.call_args.kwargs["rescale_with_baseline"])
        self.assertEqual(factory.call_args.kwargs["lang"], "en")

    def test_scoring_is_opt_in(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.assertIsNone(from_args(parser.parse_args([])))
        scorer = from_args(parser.parse_args(["--bertscore", "--bertscore-device", "cuda"]))
        self.assertEqual(scorer.device, "cuda")
        self.assertIsNone(scorer._scorer)


if __name__ == "__main__":
    unittest.main()
