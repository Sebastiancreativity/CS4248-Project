"""Exercise both selectors, output isolation, cleaning and restart without models."""
import contextlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class RunnerTests(unittest.TestCase):
    def test_agreement_suite_reuses_completed_scores(self):
        fake_bleu = SimpleNamespace(corpus_bleu=lambda *a, **k: SimpleNamespace(score=20),
                                    corpus_chrf=lambda *a, **k: SimpleNamespace(score=40),
                                    sentence_chrf=lambda *a, **k: SimpleNamespace(score=50))
        with patch.dict(sys.modules, {'cache_paths': SimpleNamespace(COMET_CACHE=Path('.')),
                                     'sacrebleu': fake_bleu,
                                     'screen_model': SimpleNamespace(MODELS=dict.fromkeys(['a','b','c']))}):
            sys.modules.pop('run_ensembles', None)
            runner = importlib.import_module('run_ensembles')
        scorer = Mock()
        scorer.get_scorer.return_value.hash = 'test'
        scorer.pair_f1.return_value = [.7]
        scorer.score.return_value = {'bertscore_f1': 70}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / 'dev'; dev.mkdir()
            (dev / 'dev_wmt.zh').write_text('source\n')
            (dev / 'dev_wmt.en').write_text('gold\n')
            combos = root / 'combos.json'; combos.write_text(json.dumps([['a','b','c']]))
            for model in ['a','b','c']:
                output = root / 'outputs' / model / 'output'; output.mkdir(parents=True)
                (output / 'dev_wmt.hyp').write_text('stale\n')
                (output / 'dev_wmt.raw.jsonl').write_text(json.dumps({'src':'source','raw':'Translation in English:\nHello.'})+'\n')
            argv = ['runner', '--combinations', str(combos), '--owner', 'P Shricharan',
                    '--dev', 'dev_wmt', '--selector', 'agreement', '--bertscore', '--run-name', 'test', '--no-generate']
            with patch.multiple(runner, DEV=dev, MODEL_OUTPUTS=root/'outputs', SCREEN=root/'screen',
                                OUT_BASE=root/'results', version=lambda _: 'test', bertscore_from_args=lambda _: scorer), patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'screen_model': SimpleNamespace(MODELS=dict.fromkeys(['a','b','c']))}), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                self.assertEqual(scorer.score.call_count, 2)
                runner.main()
                self.assertEqual(scorer.score.call_count, 2)
            out = root / 'results' / 'P Shricharan' / 'test'
            self.assertTrue((out / 'scores_comparison.csv').exists())
            self.assertEqual((root/'outputs/a/output/dev_wmt.hyp').read_text(), 'Hello.\n')
            self.assertEqual(len(list((out/'experiment_01').glob('*.checkpoint.json'))), 2)
