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
        self.run_suite("agreement", 2)

    def test_all_selectors_reuse_completed_scores(self):
        self.run_suite("all", 3)

    def test_official_test_datasets_reuse_scores_without_picking_winner(self):
        self.run_suite("all", 6, ("test_wmt22", "test_tatoeba"))

    def run_suite(self, selector, expected, datasets=("dev_train20",)):
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
            for dataset in datasets:
                (dev / f'{dataset}.zh').write_text('source\n')
                (dev / f'{dataset}.en').write_text('gold\n')
            combos = root / 'combos.json'; combos.write_text(json.dumps([['a','b','c']]))
            for model in ['a','b','c']:
                output = root / 'outputs' / model / 'output'; output.mkdir(parents=True)
                for dataset in datasets:
                    (output / f'{dataset}.hyp').write_text('stale\n')
                    (output / f'{dataset}.raw.jsonl').write_text(json.dumps({'src':'source','raw':'Translation in English:\nHello.'})+'\n')
            argv = ['runner', '--combinations', str(combos), '--owner', 'P Shricharan',
                    '--data-dir', str(dev), '--dev', *datasets, '--selector', selector, '--bertscore', '--run-name', 'test', '--no-generate']
            with patch.multiple(runner, DEV=dev, MODEL_OUTPUTS=root/'outputs', SCREEN=root/'screen',
                                OUT_BASE=root/'results', version=lambda _: 'test', bertscore_from_args=lambda _: scorer, qe_scores=lambda *a: [0.8]), patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'screen_model': SimpleNamespace(MODELS=dict.fromkeys(['a','b','c'])), 'comet': SimpleNamespace(download_model=lambda *a, **k: 'fake', load_from_checkpoint=lambda *a: object()), 'torch': SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))}), contextlib.redirect_stdout(io.StringIO()) as captured:
                runner.main()
                self.assertEqual(scorer.score.call_count, expected)
                runner.main()
                self.assertEqual(scorer.score.call_count, expected)
                if all(dataset.startswith('test_') for dataset in datasets):
                    self.assertNotIn('Best ', captured.getvalue())
                    for dataset in datasets:
                        (dev / f'{dataset}.en').write_text('updated gold\n')
                    runner.main()
                    self.assertEqual(scorer.score.call_count, expected * 2)
            out = root / 'results' / 'P Shricharan' / 'test'
            self.assertTrue((out / 'scores_comparison.csv').exists())
            for dataset in datasets:
                self.assertEqual((root / f'outputs/a/output/{dataset}.hyp').read_text(), 'Hello.\n')
            self.assertEqual(len(list((out/'experiment_01').glob('*.checkpoint.json'))), expected)

            if selector == 'all':
                import csv
                with (out / 'scores_comparison.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual({r['selector'] for r in rows}, {'cometkiwi', 'chrf', 'bertscore'})
                self.assertEqual(len(rows), expected)
                self.assertEqual({r['dev'] for r in rows}, set(datasets))
