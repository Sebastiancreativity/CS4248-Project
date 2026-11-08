"""Test official input mapping, suite routing and complete-result publication."""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_test_suite as suite


class TestSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split = self.root / "split"
        self.split.mkdir()
        fixtures = {
            "wmttest2022.zh": "wmt source\n", "wmttest2022.AnnA.en": "annotator A reference\n",
            "tatoeba.zh": "short source 1\nshort source 2\n",
            "tatoeba.en": "short reference 1\nshort reference 2\n",
            "split/test_train10.zh": "internal source\n",
            "split/test_train10.en": "internal reference\n",
        }
        for name, text in fixtures.items():
            (self.root / name).write_text(text, encoding="utf-8")
        manifest = json.loads((REPO / "combinations_team_manifest.json").read_text())
        (self.root / "combinations_team_manifest.json").write_text(json.dumps(manifest))
        for assignment in manifest["assignments"]:
            (self.root / assignment["file"]).write_text((REPO / assignment["file"]).read_text())

    def test_test_data_mapping_and_validation(self):
        datasets = suite.read_test_data("official", self.root, self.split)
        self.assertEqual(set(datasets), {"test_wmt22", "test_tatoeba"})
        self.assertEqual(datasets["test_wmt22"]["reference"], "annotator A reference\n")
        self.assertEqual(datasets["test_tatoeba"]["n"], 2)
        self.assertEqual(set(suite.read_test_data("internal", self.root, self.split)), {"test_train10"})
        self.assertEqual(len(suite.read_test_data("both", self.root, self.split)), 3)
        (self.root / "tatoeba.en").write_text("one line\n")
        with self.assertRaisesRegex(ValueError, "line-count mismatch"):
            suite.read_test_data("official", self.root, self.split)
        (self.root / "tatoeba.en").write_text("one line\n\n")
        with self.assertRaisesRegex(ValueError, "blank"):
            suite.read_test_data("official", self.root, self.split)

    def invoke(self, extra=(), run=None):
        argv = ["suite", "--official-dir", str(self.root), "--split-dir", str(self.split), *extra]
        with patch.object(suite, "ROOT", self.root), patch.object(sys, "argv", argv), \
                patch.object(suite.subprocess, "run", side_effect=run) as child, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = suite.main()
        return result, child

    def fake_evaluation(self, command, cwd):
        def argument(name):
            return command[command.index(name) + 1]
        self.assertEqual(argument("--selector"), "all")
        self.assertIn("--comet", command)
        self.assertIn("--bertscore", command)
        datasets = command[command.index("--datasets") + 1:command.index("--bertscore-device")]
        self.assertEqual(set(datasets), {"test_wmt22", "test_tatoeba"})
        data_dir = Path(argument("--data-dir"))
        combinations = json.loads(Path(argument("--combinations")).read_text())
        rows = []
        for i, models in enumerate(combinations, 1):
            for dataset in datasets:
                source = (data_dir / f"{dataset}.zh").read_text().splitlines()
                for selector in sorted(suite.SELECTORS):
                    rows.append({"experiment": i, "dev": dataset, "selector": selector,
                                 "models": " + ".join(models), "n": len(source),
                                 "bleu": 25, "chrfpp": 50, "comet22": 80, "bertscore_f1": 80})
        path = self.root / "results" / "ensembles" / argument("--owner") / argument("--run-name") / "scores_comparison.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return SimpleNamespace(returncode=0)

    def test_dry_run_does_not_generate_or_write_results(self):
        result, child = self.invoke(["--dry-run"])
        self.assertEqual(result, 0)
        child.assert_not_called()
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / "results").exists())

    def test_all_members_publish_240_test_rows_and_manifest(self):
        result, child = self.invoke(run=self.fake_evaluation)
        self.assertEqual(result, 0)
        self.assertEqual(child.call_count, 4)
        summary = self.root / "results/ensembles/test_suites/official"
        manifest = json.loads((summary / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed_rows"], 240)
        with (summary / "scores_comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 240)
        self.assertEqual({r["owner"] for r in rows}, {"Ming Zhe", "Theodore", "Reet", "Shri"})
        self.assertEqual({r["dev"] for r in rows}, {"test_wmt22", "test_tatoeba"})

    def test_failure_does_not_publish_stale_combined_results(self):
        summary = self.root / "results/ensembles/test_suites/official"
        summary.mkdir(parents=True)
        (summary / "scores_comparison.csv").write_text("old results\n")
        result, child = self.invoke(run=lambda *a, **k: SimpleNamespace(returncode=1))
        self.assertEqual(result, 1)
        self.assertEqual(child.call_count, 4)
        self.assertFalse((summary / "scores_comparison.csv").exists())
        self.assertEqual(json.loads((summary / "manifest.json").read_text())["status"], "failed")

    def test_collection_rejects_missing_or_wrong_dataset_rows(self):
        self.invoke(["--owners", "Shri"], run=self.fake_evaluation)
        datasets = suite.read_test_data("official", self.root, self.split)
        assignments = suite.read_assignments(self.root, ["Shri"])
        path = self.root / "results/ensembles/Shri/test_official_all/scores_comparison.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            suite.collect_results(self.root, assignments, datasets, "test_official_all")
        path.write_text("\n".join(lines).replace("test_wmt22", "dev_wmt") + "\n")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            suite.collect_results(self.root, assignments, datasets, "test_official_all")
