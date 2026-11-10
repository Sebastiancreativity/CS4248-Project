"""Keep final test choices frozen and insert report jobs without losing work."""
import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import priority_test as priority
from finetune_common import read_json, sha256, write_json
from model_registry import TEAM_MODELS

REAL_SCORE_INDIVIDUALS = priority.score_individuals


class PriorityTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.split = self.root / "split"
        self.split.mkdir()
        for name in ("dev_train20", "test_train10"):
            for lang in ("zh", "en"):
                (self.split / f"{name}.{lang}").write_text(f"{name}-{lang}\n")
        for name in ("tatoeba.zh", "tatoeba.en", "wmttest2022.zh", "wmttest2022.AnnA.en"):
            (self.root / name).write_text(f"{name}\n")
        (self.root / "scripts").mkdir()
        for name in ("score_finetuned_models.py", "run_ensembles.py"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.run = {"fingerprint": "run", "spec": {"split_dir": str(self.split), "counts": {"dev_train20": 1},
                    "tests": {"test_train10": 1, "test_wmt22": 1, "test_tatoeba": 1},
                    "evaluation": {"max_new_tokens": 512, "generation_batch_size": 4, "metric_batch_size": 16}}}
        self.baseline = "nllb-1.3b"  # Outside the first assigned ensemble.
        self.individuals = [dict(variant=v, model=m, dataset="dev_train20", n=1,
                                comet22=99 if (v == "finetuned" and m == self.baseline) else 80,
                                bleu=30, chrfpp=50, bertscore_f1=80)
                            for v in priority.pipeline.VARIANTS for m in TEAM_MODELS]
        priority.pipeline.save_csv(self.root / "evaluation/dev/individual_scores.csv", self.individuals)
        assignments = priority.pipeline.read_assignments(ROOT, None)
        rows = [dict(variant=v, owner=a["owner"], experiment=i, models=" + ".join(trio), selector=s,
                     dev="dev_train20", n=1, comet22=90 if v == "finetuned" else 85,
                     bleu=30, chrfpp=50, bertscore_f1=80)
                for v in priority.pipeline.VARIANTS for a in assignments
                for i, trio in enumerate(a["combinations"], 1) for s in priority.pipeline.SELECTORS]
        scores = self.root / "evaluation/dev/scores_comparison.csv"
        priority.pipeline.save_csv(scores, rows)
        choice = {**priority.pipeline.choose_on_dev(priority.pipeline.read_csv(scores)), "run_fingerprint": "run",
                  "adapters": {m: "fine-" + m for m in TEAM_MODELS}, "dev_scores_sha256": sha256(scores)}
        write_json(self.root / "selection.json", choice)
        self.commands = []
        self.stack.enter_context(patch.multiple(priority, ROOT=self.root, load_run=lambda _: self.run,
            adapter_record=lambda *a: {"run_fingerprint": "run"}, generation_identity=self.identity,
            score_individuals=self.fake_individuals))
        self.stack.enter_context(patch.object(priority.pipeline, "execute", self.execute))
        self.stack.enter_context(patch.dict(sys.modules, {"run_ensembles": SimpleNamespace(valid_cache=self.valid_cache)}))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    @staticmethod
    def identity(model, adapter, tokens, base=False):
        return ("base-" if base else "fine-") + model

    @staticmethod
    def valid_cache(path, name, src, identity):
        try:
            return (read_json(path / f"{name}.generation.json")["identity"] == identity
                    and [json.loads(line)["src"] for line in (path / f"{name}.raw.jsonl").read_text().splitlines()] == src
                    and len((path / f"{name}.hyp").read_text().splitlines()) == len(src))
        except (OSError, ValueError, KeyError):
            return False

    def execute(self, command):
        command = list(map(str, command))
        self.commands.append(command)
        def arg(flag):
            return command[command.index(flag) + 1]
        def many(flag):
            start = command.index(flag) + 1
            end = next((i for i in range(start, len(command)) if command[i].startswith("--")), len(command))
            return command[start:end]
        script = Path(command[1]).name
        if script == "screen_model.py":
            out = Path(arg("--output-dir"))
            out.mkdir(parents=True, exist_ok=True)
            for name in many("--dev"):
                src, _ = priority.read_parallel(arg("--data-dir"), name)
                (out / f"{name}.raw.jsonl").write_text("\n".join(json.dumps({"src": s, "raw": "output"}) for s in src) + "\n")
                (out / f"{name}.hyp").write_text("output\n" * len(src))
                write_json(out / f"{name}.generation.json", {"identity": self.identity(arg("--model"), None, 512, "--base-only" in command)})
        elif script == "run_ensembles.py":
            trio = read_json(arg("--combinations"))[0]
            rows = [dict(experiment=1, models=" + ".join(trio), selector=arg("--selector"), dev=d, n=1,
                         comet22=95 if "--base-only" in command else 70, bleu=30, chrfpp=50, bertscore_f1=80)
                    for d in many("--datasets")]
            priority.pipeline.save_csv(Path(arg("--results-dir")) / arg("--owner") / f"scores_{arg('--selector')}.csv", rows)
        else:
            raise AssertionError(f"Unexpected command: {command}")

    def fake_individuals(self, run_dir, run, plan, data_dir, datasets):
        return [dict(variant=v, model=self.baseline, dataset=d, n=1, comet22=90 if v == "pretrained" else 60,
                     bleu=30, chrfpp=50, bertscore_f1=80) for v in priority.pipeline.VARIANTS for d in datasets]

    def test_report_publishes_twelve_matched_results_without_reselecting_on_test(self):
        before = (self.root / "selection.json").read_bytes()
        priority.evaluate_locked(self.root)
        output = self.root / "evaluation/priority_test"
        rows = priority.pipeline.read_csv(output / "report_scores.csv")
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(priority.pipeline.read_csv(output / "report_deltas.csv")), 6)
        primary = priority.pipeline.read_csv(output / "final_selected_scores.csv")
        self.assertEqual(len(primary), 3)
        self.assertEqual({r["variant"] for r in primary}, {"finetuned"})
        self.assertEqual((self.root / "selection.json").read_bytes(), before)
        self.assertEqual(read_json(output / "status.json")["status"], "complete")
        generations = [c for c in self.commands if Path(c[1]).name == "screen_model.py"]
        self.assertEqual(len(generations), 8)
        self.assertEqual(len({c[c.index('--model') + 1] for c in generations}), 4)
        count = len(self.commands)
        priority.evaluate_locked(self.root)
        self.assertEqual(len(self.commands), count)
        self.assertFalse((self.root / "evaluation/test/scores_comparison.csv").exists())

    def test_test_work_cannot_start_before_dev_selection(self):
        (self.root / "selection.json").unlink()
        with self.assertRaises(FileNotFoundError):
            priority.evaluate_locked(self.root)
        self.assertEqual(self.commands, [])

    def test_missing_baseline_rows_do_not_publish_complete_report(self):
        with patch.object(priority, "score_individuals", return_value=[]):
            with self.assertRaises(ValueError):
                priority.evaluate_locked(self.root)
        self.assertFalse((self.root / "evaluation/priority_test/report_scores.csv").exists())

    def test_baseline_choice_rejects_partial_or_test_data(self):
        self.assertEqual(priority.select_baseline(self.individuals, 1)["model"], self.baseline)
        with self.assertRaises(ValueError):
            priority.select_baseline(self.individuals[:-1], 1)
        changed = [{**r, "dataset": "test_train10"} for r in self.individuals]
        with self.assertRaises(ValueError):
            priority.select_baseline(changed, 1)

    def test_original_full_runner_reuses_priority_baseline_checkpoints(self):
        run, plan, output, data_dir, datasets = priority.prepare(self.root)
        priority.generate_needed(self.root, run, plan, data_dir, datasets)
        scorer = Mock(return_value={"comet22": 90, "bleu": 30, "chrfpp": 50, "bertscore_f1": 80})
        modules = {"run_ensembles": SimpleNamespace(valid_cache=self.valid_cache, score=scorer),
                   "cache_paths": SimpleNamespace(COMET_CACHE=self.root),
                   "comet": SimpleNamespace(download_model=lambda *a, **k: "fake", load_from_checkpoint=lambda *a: object())}
        with patch.dict(sys.modules, modules), patch.object(priority, "version", return_value="test"):
            rows = REAL_SCORE_INDIVIDUALS(self.root, run, plan, data_dir, datasets)
            self.assertEqual(len(rows), 6)
            self.assertEqual(scorer.call_count, 6)
            spec = importlib.util.spec_from_file_location("priority_baseline_compatibility", ROOT / "scripts/score_finetuned_models.py")
            original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original)
            with patch.multiple(original, load_run=lambda _: self.run, TEAM_MODELS=(self.baseline,),
                                generation_identity=self.identity, adapter_record=lambda *a: {"run_fingerprint": "run"},
                                version=lambda _: "test"):
                for variant in priority.pipeline.VARIANTS:
                    with patch.object(sys, "argv", ["score", "--run-dir", str(self.root), "--stage", "test",
                                                   "--variant", variant, "--data-dir", str(data_dir), "--datasets", *datasets]):
                        original.main()
            self.assertEqual(scorer.call_count, 6)


GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash required for scheduling tests")
class PrioritySubmissionTests(unittest.TestCase):
    def launch(self, mode="normal"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/priority_test.py").touch()
            run = root / "artifacts/finetune/example"
            run.mkdir(parents=True)
            (run / "run.json").write_text("{}")
            script = r'''
mode="$2"
scontrol() {
    printf 'control %s\n' "$*" >> calls
    if [[ "$1" == -o ]]; then
        [[ "$mode" != probe_failure || ! -f held ]] || return 1
        if [[ "$mode" == running || ( "$mode" == race && -f held ) ]]; then
            echo 'JobId=456 JobState=RUNNING Reason=None Dependency=(null)'
        else
            echo 'JobId=456 JobState=PENDING Reason=Dependency Dependency=afterok:123(unfulfilled)'
        fi
    elif [[ "$1" == hold ]]; then
        touch held
    elif [[ "$1" == update && "$*" == *afterok:777 && "$mode" == update_failure ]]; then
        return 1
    elif [[ "$1" == release && "$mode" == release_failure ]]; then
        return 1
    fi
    return 0
}
sbatch() {
    printf 'submit %s\n' "$*" >> calls
    [[ "$mode" != submit_failure ]] || return 1
    echo '777;test-cluster'
}
scancel() { printf 'cancel %s\n' "$*" >> calls; }
source "$1" example 123 456
'''
            result = subprocess.run([BASH, "-c", script, "priority-submission-test",
                                     (ROOT / "scripts/submit_priority_test.sh").as_posix(), mode],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            return result, (root / "calls").read_text().splitlines()

    def test_pending_test_is_rewired_without_cancelling_running_dev(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("control hold 456", calls)
        self.assertIn("control update JobId=456 Dependency=afterok:777", calls)
        self.assertIn("control release 777 456", calls)
        self.assertTrue(any("--dependency=afterok:123" in c and "--hold" in c for c in calls if c.startswith("submit")))
        self.assertFalse(any(c.startswith("cancel") or "hold 123" in c for c in calls))

    def test_failure_restores_original_dependency_and_cancels_only_new_job(self):
        for mode in ("probe_failure", "submit_failure", "update_failure"):
            with self.subTest(mode=mode):
                result, calls = self.launch(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("control update JobId=456 Dependency=afterok:123", calls)
                self.assertIn("control release 456", calls)
                cancellations = [c for c in calls if c.startswith("cancel")]
                self.assertEqual(cancellations, ["cancel 777"] if mode == "update_failure" else [])

    def test_running_test_or_startup_race_never_submits_another_evaluation(self):
        for mode in ("running", "race"):
            with self.subTest(mode=mode):
                result, calls = self.launch(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c.startswith("submit") or c.startswith("cancel") for c in calls))
                if mode == "race":
                    self.assertIn("control release 456", calls)

    def test_release_failure_preserves_the_new_chain_and_prints_recovery(self):
        result, calls = self.launch("release_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scontrol release 777 456", result.stderr)
        self.assertNotIn("control update JobId=456 Dependency=afterok:123", calls)
        self.assertFalse(any(c.startswith("cancel") for c in calls))
