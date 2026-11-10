"""Exercise cache reuse, isolated workers, collection and Slurm dependencies."""
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
import parallel_finetune_eval as parallel
from ensemble_cache import digest
from finetune_common import read_json, write_json
from model_registry import TEAM_MODELS


class ParallelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.split = self.root / "split"
        self.split.mkdir()
        for name in ("train", "dev_train20", "test_train10"):
            for lang in ("zh", "en"):
                (self.split / f"{name}.{lang}").write_text(f"{name}-{lang}\n", encoding="utf-8")
        for name in ("tatoeba.zh", "tatoeba.en", "wmttest2022.zh", "wmttest2022.AnnA.en"):
            (self.root / name).write_text(f"{name}\n", encoding="utf-8")
        self.run = {"fingerprint": "run", "spec": {"split_dir": str(self.split),
                    "counts": {"dev_train20": 1}, "evaluation": {
                        "max_new_tokens": 512, "generation_batch_size": 4, "metric_batch_size": 16}}}
        write_json(self.root / "run.json", self.run)
        self.assignments = parallel.pipeline.read_assignments(ROOT, None)
        self.commands = []
        self.stack.enter_context(patch.multiple(parallel, ROOT=self.root, load_run=lambda _: self.run,
            verified_adapters=lambda *args: {m: "fine-" + m for m in TEAM_MODELS},
            generation_identity=lambda m, a, t, base=False: ("base-" if base else "fine-") + m,
            metric_fingerprint=lambda _: "metrics", execute=self.execute))
        self.stack.enter_context(patch.object(parallel.pipeline, "read_assignments", return_value=self.assignments))
        self.stack.enter_context(patch.dict(sys.modules, {"run_ensembles": SimpleNamespace(valid_cache=self.valid_cache)}))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    @staticmethod
    def valid_cache(path, name, src, identity):
        try:
            records = [json.loads(line) for line in (path / f"{name}.raw.jsonl").read_text().splitlines()]
            return (read_json(path / f"{name}.generation.json")["identity"] == identity
                    and [r["src"] for r in records] == src
                    and len((path / f"{name}.hyp").read_text().splitlines()) == len(src))
        except (OSError, ValueError, KeyError):
            return False

    def execute(self, command, lock_fd=None):
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
            output, model = Path(arg("--output-dir")), arg("--model")
            output.mkdir(parents=True, exist_ok=True)
            for name in many("--dev"):
                src, _ = parallel.read_parallel(arg("--data-dir"), name)
                (output / f"{name}.raw.jsonl").write_text("\n".join(json.dumps({"src": s, "raw": "output"}) for s in src) + "\n")
                (output / f"{name}.hyp").write_text("output\n" * len(src))
                write_json(output / f"{name}.generation.json", {"identity": ("base-" if "--base-only" in command else "fine-") + model})
        elif script == "score_finetuned_models.py":
            rows = [dict(variant=arg("--variant"), model=m, dataset=d, n=1,
                         comet22=85, bleu=30, chrfpp=50, bertscore_f1=80)
                    for m in TEAM_MODELS for d in many("--datasets")]
            parallel.pipeline.save_csv(self.root / "evaluation" / arg("--stage") / arg("--variant") / "individual/scores.csv", rows)
        elif script == "run_ensembles.py":
            assignment = next(a for a in self.assignments if a["owner"] == arg("--owner"))
            output = Path(arg("--results-dir")) / assignment["owner"]
            rows = []
            for selector in parallel.pipeline.SELECTORS:
                for i, trio in enumerate(assignment["combinations"], 1):
                    for name in many("--datasets"):
                        src, ref = parallel.read_parallel(arg("--data-dir"), name)
                        hyp, counts = ["output"], {m: int(k == 0) for k, m in enumerate(trio)}
                        exp = output / f"experiment_{i:02d}"
                        exp.mkdir(parents=True, exist_ok=True)
                        hyp_path = exp / f"{name}_{selector}.hyp"
                        hyp_path.write_text("output\n")
                        row = dict(experiment=i, dev=name, models=" + ".join(trio), selector=selector, n=1,
                                   comet22=90 if "--base-only" in command else 89, bleu=30, chrfpp=50,
                                   bertscore_f1=80, selection_counts=json.dumps(counts), hyp_file=str(hyp_path))
                        write_json(exp / f"{name}_{selector}.checkpoint.json",
                                   {"digest": digest(["metrics", i, trio, name, selector, src, ref, hyp, counts]), "row": row})
                        rows.append(row)
            parallel.pipeline.save_csv(output / "scores_comparison.csv", rows)
        else:
            raise AssertionError(command)

    def generate_all(self, stage):
        for index in range(16):
            parallel.generate(self.root, stage, index)

    def score_all(self, stage):
        for index in range(10):
            parallel.score_worker(self.root, stage, index)

    def test_full_dev_then_test_preserves_frozen_choice_and_exact_coverage(self):
        with self.assertRaises(FileNotFoundError):
            parallel.initialize(self.root, "test")
        self.assertEqual(self.commands, [])
        parallel.initialize(self.root, "dev")
        self.generate_all("dev")
        self.score_all("dev")
        parallel.collect(self.root, "dev")
        dev = parallel.pipeline.read_csv(self.root / "evaluation/dev/scores_comparison.csv")
        self.assertEqual(len(dev), 240)
        choice = (self.root / "selection.json").read_bytes()
        self.assertEqual(read_json(self.root / "selection.json")["primary"]["variant"], "pretrained")
        self.assertEqual(len(parallel.pipeline.read_csv(self.root / "evaluation/dev/individual_scores.csv")), 16)
        parallel.initialize(self.root, "test")
        self.generate_all("test")
        self.score_all("test")
        parallel.collect(self.root, "test")
        self.assertEqual(len(parallel.pipeline.read_csv(self.root / "evaluation/test/scores_comparison.csv")), 720)
        self.assertEqual(len(parallel.pipeline.read_csv(self.root / "evaluation/test/individual_scores.csv")), 48)
        selected = parallel.pipeline.read_csv(self.root / "evaluation/test/final_selected_scores.csv")
        self.assertEqual({row["dev"] for row in selected}, {"test_train10", "test_wmt22", "test_tatoeba"})
        self.assertEqual(len(selected), 3)
        self.assertEqual((self.root / "selection.json").read_bytes(), choice)

    def test_cached_generation_and_finished_owner_are_not_recomputed(self):
        parallel.initialize(self.root, "dev")
        self.generate_all("dev")
        before = len(self.commands)
        self.generate_all("dev")
        self.assertEqual(len(self.commands), before)
        parallel.score_worker(self.root, "dev", 0)
        before = len(self.commands)
        parallel.score_worker(self.root, "dev", 0)
        self.assertEqual(len(self.commands), before)
        # Wrong generation provenance must cause generation rather than reuse.
        path = self.root / "candidates/pretrained" / TEAM_MODELS[0] / "output/dev_train20.generation.json"
        write_json(path, {"identity": "wrong-adapter"})
        parallel.generate(self.root, "dev", 0)
        self.assertEqual(len(self.commands), before + 1)

    def test_workers_isolate_candidates_and_caches_and_recover_partial_json(self):
        parallel.initialize(self.root, "dev")
        self.generate_all("dev")
        self.score_all("dev")
        commands = [c for c in self.commands if Path(c[1]).name == "run_ensembles.py"]
        candidate_dirs = [Path(c[c.index("--model-outputs-dir") + 1]) for c in commands]
        results_dirs = [Path(c[c.index("--results-dir") + 1]) for c in commands]
        self.assertEqual(len(set(candidate_dirs)), 8)
        self.assertEqual(len(set(results_dirs)), 8)
        copied = candidate_dirs[0] / "hunyuan-mt-7b/output/dev_train20.hyp"
        copied.write_text("worker-local change\n")
        canonical = self.root / "candidates/pretrained/hunyuan-mt-7b/output/dev_train20.hyp"
        self.assertEqual(canonical.read_text(), "output\n")
        # Force a retry of this owner and simulate a QE cache interrupted mid-write.
        original = self.root / "evaluation/dev/pretrained" / self.assignments[0]["owner"] / "scores_comparison.csv"
        original.unlink()
        damaged = results_dirs[0] / ".cache/qe_cache/example/dev_train20.json"
        damaged.parent.mkdir(parents=True, exist_ok=True)
        damaged.write_text('{"scores": [')
        parallel.score_worker(self.root, "dev", 0)
        self.assertFalse(damaged.exists())
        self.assertEqual(len(list(damaged.parent.glob("*.incomplete-*"))), 1)

    def test_incomplete_or_changed_results_cannot_freeze_selection(self):
        parallel.initialize(self.root, "dev")
        self.generate_all("dev")
        with self.assertRaises(FileNotFoundError):
            parallel.collect(self.root, "dev")
        self.assertFalse((self.root / "selection.json").exists())
        self.score_all("dev")
        marker = read_json(parallel.workspace(self.root, "dev") / "done/score-0.json")
        output = Path(marker["output"])
        output.write_text(output.read_text() + "tampered")
        with self.assertRaises(ValueError):
            parallel.collect(self.root, "dev")
        self.assertFalse((self.root / "selection.json").exists())

    def test_modified_prepared_test_or_dev_data_is_rejected(self):
        parallel.initialize(self.root, "dev")
        (self.split / "dev_train20.en").write_text("changed reference\n")
        with self.assertRaises(ValueError):
            parallel.generate(self.root, "dev", 0)
        self.assertEqual(self.commands, [])


class OriginalRecipeCompatibilityTests(unittest.TestCase):
    def test_original_runner_checkpoints_are_reusable_without_rescoring(self):
        fake_bleu = SimpleNamespace(corpus_bleu=lambda *a, **k: SimpleNamespace(score=20),
                                    corpus_chrf=lambda *a, **k: SimpleNamespace(score=40),
                                    sentence_chrf=lambda *a, **k: SimpleNamespace(score=50))
        spec = importlib.util.spec_from_file_location("original_parallel_cache_test", ROOT / "scripts/run_ensembles.py")
        runner = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"cache_paths": SimpleNamespace(COMET_CACHE=Path(".")), "sacrebleu": fake_bleu}):
            spec.loader.exec_module(runner)
        scorer = Mock()
        scorer.get_scorer.return_value.hash = "test"
        scorer.pair_f1.return_value = [0.7]
        scorer.score.return_value = {"bertscore_f1": 70}
        trio = list(TEAM_MODELS[:3])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "dev_train20.zh").write_text("source\n")
            (data / "dev_train20.en").write_text("gold\n")
            combos = root / "combos.json"
            combos.write_text(json.dumps([trio]))
            for model in trio:
                output = root / "candidates" / model / "output"
                output.mkdir(parents=True)
                (output / "dev_train20.hyp").write_text("output\n")
                (output / "dev_train20.raw.jsonl").write_text(json.dumps({"src": "source", "raw": "output"}) + "\n")
            argv = ["runner", "--combinations", str(combos), "--owner", "Shri", "--data-dir", str(data),
                    "--datasets", "dev_train20", "--model-outputs-dir", str(root / "candidates"),
                    "--results-dir", str(root / "results"), "--selector", "all", "--comet", "--bertscore",
                    "--bertscore-device", "cuda", "--bertscore-batch-size", "16", "--no-generate"]
            comet_model = SimpleNamespace(predict=lambda *a, **k: SimpleNamespace(system_score=0.9))
            with patch.multiple(runner, version=lambda _: "test", bertscore_from_args=lambda _: scorer,
                                qe_scores=lambda *a: [0.8]), patch.object(sys, "argv", argv), patch.dict(sys.modules, {
                                    "screen_model": SimpleNamespace(MODELS=dict.fromkeys(trio)),
                                    "comet": SimpleNamespace(download_model=lambda *a, **k: "fake",
                                                             load_from_checkpoint=lambda *a: comet_model),
                                    "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
                                }), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            assignment = {"owner": "Shri", "combinations": [trio]}
            plan = {"datasets": {"dev_train20": 1}, "data_dir": str(data)}
            with patch.object(parallel, "version", return_value="test"):
                self.assertTrue(parallel.completed_owner(root / "results", assignment, plan, {"metric_batch_size": 16}))
            with patch.object(parallel, "version", return_value="different-version"):
                self.assertFalse(parallel.completed_owner(root / "results", assignment, plan, {"metric_batch_size": 16}))


GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash required for submission tests")
class ParallelSubmissionTests(unittest.TestCase):
    def launch(self, *, fail_at=0, chosen=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/run_parallel_finetune_eval.slurm").touch()
            run = root / "artifacts/finetune/example"
            run.mkdir(parents=True)
            (run / "run.json").write_text("{}")
            if chosen:
                (run / "selection.json").write_text("{}")
            script = r'''
sbatch() {
    local n=0
    [[ ! -f count ]] || read -r n < count
    n=$((n + 1))
    printf '%s\n' "$n" > count
    printf 'submit %s\n' "$*" >> calls
    [[ "$n" -ne "$2_FAIL" ]] || return 1
    printf '%s;test-cluster\n' "$((100 + n))"
}
scontrol() { printf 'control %s\n' "$*" >> calls; }
scancel() { printf 'cancel %s\n' "$*" >> calls; }
export CS4248_EVAL_PARALLEL=4
source "$1" example
'''.replace('"$2_FAIL"', str(fail_at))
            result = subprocess.run([BASH, "-c", script, "submission-test",
                                     (ROOT / "scripts/submit_parallel_finetune_eval.sh").as_posix()],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            calls = (root / "calls").read_text().splitlines()
            return result, calls

    def test_arrays_have_throttle_and_test_waits_for_dev_collection(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        submits = [c for c in calls if c.startswith("submit")]
        self.assertEqual(len(submits), 8)
        self.assertTrue(all("--hold" in c and "--kill-on-invalid-dep=yes" in c for c in submits))
        self.assertIn("--array=0-15%4", submits[1])
        self.assertIn("--array=0-9%4", submits[2])
        self.assertIn("--dependency=afterok:104", submits[4])
        self.assertTrue(all(c.startswith("submit") for c in calls[:8]))
        self.assertEqual(len([c for c in calls if c.startswith("control release")]), 8)

    def test_submission_failure_cancels_only_its_held_jobs(self):
        result, calls = self.launch(fail_at=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cancel 101 102", calls)
        self.assertFalse(any(c.startswith("control release") for c in calls))

    def test_existing_dev_selection_starts_at_test(self):
        result, calls = self.launch(chosen=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        submits = [c for c in calls if c.startswith("submit")]
        self.assertEqual(len(submits), 4)
        self.assertTrue(all(" test example" in c for c in submits))
