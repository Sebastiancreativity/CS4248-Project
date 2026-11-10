"""Protect existing runs while changing decoding and handing over queued jobs."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eos_revision as eos
import finetune_pipeline as pipeline
import parallel_finetune_eval as parallel
from finetune_common import fingerprint, read_json, sha256, write_json
from model_registry import TEAM_MODELS


class DecodingTests(unittest.TestCase):
    def test_only_tower_eos_changes_for_both_variants(self):
        for base in (False, True):
            tok = SimpleNamespace(eos_token="<|im_end|>", eos_token_id=32000)
            config = SimpleNamespace(eos_token_id=2, use_cache=True, max_new_tokens=512, do_sample=False)
            model = SimpleNamespace(generation_config=config)
            loader = Mock(return_value=(tok, model))
            with contextlib.redirect_stdout(io.StringIO()):
                eos.corrected_load(loader, {"hf": eos.POLICY["hf"]}, Path("adapter"), base)
            self.assertEqual(vars(config), dict(eos_token_id=32000, use_cache=True, max_new_tokens=512, do_sample=False))
            loader.assert_called_once_with({"hf": eos.POLICY["hf"]}, Path("adapter"), base)
        config.eos_token_id = 2
        eos.corrected_load(loader, {"hf": "unrelated-model"})
        self.assertEqual(config.eos_token_id, 2)

    def test_wrong_tokenizer_is_rejected(self):
        loader = Mock(return_value=(SimpleNamespace(eos_token="</s>", eos_token_id=2), SimpleNamespace()))
        with self.assertRaisesRegex(ValueError, "Unexpected Tower"):
            eos.corrected_load(loader, {"hf": eos.POLICY["hf"]})

    def test_old_tower_candidates_are_invalidated_but_other_models_reuse_identity(self):
        with patch.object(eos, "original_identity", side_effect=lambda m, a, t, b: f"{m}-{t}-{b}"):
            for model in TEAM_MODELS:
                for base in (False, True):
                    old = f"{model}-512-{base}"
                    new = eos.revised_identity(model, None, 512, base)
                    self.assertEqual(old == new, model != eos.TARGET)
            self.assertNotEqual(eos.revised_identity(eos.TARGET, None, 512, True),
                                eos.revised_identity(eos.TARGET, None, 512, False))

    def test_subprocesses_reenter_revision_and_cannot_generate_from_ensemble(self):
        revision = (ROOT / "artifacts/example-revision").resolve()
        command = [sys.executable, ROOT / "scripts/run_ensembles.py", "--no-generate", "--owner", "Ming Zhe"]
        actual = eos.redirected_command(command, revision)
        self.assertIn("child", actual)
        self.assertEqual(actual[actual.index("--task") + 1], "ensemble")
        self.assertEqual(actual[actual.index("--") + 1:], ["--no-generate", "--owner", "Ming Zhe"])
        with self.assertRaises(ValueError):
            eos.redirected_command(command[:2], revision)
        with self.assertRaises(ValueError):
            eos.redirected_command([sys.executable, ROOT / "scripts/finetune_model.py"], revision)
        with self.assertRaises(ValueError):
            eos.validate_child_paths("screen", ["--output-dir", str(ROOT)], revision)
        eos.validate_child_paths("screen", ["--output-dir", str(revision / "candidates/model")], revision)

    def test_runtime_routes_children_and_restores_original_bindings(self):
        revision = (ROOT / "artifacts/example-revision").resolve()
        old = (pipeline.load_run, pipeline.generation_identity, pipeline.execute, parallel.completed_owner)
        commands = []
        with patch.object(eos, "checked_plan", return_value=({"fingerprint": "run"}, {})), \
                patch.object(parallel, "execute", side_effect=lambda cmd, lock_fd=None: commands.append((cmd, lock_fd))):
            with self.assertRaisesRegex(RuntimeError, "exercise cleanup"):
                with eos.runtime(revision):
                    self.assertIs(pipeline.generation_identity, eos.revised_identity)
                    self.assertEqual(pipeline.load_run(revision), {"fingerprint": "run"})
                    with self.assertRaises(ValueError):
                        pipeline.load_run(ROOT)
                    pipeline.execute([sys.executable, ROOT / "scripts/screen_model.py", "--model", eos.TARGET])
                    raise RuntimeError("exercise cleanup")
        self.assertEqual(old, (pipeline.load_run, pipeline.generation_identity, pipeline.execute, parallel.completed_owner))
        self.assertIn("--task", commands[0][0])

    def test_changed_dev_inputs_never_skip_old_completed_owner_scores(self):
        revision = (ROOT / "artifacts/example-revision").resolve()
        old_completed = Mock(return_value=True)
        assignment = {"combinations": [[eos.TARGET, TEAM_MODELS[0], TEAM_MODELS[1]]]}
        with patch.object(eos, "checked_plan", return_value=({}, {})), \
                patch.object(eos, "same_dev_inputs", return_value=False) as same, \
                patch.object(parallel, "completed_owner", old_completed):
            with eos.runtime(revision):
                self.assertFalse(parallel.completed_owner(Path("finetuned"), assignment, {"stage": "dev"}, {}))
                old_completed.assert_not_called()
                same.return_value = True
                self.assertTrue(parallel.completed_owner(Path("finetuned"), assignment, {"stage": "dev"}, {}))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.source, self.revision = self.root / "training", self.root / "revision"
        self.source.mkdir()
        self.split = self.root / "data"
        self.split.mkdir()
        self.src = ["source one", "source two"]
        for lang in ("zh", "en"):
            (self.split / f"dev_train20.{lang}").write_text("\n".join(self.src) + "\n", encoding="utf-8")
        spec = {"split_dir": str(self.split), "input_sha256": {}, "recipe_sha256": {}, "counts": {"dev_train20": 2},
                "evaluation": {"generation_batch_size": 4, "max_new_tokens": 512, "metric_batch_size": 16}}
        self.run = {"spec": spec, "fingerprint": fingerprint(spec)}
        write_json(self.source / "run.json", self.run)
        for model in TEAM_MODELS:
            adapter = self.source / "adapters" / model
            adapter.mkdir(parents=True)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (adapter / name).write_text("test-only-dummy", encoding="utf-8")
            write_json(adapter / "training_complete.json", {"model": model, "run_fingerprint": self.run["fingerprint"],
                       "global_step": 1, "base_revision": "revision", "base_model": "test", "quantized": True,
                       "files": {name: sha256(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors")}})
            for variant in pipeline.VARIANTS:
                self.candidates(self.source, variant, model, ["translation one", "translation two"], revised=False)
        probe = self.root / "probe"
        eos.link_adapters(self.source / "adapters", probe)
        if probe.is_symlink():
            probe.unlink()
        else:
            probe.rmdir()
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def candidates(self, folder, variant, model, translations, revised):
        output = folder / "candidates" / variant / model / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "dev_train20.raw.jsonl").write_text("\n".join(json.dumps({"src": s, "raw": h})
                        for s, h in zip(self.src, translations)) + "\n", encoding="utf-8")
        (output / "dev_train20.hyp").write_text("\n".join(translations) + "\n", encoding="utf-8")
        identity = eos.revised_identity if revised else eos.original_identity
        write_json(output / "dev_train20.generation.json", {"identity": identity(model, self.source / "adapters" / model,
                                                                                 512, variant == "pretrained")})

    def finish_parent(self):
        rows = [dict(variant=v, models=" + ".join(TEAM_MODELS[:3]), dev="dev_train20", selector=s,
                     owner="Shri", experiment=1, n=2, comet22=80, bleu=30, chrfpp=50)
                for v in pipeline.VARIANTS for s in pipeline.SELECTORS]
        output = self.source / "evaluation/dev"
        pipeline.save_csv(output / "scores_comparison.csv", rows)
        pipeline.save_csv(output / "individual_scores.csv", [{"model": "example", "comet22": 80}])
        write_json(output / "status.json", {"status": "complete"})
        ids = {m: eos.original_identity(m, self.source / "adapters" / m, 512) for m in TEAM_MODELS}
        frozen = pipeline.choose_on_dev(pipeline.read_csv(output / "scores_comparison.csv"))
        write_json(self.source / "selection.json", {**frozen, "adapters": ids,
                   "run_fingerprint": self.run["fingerprint"], "dev_scores_sha256": sha256(output / "scores_comparison.csv")})
        write_json(output / "finetuned/Shri/experiment_01/cache.json", {"example": "cached result"})

    def test_prepare_can_overlap_parent_scoring_and_never_relabels_old_tower_outputs(self):
        eos.prepare(self.source, self.revision)
        self.assertFalse((self.revision / "run.json").exists())
        self.assertFalse((self.revision / "selection.json").exists())
        self.assertEqual((self.revision / "adapters").resolve(), (self.source / "adapters").resolve())
        run, plan = eos.checked_plan(self.revision)
        self.assertEqual(run, self.run)
        for variant in pipeline.VARIANTS:
            for model in TEAM_MODELS:
                output = self.revision / "candidates" / variant / model / "output"
                old = read_json(output / "dev_train20.generation.json")["identity"]
                new = eos.revised_identity(model, self.revision / "adapters" / model, 512, variant == "pretrained")
                self.assertEqual(old == new, model != eos.TARGET)
        # Parent scorers may re-clean .hyp while validation runs; immutable raw
        # translations remain the authority, not the concurrently written .hyp.
        (self.source / "candidates/finetuned" / eos.TARGET / "output/dev_train20.hyp").write_text("")
        eos.checked_plan(self.revision)
        self.assertEqual(plan["training_run"], str(self.source))

    def test_metric_snapshot_waits_for_completed_dev_and_does_not_copy_selection(self):
        eos.prepare(self.source, self.revision)
        with self.assertRaises(FileNotFoundError):
            eos.seed_scores(self.revision)
        self.finish_parent()
        eos.seed_scores(self.revision)
        self.assertTrue((self.revision / "evaluation/dev/finetuned/Shri/experiment_01/cache.json").exists())
        self.assertFalse((self.revision / "selection.json").exists())
        self.assertFalse((self.revision / "evaluation/dev/scores_comparison.csv").exists())
        eos.checked_plan(self.revision)
        (self.source / "evaluation/dev/scores_comparison.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Original dev scores changed"):
            eos.checked_plan(self.revision)

    def test_full_dev_comparison_detects_one_changed_translation(self):
        eos.prepare(self.source, self.revision)
        for variant in pipeline.VARIANTS:
            self.candidates(self.revision, variant, eos.TARGET,
                            ["translation one", "changed" if variant == "finetuned" else "translation two"], revised=True)
        eos.dev_comparison(self.revision)
        rows = read_json(self.revision / "dev_eos_comparison.json")["comparisons"]
        self.assertEqual([r["changed_cleaned_translations"] for r in rows], [0, 1])
        self.assertTrue(eos.same_dev_inputs(self.revision, eos.checked_plan(self.revision)[1], self.run, "pretrained", [eos.TARGET]))
        self.assertFalse(eos.same_dev_inputs(self.revision, eos.checked_plan(self.revision)[1], self.run, "finetuned", [eos.TARGET]))

    def test_changed_adapter_and_source_are_rejected(self):
        eos.prepare(self.source, self.revision)
        weights = self.source / "adapters" / TEAM_MODELS[0] / "adapter_model.safetensors"
        original = weights.read_bytes()
        weights.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Changed or incomplete adapter"):
            eos.checked_plan(self.revision)
        weights.write_bytes(original)
        raw = self.source / "candidates/finetuned" / TEAM_MODELS[0] / "output/dev_train20.raw.jsonl"
        raw.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Source artifact changed"):
            eos.checked_plan(self.revision)


GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash required")
class SubmissionTests(unittest.TestCase):
    def launch(self, mode="normal"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            for name in ("eos_revision.py", "run_eos_revision.slurm"):
                (root / "scripts" / name).touch()
            run = root / "artifacts/finetune/original"
            run.mkdir(parents=True)
            (run / "run.json").write_text("{}")
            if mode.startswith("checkout"):
                (root / "eos_checkout.json").write_text("{}")
                (root / "scripts/run_eos_checkout.slurm").touch()
            script = r'''
mode="$2"
python3() {
    printf 'verify %s\n' "$*" >> calls
    [[ "$mode" != checkout_invalid ]]
}
scontrol() {
    printf 'control %s\n' "$*" >> calls
    if [[ "$1" == -o ]]; then
        [[ "$mode" != probe_failure || ! -f held ]] || return 1
        if [[ "$mode" == running || ( "$mode" == race && -f held ) ]]; then
            echo 'JobId=456 JobState=RUNNING Reason=None Dependency=(null)'
        else
            reason=Dependency
            [[ "$mode" != already_held ]] || reason=JobHeldUser
            echo "JobId=456 JobState=PENDING Reason=$reason Dependency=afterok:123(unfulfilled)"
        fi
    elif [[ "$1" == hold ]]; then
        touch held
    elif [[ "$1" == release && "$mode" == release_failure ]]; then
        return 1
    fi
    return 0
}
sbatch() {
    n=700
    [[ ! -f counter ]] || read -r n < counter
    n=$((n+1))
    echo "$n" > counter
    printf 'submit %s\n' "$*" >> calls
    [[ "$mode" != submit_failure || "$n" != 705 ]] || return 1
    echo "$n;test-cluster"
}
scancel() {
    printf 'cancel %s\n' "$*" >> calls
    [[ "$mode" != cancel_failure || "$1" != 456 ]]
}
source "$1" original revised 123 456
'''
            result = subprocess.run([BASH, "-c", script, "eos-submission-test",
                                     (ROOT / "scripts/submit_eos_revision.sh").as_posix(), mode],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            return result, (root / "calls").read_text().splitlines()

    def test_handover_keeps_dev_running_and_only_cancels_pending_test_after_queueing(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = [c for c in calls if c.startswith("submit")]
        self.assertEqual(len(submissions), 10)
        self.assertNotIn("--dependency", submissions[0])
        self.assertIn("--array=5,13%4", submissions[1])
        self.assertIn("--dependency=afterok:702:123", submissions[2])
        self.assertIn("--array=0-15%4", submissions[6])
        self.assertIn("--dependency=afterok:707", submissions[7])
        self.assertIn("priority test", submissions[7])
        self.assertTrue(all("--hold" in c for c in submissions))
        self.assertEqual([c for c in calls if c.startswith("cancel")], ["cancel 456"])
        self.assertLess(calls.index(submissions[-1]), calls.index("cancel 456"))
        self.assertTrue(calls[-1].startswith("control release 701 702"))
        self.assertFalse(any(c in ("cancel 123", "control hold 123") for c in calls))

    def test_failures_restore_old_test_and_cancel_only_new_held_jobs(self):
        for mode in ("probe_failure", "submit_failure", "cancel_failure"):
            with self.subTest(mode=mode):
                result, calls = self.launch(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("control release 456", calls)
                self.assertNotIn("cancel 123", calls)
                if mode == "submit_failure":
                    self.assertIn("cancel 701 702 703 704", calls)
                    self.assertNotIn("cancel 456", calls)

    def test_running_or_racing_test_never_gets_cancelled(self):
        for mode in ("running", "race"):
            with self.subTest(mode=mode):
                result, calls = self.launch(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c.startswith(("submit", "cancel")) for c in calls))

    def test_release_failure_preserves_replacement_and_reports_recovery(self):
        result, calls = self.launch("release_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Retry release: scontrol release 701 702", result.stderr)
        self.assertEqual([c for c in calls if c.startswith("cancel")], ["cancel 456"])

    def test_previously_user_held_test_can_be_replaced(self):
        result, calls = self.launch("already_held")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cancel 456", calls)

    def test_separate_checkout_uses_the_guarded_launcher_for_every_job(self):
        result, calls = self.launch("checkout")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(calls[0].startswith("verify -B scripts/prepare_eos_checkout.py --verify"))
        submissions = [c for c in calls if c.startswith("submit")]
        self.assertEqual(len(submissions), 10)
        self.assertTrue(all("scripts/run_eos_checkout.slurm" in c for c in submissions))

    def test_changed_separate_copy_stops_before_touching_jobs(self):
        result, calls = self.launch("checkout_invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("verify"))


if __name__ == "__main__":
    unittest.main()
