import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eos_pool as pool
from finetune_common import read_json, write_json


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.plan = {"stages": pool.stages(), "fingerprint": "plan", "dev_job": "123",
                     "source": str(self.root / "original"), "revision": str(self.root / "revision")}
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def key(self, task):
        phase = next(i for i, tasks in enumerate(self.plan["stages"]) if task in tasks)
        return pool.task_key(phase, task)

    def test_one_worker_can_complete_every_task_without_other_workers_starting(self):
        seen = []
        pool.work(self.root, self.plan, "only-worker", runner=lambda p, t, fd: seen.append(self.key(t)),
                  ready=lambda p: True, pause=0.001)
        expected = [pool.task_key(i, task) for i, tasks in enumerate(pool.stages()) for task in tasks]
        self.assertEqual(seen, expected)
        self.assertEqual(len(seen), 44)
        self.assertTrue(all(pool.done(self.root, self.plan, key) for key in expected))

    def test_four_workers_execute_each_task_once_and_respect_all_stage_barriers(self):
        started, completed = set(), set()
        mutex = threading.Lock()

        def run(plan, task, descriptors):
            key = self.key(task)
            phase = int(key[:2])
            with mutex:
                self.assertNotIn(key, started)
                for previous_phase in range(phase):
                    for previous in plan["stages"][previous_phase]:
                        self.assertIn(pool.task_key(previous_phase, previous), completed)
                started.add(key)
            time.sleep(0.003)
            with mutex:
                completed.add(key)

        with ThreadPoolExecutor(max_workers=4) as workers:
            pending = [workers.submit(pool.work, self.root, self.plan, f"worker-{i}", run, lambda p: True, 0.001)
                       for i in range(4)]
            for result in pending:
                result.result(timeout=20)
        self.assertEqual(len(completed), 44)

    def test_failed_task_blocks_test_and_retry_reuses_completed_work(self):
        calls = []
        fail = True

        def run(plan, task, descriptors):
            nonlocal fail
            calls.append(self.key(task))
            if task == {"action": "score", "stage": "dev", "index": 0} and fail:
                fail = False
                raise RuntimeError("simulated metric failure")

        with self.assertRaisesRegex(RuntimeError, "simulated metric failure"):
            pool.work(self.root, self.plan, "first", run, lambda p: True, 0.001)
        self.assertTrue((self.root / "error.json").exists())
        self.assertFalse(any("-test-" in key for key in calls))
        completed = set(calls[:-1])
        pool.retry(self.root)
        pool.work(self.root, self.plan, "retry", run, lambda p: True, 0.001)
        self.assertTrue(all(calls.count(key) == 1 for key in completed))
        self.assertEqual(len(calls), 45)

    def test_retry_refuses_to_clear_failure_while_a_worker_is_active(self):
        write_json(self.root / "error.json", {"error": "failed"})
        with pool.lock(self.root / "workers/alive.lock"):
            with self.assertRaisesRegex(RuntimeError, "still active"):
                pool.retry(self.root)
        self.assertTrue((self.root / "error.json").exists())

    def test_original_dev_completion_is_required_before_seed_and_test(self):
        calls = []
        checks = 0

        def ready(plan):
            nonlocal checks
            checks += 1
            self.assertEqual(len(calls), 3)  # prepare + two full dev generations
            return checks >= 3

        pool.work(self.root, self.plan, "one", lambda p, t, fd: calls.append(self.key(t)), ready, 0.001)
        self.assertEqual(checks, 3)
        self.assertEqual(len(calls), 44)

    def test_source_job_failure_is_reported_instead_of_waiting_for_hours(self):
        result = subprocess.CompletedProcess([], 0, "", "")
        failed = subprocess.CompletedProcess([], 0, "123|TIMEOUT|\n", "")
        with patch.object(pool.subprocess, "run", side_effect=[result, failed]):
            with self.assertRaisesRegex(RuntimeError, "TIMEOUT"):
                pool.original_dev_ready(self.plan)
        write_json(Path(self.plan["source"]) / "evaluation/dev/status.json", {"status": "complete"})
        with patch.object(pool.subprocess, "run") as scheduler:
            self.assertTrue(pool.original_dev_ready(self.plan))
            scheduler.assert_not_called()

    def test_child_commands_use_existing_corrected_driver_and_never_retrain(self):
        commands = [pool.task_command(self.plan, task) for tasks in pool.stages() for task in tasks]
        self.assertTrue(all(Path(cmd[2]).name == "eos_revision.py" for cmd in commands))
        self.assertFalse(any("train" in cmd for cmd in commands))
        self.assertEqual(commands[0][-2:], ["--source-run", self.plan["source"]])
        self.assertEqual(sum("--index" in cmd for cmd in commands), 38)

    def test_pool_initialization_is_repeatable_but_changed_code_is_rejected(self):
        source, revision = self.root / "source", self.root / "revision"
        write_json(source / "run.json", {"training": "fixed"})
        code = self.root / "scripts"
        code.mkdir()
        for name in pool.POOL_CODE:
            (code / name).write_text(name)
        with patch.object(pool, "ROOT", self.root):
            plan = pool.initialize(source, revision, "123")
            self.assertEqual(pool.initialize(source, revision, "123"), plan)
            self.assertEqual(pool.load_plan(revision), plan)
            self.assertFalse(revision.exists())  # EOS prepare needs a fresh directory.
            (code / "eos_revision.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Pool code changed"):
                pool.load_plan(revision)

    def test_targeted_git_restore_preserves_existing_checkout_edits_and_head(self):
        from install_eos_additions import git
        repo = self.root / "deployment"
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "core.autocrlf", "false")
        protected = repo / "active_recipe.py"
        protected.write_text("tracked recipe")
        git(repo, "add", ".")
        commit_args = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false")
        git(repo, *commit_args, "commit", "-qm", "original")
        original = git(repo, "rev-parse", "HEAD").decode().strip()
        paths = ["scripts/eos_pool.py", "scripts/run_eos_pool.slurm", "scripts/submit_eos_pool.sh"]
        (repo / "scripts").mkdir()
        for name in paths:
            (repo / name).write_text("new helper")
        git(repo, "add", "scripts")
        git(repo, *commit_args, "commit", "-qm", "pool")
        updated = git(repo, "rev-parse", "HEAD").decode().strip()
        git(repo, "checkout", "--quiet", "--detach", original)
        protected.write_text("active local recipe edits")
        git(repo, "restore", "--source", updated, "--", *paths)
        self.assertEqual(protected.read_text(), "active local recipe edits")
        self.assertEqual(git(repo, "rev-parse", "HEAD").decode().strip(), original)
        self.assertTrue(all((repo / name).read_text() == "new helper" for name in paths))


GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash required")
class SubmissionTests(unittest.TestCase):
    def launch(self, mode="normal"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/run_eos_pool.slurm").touch()
            source = root / "artifacts/finetune/original"
            source.mkdir(parents=True)
            (source / "run.json").write_text("{}")
            script = r'''
mode="$2"
python3() {
    printf 'init %s\n' "$*" >> calls
    mkdir -p artifacts/finetune/.eos-pool-revised
}
flock() { return 0; }
scontrol() {
    printf 'control %s\n' "$*" >> calls
    if [[ "$1" == -o ]]; then
        if [[ "$mode" == running || ( "$mode" == race && -f held ) ]]; then
            echo 'JobId=456 JobState=RUNNING Reason=None Dependency=(null)'
        else
            echo 'JobId=456 JobState=PENDING Reason=JobHeldUser Dependency=afterok:123(unfulfilled)'
        fi
    elif [[ "$1" == hold ]]; then
        touch held
    elif [[ "$1" == release && "$mode" == release_failure ]]; then
        return 1
    fi
    return 0
}
sbatch() {
    printf 'submit %s\n' "$*" >> calls
    if [[ "$mode" == quota_none || ( "$mode" == quota_two && ( "$*" == *0-3%4* || "$*" == *0-2%3* ) ) ]]; then
        echo 'sbatch: error: AssocMaxSubmitJobLimit' >&2
        return 1
    fi
    if [[ "$mode" == other_error ]]; then echo 'sbatch: error: Invalid partition' >&2; return 1; fi
    echo '777;cluster'
}
scancel() {
    printf 'cancel %s\n' "$*" >> calls
    [[ "$mode" != cancel_failure || "$1" != 456 ]]
}
source "$1" original revised 123 456
'''
            result = subprocess.run([BASH, "-c", script, "pool-submission-test",
                                     (ROOT / "scripts/submit_eos_pool.sh").as_posix(), mode],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            return result, (root / "calls").read_text().splitlines()

    def test_only_one_four_worker_array_is_submitted_and_dev_is_untouched(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = [c for c in calls if c.startswith("submit")]
        self.assertEqual(len(submissions), 1)
        self.assertIn("--hold", submissions[0])
        self.assertIn("--array=0-3%4", submissions[0])
        self.assertEqual([c for c in calls if c.startswith("cancel")], ["cancel 456"])
        self.assertIn("control release 777", calls)
        self.assertFalse(any(c in ("control hold 123", "cancel 123") for c in calls))

    def test_submit_limit_falls_back_to_two_workers_without_queuing_failed_attempts(self):
        result, calls = self.launch("quota_two")
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = [c for c in calls if c.startswith("submit")]
        self.assertEqual(len(submissions), 3)
        self.assertIn("--array=0-1%2", submissions[-1])
        self.assertEqual([c for c in calls if c.startswith("cancel")], ["cancel 456"])

    def test_exhausted_quota_or_other_error_keeps_original_test_held(self):
        for mode, attempts in (("quota_none", 4), ("other_error", 1)):
            with self.subTest(mode=mode):
                result, calls = self.launch(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len([c for c in calls if c.startswith("submit")]), attempts)
                self.assertFalse(any(c.startswith("cancel") or c == "control release 456" for c in calls))

    def test_running_test_is_never_cancelled(self):
        for mode in ("running", "race"):
            result, calls = self.launch(mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(any(c.startswith(("submit", "cancel")) for c in calls))

    def test_failed_handover_cancels_only_new_array(self):
        result, calls = self.launch("cancel_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cancel 777", calls)
        self.assertNotIn("cancel 123", calls)

    def test_release_failure_keeps_new_array_and_prints_recovery(self):
        result, calls = self.launch("release_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scontrol release 777", result.stderr)
        self.assertEqual([c for c in calls if c.startswith("cancel")], ["cancel 456"])


if __name__ == "__main__":
    unittest.main()
