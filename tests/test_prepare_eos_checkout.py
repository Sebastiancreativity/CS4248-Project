from pathlib import Path
import tempfile
import unittest

import test_install_eos_additions as fixture_tests
import install_eos_additions as installer
import prepare_eos_checkout as checkout


class CheckoutTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_tests.InstallTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.parent = Path(self.fixture.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.destination = self.parent / "fresh"
        installer.git(self.fixture.repo, "worktree", "add", "--quiet", "--detach", str(self.destination), self.fixture.commit)

    def test_separate_copy_keeps_original_files_and_reuses_same_training_directory(self):
        original = {p: p.read_bytes() for p in (self.fixture.code, self.fixture.data, self.fixture.run / "run.json")}
        before_head = installer.git(self.fixture.repo, "rev-parse", "HEAD")
        checkout.prepare(self.fixture.repo, self.destination, "current")
        plan = checkout.verify(self.destination)
        self.assertEqual(plan["source_run"], str(self.fixture.run))
        self.assertEqual((self.destination / "artifacts/finetune/current").resolve(), self.fixture.run)
        self.assertEqual((self.destination / "scripts/finetune_pipeline.py").read_bytes(), self.fixture.code.read_bytes())
        self.assertEqual((self.destination / "train.zh-en.zh").read_bytes(), self.fixture.data.read_bytes())
        for path, value in original.items():
            self.assertEqual(path.read_bytes(), value)
        self.assertEqual(installer.git(self.fixture.repo, "rev-parse", "HEAD"), before_head)
        for name, value in self.fixture.payloads.items():
            self.assertEqual((self.destination / name).read_bytes(), value)

    def test_changed_copy_is_detected_before_a_worker_can_use_it(self):
        checkout.prepare(self.fixture.repo, self.destination, "current")
        (self.destination / "train.zh-en.zh").write_text("different data")
        with self.assertRaisesRegex(ValueError, "Source/copy differs"):
            checkout.verify(self.destination)

    def test_dirty_destination_and_original_directory_are_refused(self):
        with self.assertRaisesRegex(ValueError, "separate sibling"):
            checkout.prepare(self.fixture.repo, self.fixture.repo, "current")
        path = self.destination / "scripts/finetune_pipeline.py"
        path.write_text("new checkout has a user edit")
        with self.assertRaisesRegex(ValueError, "fresh, clean checkout"):
            checkout.prepare(self.fixture.repo, self.destination, "current")
        self.assertEqual(path.read_text(), "new checkout has a user edit")

    def test_repeated_preparation_verifies_without_rewriting(self):
        checkout.prepare(self.fixture.repo, self.destination, "current")
        path = self.destination / "train.zh-en.zh"
        timestamp = path.stat().st_mtime_ns
        checkout.prepare(self.fixture.repo, self.destination, "current")
        self.assertEqual(path.stat().st_mtime_ns, timestamp)


if __name__ == "__main__":
    unittest.main()
