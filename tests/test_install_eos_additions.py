import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import install_eos_additions as installer


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.repo = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        installer.git(self.repo, "init", "-q")
        installer.git(self.repo, "config", "core.autocrlf", "false")
        self.payloads = {name: ("new helper: " + name + "\n").encode() for name in installer.FILES}
        for name, value in self.payloads.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        self.code = self.repo / "scripts/finetune_pipeline.py"
        self.code.write_text("original tracked version\n")
        self.data = self.repo / "train.zh-en.zh"
        self.data.write_text("original tracked data\n")
        installer.git(self.repo, "add", ".")
        installer.git(self.repo, "-c", "user.name=Installer Test", "-c", "user.email=test@example.invalid",
                      "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        self.commit = installer.git(self.repo, "rev-parse", "HEAD").decode().strip()
        for name in self.payloads:
            (self.repo / name).unlink()
        # Simulate the cluster's locally copied training recipe and cleaned data.
        self.code.write_text("local recipe used by the active run\n")
        self.data.write_text("local cleaned training data\n")
        self.run = self.repo / "artifacts/finetune/current"
        self.run.mkdir(parents=True)
        spec = {"recipe_sha256": {"scripts/finetune_pipeline.py": installer.sha256(self.code)},
                "input_sha256": {str(self.data): installer.sha256(self.data)}}
        manifest = {"spec": spec, "fingerprint": hashlib.sha256(
                    json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        (self.run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_install_preserves_dirty_code_data_index_and_head(self):
        before = {p: p.read_bytes() for p in (self.code, self.data, self.run / "run.json", self.repo / ".git/index")}
        installer.install(self.repo, self.run, self.commit)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        for name, content in self.payloads.items():
            self.assertEqual((self.repo / name).read_bytes(), content)
        self.assertEqual(installer.git(self.repo, "rev-parse", "HEAD").decode().strip(), self.commit)
        self.assertFalse(list((self.repo / "scripts").glob(".eos-install-*")))

    def test_existing_different_helper_blocks_all_writes(self):
        existing = self.repo / installer.FILES[-1]
        existing.write_text("user edit\n")
        with self.assertRaisesRegex(ValueError, "nothing was installed or overwritten"):
            installer.install(self.repo, self.run, self.commit)
        self.assertEqual(existing.read_text(), "user edit\n")
        self.assertTrue(all(not (self.repo / name).exists() for name in installer.FILES[:-1]))

    def test_identical_helpers_are_not_rewritten(self):
        installer.install(self.repo, self.run, self.commit)
        times = {name: (self.repo / name).stat().st_mtime_ns for name in installer.FILES}
        installer.install(self.repo, self.run, self.commit)
        self.assertEqual(times, {name: (self.repo / name).stat().st_mtime_ns for name in installer.FILES})

    def test_changed_active_input_prevents_installation(self):
        self.data.write_text("changed since training preparation\n")
        with self.assertRaisesRegex(ValueError, "active run's code/data changed"):
            installer.install(self.repo, self.run, self.commit)
        self.assertTrue(all(not (self.repo / name).exists() for name in installer.FILES))

    def test_manifest_tampering_prevents_installation(self):
        path = self.run / "run.json"
        manifest = json.loads(path.read_text())
        manifest["spec"]["recipe_sha256"] = {}
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest was modified"):
            installer.install(self.repo, self.run, self.commit)
        self.assertTrue(all(not (self.repo / name).exists() for name in installer.FILES))


if __name__ == "__main__":
    unittest.main()
