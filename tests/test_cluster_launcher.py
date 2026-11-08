"""Check the environment-lock descriptor used by Slurm setup and training."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is required to exercise the launcher's file descriptor")
class ClusterLockTests(unittest.TestCase):
    def test_environment_lock_supports_both_read_and_write_without_truncating(self):
        script = (ROOT / "scripts/run_finetune.slurm").read_text(encoding="utf-8")
        opening = next(line for line in script.splitlines() if line.startswith("exec 9"))
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "finetune-environment.lock"
            lock.write_bytes(b"existing\n")
            # NFS implements shared flock with a read lock, so the descriptor
            # must support reading as well as the exclusive setup lock's writes.
            command = '\n'.join([
                'set -euo pipefail', 'CS4248_CLUSTER_DIR="$1"', opening,
                'IFS= read -r -u 9 contents', '[[ "$contents" == existing ]]',
                'printf "checked\\n" >&9',
            ])
            result = subprocess.run([BASH, "-c", command, "lock-check", Path(directory).as_posix()],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(lock.read_text(encoding="utf-8"), "existing\nchecked\n")
