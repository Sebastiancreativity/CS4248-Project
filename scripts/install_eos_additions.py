"""Install only missing EOS helper files from a fetched Git commit.

For cluster checkouts populated by SCP: no merge, stash, reset, index update,
package install or modification of existing files. Python standard library only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

FILES = ("scripts/eos_revision.py", "scripts/parallel_finetune_eval.py",
         "scripts/priority_test.py", "scripts/run_eos_revision.slurm",
         "scripts/submit_eos_revision.sh")


def git(repo, *args):
    return subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(repo, run_dir):
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(json.dumps(run["spec"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if run["fingerprint"] != actual:
        raise ValueError("The existing run manifest was modified. No helper files installed.")
    protected = run["spec"]["recipe_sha256"]
    if set(FILES) & set(protected):
        raise ValueError("An addition is part of this run's protected recipe. No files installed.")
    checks = [(repo / name, expected) for name, expected in protected.items()]
    checks += [(Path(name), expected) for name, expected in run["spec"]["input_sha256"].items()]
    for path, expected in checks:
        if sha256(path) != expected:
            raise ValueError(f"The active run's code/data changed: {path}. No helper files installed.")
    return protected


def install(repo, run_dir, ref):
    repo, run_dir = repo.resolve(), run_dir.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", ref):
        raise ValueError("Use a commit hash or branch name")
    root = Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if root != repo:
        raise ValueError("Run from the repository root")
    commit = git(repo, "rev-parse", "--verify", ref + "^{commit}").decode().strip()
    before = git(repo, "rev-parse", "HEAD")
    verify_run(repo, run_dir)
    payloads = {name: git(repo, "show", f"{commit}:{name}") for name in FILES}
    missing, conflicts = [], []
    for name, data in payloads.items():
        target = repo / name
        if target.is_symlink() or not target.resolve().is_relative_to(repo):
            conflicts.append(name + " (linked outside the installation target)")
        elif target.exists():
            if not target.is_file() or target.read_bytes() != data:
                conflicts.append(name)
        else:
            missing.append(name)
    if conflicts:
        raise ValueError("Existing helper files differ; nothing was installed or overwritten: " + ", ".join(conflicts))
    for name in missing:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete file, and refuse to replace a file created by
        # another process since preflight. Both names are on the same filesystem.
        descriptor, temporary_name = tempfile.mkstemp(prefix=".eos-install-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payloads[name])
            temporary.chmod(0o644)
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Installed {name}")
    verify_run(repo, run_dir)
    if git(repo, "rev-parse", "HEAD") != before:
        raise ValueError("Another process changed Git HEAD during installation; inspect before submitting")
    print(f"Ready: {len(missing)} files installed, {len(FILES) - len(missing)} already identical; source commit {commit[:12]}.")
    print("Existing code, data, results, adapters and Git HEAD were preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/finetune/ft702010_v1"))
    parser.add_argument("--commit", default="origin/MT")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else args.repo / args.run_dir
    try:
        install(args.repo, run_dir, args.commit)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Installation stopped: {exc}\n")


if __name__ == "__main__":
    main()
