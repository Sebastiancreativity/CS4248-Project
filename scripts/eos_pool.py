"""Run the existing EOS workflow with a small reusable pool of GPU jobs.

Workers claim tasks using filesystem locks, so progress never requires every
worker to start. No sbatch is called inside a worker. A lock is inherited by its
recipe child: another worker cannot duplicate a task while that child is alive.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from finetune_common import ROOT, fingerprint, read_json, sha256, write_json

POOL_CODE = ("eos_pool.py", "run_eos_pool.slurm", "eos_revision.py", "parallel_finetune_eval.py",
             "priority_test.py", "run_eos_revision.slurm")


def stages():
    def task(action, stage, index=None):
        return {"action": action, "stage": stage, "index": index}
    return [
        [task("prepare", "dev")],
        [task("generate", "dev", i) for i in (5, 13)],
        [task("seed", "dev")],
        [task("score", "dev", i) for i in range(10)],
        [task("collect", "dev")],
        [task("init", "test")],
        [task("generate", "test", i) for i in range(16)],
        [task("priority", "test")],
        [task("score", "test", i) for i in range(10)],
        [task("collect", "test")],
    ]


def location(revision):
    return revision.parent / (".eos-pool-" + revision.name)


def task_key(phase, task):
    return f"{phase:02d}-{task['action']}-{task['stage']}-{task['index'] if task['index'] is not None else 'all'}"


@contextmanager
def lock(path, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        from filelock import FileLock, Timeout
        mutex = FileLock(str(path))
        try:
            mutex.acquire(timeout=60 if blocking else 0)
        except Timeout:
            yield None
            return
        try:
            yield ()
        finally:
            mutex.release()
        return
    import fcntl
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
            return
        os.set_inheritable(handle.fileno(), True)
        yield (handle.fileno(),)
        # Close only: a still-running recipe child retains this lock.


def initialize(source, revision, dev_job):
    source, revision = source.resolve(), revision.resolve()
    if source == revision or source in revision.parents or revision in source.parents:
        raise ValueError("Use a separate evaluation revision")
    if not str(dev_job).isdigit():
        raise ValueError("The original dev job ID must be numeric")
    if (ROOT / "eos_checkout.json").exists():
        from prepare_eos_checkout import verify
        verify(ROOT)
    if not (source / "run.json").exists():
        raise FileNotFoundError("Original training run is missing")
    plan = {"source": str(source), "revision": str(revision), "dev_job": str(dev_job),
            "training_manifest_sha256": sha256(source / "run.json"), "stages": stages(),
            "code_sha256": {name: sha256(ROOT / "scripts" / name) for name in POOL_CODE}}
    plan["fingerprint"] = fingerprint(plan)
    directory = location(revision)
    with lock(directory / "control.lock", blocking=True) as held:
        if held is None:
            raise RuntimeError("Could not acquire pool initialization lock")
        path = directory / "plan.json"
        if path.exists():
            if read_json(path) != plan:
                raise ValueError("Pool code/settings changed; use a new revision name")
        elif revision.exists():
            raise ValueError("Revision already exists without a matching pool; use a new revision name")
        else:
            write_json(path, plan)
    return plan


def load_plan(revision):
    plan = read_json(location(revision) / "plan.json")
    if (plan["fingerprint"] != fingerprint({k: v for k, v in plan.items() if k != "fingerprint"})
            or Path(plan["revision"]) != revision or plan["stages"] != stages()):
        raise ValueError("Pool plan was modified")
    for name, expected in plan["code_sha256"].items():
        if sha256(ROOT / "scripts" / name) != expected:
            raise ValueError(f"Pool code changed: {name}")
    if sha256(Path(plan["source"]) / "run.json") != plan["training_manifest_sha256"]:
        raise ValueError("Training manifest changed")
    return plan


def done(directory, plan, key):
    path = directory / "done" / (key + ".json")
    return path.exists() and read_json(path).get("plan") == plan["fingerprint"]


def task_command(plan, task):
    command = [sys.executable, "-u", str(ROOT / "scripts/eos_revision.py"), task["action"],
               "--revision-dir", plan["revision"], "--stage", task["stage"]]
    if task["action"] == "prepare":
        command += ["--source-run", plan["source"]]
    if task["index"] is not None:
        command += ["--index", str(task["index"])]
    return command


def original_dev_ready(plan):
    path = Path(plan["source"]) / "evaluation/dev/status.json"
    status = read_json(path) if path.exists() else {}
    if status.get("status") == "complete":
        return True
    if status.get("status") == "failed":
        raise RuntimeError(f"Original dev evaluation failed: {status.get('error', 'check its log')}")
    # Do not keep GPUs waiting indefinitely after a failed/expired original job.
    result = subprocess.run(["squeue", "-h", "-j", plan["dev_job"], "-o", "%T"],
                            capture_output=True, text=True, timeout=20)
    if result.returncode == 0 and result.stdout.strip():
        return False
    accounting = subprocess.run(["sacct", "-n", "-X", "-j", plan["dev_job"],
                                 "--format=JobIDRaw,State", "--parsable2"],
                                capture_output=True, text=True, timeout=20)
    if accounting.returncode == 0:
        terminal = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "COMPLETED"}
        for row in accounting.stdout.splitlines():
            fields = row.strip().split("|")
            if len(fields) >= 2 and fields[0] == plan["dev_job"]:
                state = fields[1].split()[0].rstrip("+")
                if state in terminal:
                    # Recheck for a completion marker published just before exit.
                    if path.exists() and read_json(path).get("status") == "complete":
                        return True
                    raise RuntimeError(f"Original dev job is {state} without completed dev results; inspect its log")
    if result.returncode != 0:
        raise RuntimeError("Cannot verify the original dev job; retry after checking the scheduler")
    raise RuntimeError("Original dev job is absent and results are incomplete; inspect it before retrying")


def execute(plan, task, descriptors):
    command = task_command(plan, task)
    kwargs = {"pass_fds": descriptors} if os.name != "nt" else {}
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def work(directory, plan, worker, runner=execute, ready=original_dev_ready, pause=10):
    error_path = directory / "error.json"
    wait_logged = False
    with lock(directory / "workers" / (worker + ".lock")) as worker_fd:
        if worker_fd is None:
            raise RuntimeError("This worker ID is already active")
        while True:
            if error_path.exists():
                raise RuntimeError(f"Pool stopped: {read_json(error_path)['error']}")
            pending = next(((phase, tasks) for phase, tasks in enumerate(plan["stages"])
                            if any(not done(directory, plan, task_key(phase, task)) for task in tasks)), None)
            if pending is None:
                print("EOS workflow complete: all dev, priority report and full test results collected.", flush=True)
                return
            phase, tasks = pending
            try:
                if tasks[0]["action"] == "seed" and not ready(plan):
                    if not wait_logged:
                        print(f"Corrected dev generation ready; waiting for original dev job {plan['dev_job']} to finish scoring.", flush=True)
                        wait_logged = True
                    time.sleep(pause)
                    continue
                claimed = False
                for task in tasks:
                    key = task_key(phase, task)
                    if done(directory, plan, key):
                        continue
                    with lock(directory / "tasks" / (key + ".lock")) as task_fd:
                        if task_fd is None or done(directory, plan, key):
                            continue
                        if error_path.exists():
                            raise RuntimeError("Another pool task failed")
                        claimed = True
                        record = {"plan": plan["fingerprint"], "task": task, "worker": worker,
                                  "host": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat()}
                        write_json(directory / "running" / (key + ".json"), record)
                        print(f"START {key} worker={worker}", flush=True)
                        runner(plan, task, worker_fd + task_fd)
                        write_json(directory / "done" / (key + ".json"), {**record, "completed_utc": datetime.now(timezone.utc).isoformat()})
                        (directory / "running" / (key + ".json")).unlink(missing_ok=True)
                        print(f"DONE {key}", flush=True)
                    break
                if not claimed:
                    time.sleep(pause)
            except Exception as exc:
                with lock(directory / "control.lock", blocking=True):
                    if not error_path.exists():
                        write_json(error_path, {"plan": plan["fingerprint"], "worker": worker, "phase": phase, "error": str(exc)})
                raise


def retry(directory):
    # A retry never clears a live worker's failure barrier.
    from contextlib import ExitStack
    with ExitStack() as stack:
        for path in sorted((directory / "workers").glob("*.lock")):
            if stack.enter_context(lock(path)) is None:
                raise RuntimeError("A pool worker/child is still active; wait for it to stop before retrying")
        with lock(directory / "control.lock", blocking=True):
            path = directory / "error.json"
            if path.exists():
                archived = directory / ("error-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
                path.replace(archived)


def progress(revision):
    directory = location(revision)
    if not (directory / "plan.json").exists():
        print("Pool has not been prepared")
        return
    plan = load_plan(revision)
    count = 0
    for phase, tasks in enumerate(plan["stages"]):
        complete = sum(done(directory, plan, task_key(phase, t)) for t in tasks)
        count += complete
        print(f"{phase + 1:02d} {tasks[0]['stage']} {tasks[0]['action']}: {complete}/{len(tasks)}")
    print(f"Completed tasks: {count}/44 (executed within reusable GPU jobs)")
    if (directory / "error.json").exists():
        print("FAILED: " + read_json(directory / "error.json")["error"])
    print(f"Report: {revision / 'evaluation/priority_test/report_scores.csv'}")
    print(f"Full results: {revision / 'evaluation/test/scores_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "run", "retry", "status"])
    parser.add_argument("--revision-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--dev-job")
    args = parser.parse_args()
    revision = args.revision_dir.resolve()
    if args.action == "status":
        return progress(revision)
    if args.action == "init":
        if args.source_run is None or args.dev_job is None:
            parser.error("init requires --source-run and --dev-job")
        initialize(args.source_run, revision, args.dev_job)
        print(f"Pool prepared: {location(revision)}")
        return
    plan = load_plan(revision)
    if args.action == "retry":
        return retry(location(revision))
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU workers must be submitted through the pool Slurm launcher")
    worker = os.environ["SLURM_JOB_ID"] + "-" + os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    work(location(revision), plan, worker)


if __name__ == "__main__":
    main()
