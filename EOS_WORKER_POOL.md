# Run within the cluster's submission limit

The original EOS launcher submitted 44 array elements/jobs in advance. `%4`
limited simultaneous execution but did not reduce the queued job count. Slurm
counts each array element toward `MaxSubmitJobs`, which caused
`AssocMaxSubmitJobLimit`.

The pool submits **one array with at most four GPU workers**. Those workers run
the same 44 workflow tasks within their allocations, with no further `sbatch`
calls. If submission quota is too small, the helper tries three, two, then one
worker. If no slot is available, it preserves the original jobs and reports the
problem. Other submission errors stop immediately.

Workers take available tasks rather than having fixed assignments. The workflow
can finish even if only one worker receives a GPU and the others remain pending.
Phase barriers preserve dev selection before test. File locks prevent duplicate
generation/scoring; completed tasks survive a retry. The original dev job is
never cancelled or modified by the helper.

## Update the already prepared separate checkout

Run these commands once from the cluster login node. Fetch and restore only the
three new pool files; do not merge/reset the checkout's preserved recipe/data.

```bash
cd ~/cs4248-mt-eos &&
git fetch origin MT &&
git restore --source origin/MT -- scripts/eos_pool.py scripts/run_eos_pool.slurm scripts/submit_eos_pool.sh &&
bash scripts/submit_eos_pool.sh ft702010_v1 ft702010_eos_v1 1096786 1096787
```

The helper verifies the prepared checkout, holds the old pending test (already
held is fine), submits the worker array on hold, and cancels **only the old
pending test** after the replacement is accepted. It then releases the workers.
A failed submission leaves the original test's hold state as it was. A failed
release prints the exact release command. Do not rerun the handover after it
successfully replaces the old test.

To request fewer workers from the outset:

```bash
CS4248_EVAL_PARALLEL=2 bash scripts/submit_eos_pool.sh ft702010_v1 ft702010_eos_v1 1096786 1096787
```

## Progress and outputs

```bash
squeue -u "$USER"
python3 scripts/eos_pool.py status --revision-dir artifacts/finetune/ft702010_eos_v1
```

Logs are `slurm-eos-pool-<array-id>_<worker>.log`. The pool first validates the two
corrected Tower-Mistral dev candidate sets, then waits for the original dev job
to finish scoring. It reuses unchanged dev results and freezes the corrected
dev choice. **Test generation still follows dev selection.** It generates all
test candidates with the fixed stopping rule, publishes the selected-system
report, then scores all members' combinations with all three selectors.

Outputs under `artifacts/finetune/ft702010_eos_v1/` are unchanged:

- `evaluation/priority_test/report_scores.csv`: 12 report rows.
- `evaluation/test/scores_comparison.csv`: all 720 ensemble test rows, with the
  full BLEU/chrF++/COMET22/BERTScore fields and selection counts.
- `evaluation/test/individual_scores.csv`: 48 individual test rows.
- `evaluation/test/final_selected_scores.csv`: the three primary final results.

The pool uses the existing adapters, corrected EOS evaluator, data, metric
caches and runtime environment. It does not retrain or change decoding further.
Each worker requests one A100-40 slice, 8 CPUs, 64 GB RAM and a 48-hour limit.
The test/model/scoring work is unchanged; the number of jobs submitted changes.

## If a worker fails

The pool records a shared error and stops claiming new tasks. Already running
tasks can finish and save checkpoints. After inspecting the failure and waiting
for every pool worker to stop, clear the failure barrier and resubmit workers:

```bash
python3 scripts/eos_pool.py retry --revision-dir artifacts/finetune/ft702010_eos_v1 &&
sbatch --array=0-1%2 scripts/run_eos_pool.slurm ft702010_eos_v1
```

Retry refuses to clear the barrier while a worker or its recipe child still
holds a worker lock. Completed tasks are retained. Do not modify the pinned
driver, source data or adapters while the pool is running.
