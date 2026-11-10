# Get the main report results before the full test grid

For the submission deadline of **Thursday 12 November 2026, 11pm Singapore
time**, aim to finish the main result tables by Thursday noon. This is a planning
target, not a guaranteed runtime. Start writing the introduction, data, methods,
metrics and experimental setup while the GPU jobs run.

The original full test stage scores 720 ensemble/dataset configurations and
publishes the final chosen ensemble's combined table only after the full sweep.
This extra stage evaluates the systems needed for a smaller final report table
first, then lets the existing full test job continue.

## What it evaluates

All model and selector choices are frozen using **dev scores only**:

| System | Pretrained | Fine-tuned |
|---|---|---|
| Dev-selected ensemble, same three models and selector | Yes | Yes |
| Individual model with the best dev system score | Yes | Yes |

The strongest individual model is chosen across the 16 pretrained/fine-tuned dev
systems, prioritizing COMET22, then BLEU and chrF++. A complete tie prefers the
pretrained system, then registry model order. Both variants of that **same model**
are tested, making its before/after comparison matched.

Each of the four systems runs on WMT22, Tatoeba and the internal held-out 10%:
**12 report rows**, with BLEU, chrF++, COMET22 and BERTScore. The ensemble's primary
variant is still the one selected on dev; a higher test score cannot replace it.
All three selectors remain in the full dev comparison and the later full test
sweep. The priority stage uses only the frozen winning selector for its two
matched ensemble variants.

This requires test translations from six model/variant systems if the strongest
individual model is already in the ensemble, or eight if it is outside the trio.
It reuses valid candidates and individual-score checkpoints. The later full test
stage reuses those candidates and baseline checkpoints; some ensemble metrics
may be scored again in the full assignment layout. The running dev job and the
original fingerprinted training/evaluation files are unchanged.

## Queue it without cancelling the current dev job

In Windows PowerShell, from your local project directory, transfer only the new
files. Do not update the original recipe files during this run:

```powershell
scp -J shri@sjump.comp.nus.edu.sg scripts/priority_test.py scripts/run_priority_test.slurm scripts/submit_priority_test.sh shri@xlogin.comp.nus.edu.sg:/home/s/shri/cs4248-mt/scripts/
```

On the cluster, while dev job `1096786` runs and test job `1096787` is still pending
on it:

```bash
cd ~/cs4248-mt
bash scripts/submit_priority_test.sh ft702010_v1 1096786 1096787
squeue -u "$USER"
```

The submitter checks the existing test job and its dependency, temporarily holds
that pending job, inserts the priority job, updates the full test dependency,
then releases both. The chain becomes:

```text
Current dev job -> priority test results -> existing full test job
```

No running job is cancelled. Setup failure attempts to restore the original
dependency. If releasing the new chain fails, the printed command lets you
release it manually. If a prerequisite fails, dependent stages remain blocked
or are cancelled by Slurm; inspect the failed job before retrying.

Do not use this insertion command after switching to the parallel launcher, or
when the full test job has already started. Its checks will reject the wrong
state. It requires a pending full test job directly dependent on the supplied
dev job. For a different run, use that run's job IDs.

If dev already finished and **no other evaluation for this run is active**, the
priority stage can be submitted on its own:

```bash
sbatch scripts/run_priority_test.slurm ft702010_v1
```

This does not automatically schedule a subsequent full test job. The shared run
lock prevents concurrent evaluation from overwriting candidate or result files.

## Read and download the results

The submitter prints the new job ID. Its log is
`slurm-priority-test-<job-id>.log`. Status works on the login node without loading
GPU libraries:

```bash
python3 scripts/priority_test.py status --run-dir artifacts/finetune/ft702010_v1
```

Results appear under `artifacts/finetune/ft702010_v1/evaluation/priority_test/`:

| File | Use |
|---|---|
| `plan.json` | Frozen dev choices and their provenance |
| `report_scores.csv` | Four systems × three test sets, all four metrics |
| `report_deltas.csv` | Matched fine-tuned minus pretrained metric differences |
| `final_selected_scores.csv` | Three test rows for the dev-selected primary ensemble |
| `status.json` | `complete` only after every required result passes validation |

Download the result folder from Windows PowerShell:

```powershell
scp -r -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:/home/s/shri/cs4248-mt/artifacts/finetune/ft702010_v1/evaluation/priority_test .
```

Use the full 240-row **dev** table for the model/selector search and ablations,
and these frozen-system **test** tables for final performance. Do not label the
priority report as the full 720-row sweep. Document the LoRA recipe, data splits,
model/selector selection rule and any evaluation that remains unfinished.

Suggested writing schedule: complete data/methods/setup on Wednesday, integrate
results and error analysis on Thursday afternoon, and reserve Thursday evening
for figures, references, PDF checks and submission.

[Slurm holds, releases and job dependencies](https://slurm.schedmd.com/scontrol.html)
