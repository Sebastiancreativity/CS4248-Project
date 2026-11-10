# Resume evaluation across GPUs or share work with the team

All eight training jobs must have finished first. The adapters in
`artifacts/finetune/ft702010_v1/adapters/` are the trained LoRA weights and
tokenizers. Training used the 70% training partition; dev loss selected the saved
checkpoint. Dev translation generation and ensemble evaluation happen afterward.

The original evaluation job processes eight pretrained controls, their individual
scores and all four members' combinations, then repeats that work for the eight
fine-tuned models. This is 240 dev ensemble evaluations, with all three selectors,
over 3,068 sentences. Test evaluation contains 720 rows across three test sets.

## Inspect saved progress now

Run on the cluster, from `~/cs4248-mt`:

```bash
python3 scripts/parallel_finetune_eval.py status --run-dir artifacts/finetune/ft702010_v1
find artifacts/finetune/ft702010_v1/evaluation/dev -name 'scores_comparison.csv' -print
```

For the original sequential job, its log contains the current step:

```bash
tail -n 60 slurm-finetune-1096786_4294967294.log
```

`screen_model.py` means translation generation. `BERTScore agreement` means
peer-candidate scoring. Lines such as `01 dev_train20 cometkiwi: ...` report a
selected ensemble's reference-based evaluation. COMET22 and reference BERTScore
evaluate the chosen output; neither gold reference is used to pick candidates.

Read a finished member's results, for example:

```bash
python3 - <<'PY'
import csv
with open('artifacts/finetune/ft702010_v1/evaluation/dev/pretrained/Shri/scores_comparison.csv') as f:
    for r in csv.DictReader(f):
        print(r['experiment'], r['selector'], 'COMET22:', r['comet22'], 'BLEU:', r['bleu'])
PY
```

## Switch an unfinished run to parallel evaluation

This adds three new scripts; it does not edit the fingerprinted training recipe,
change decoding settings, retrain models, or discard completed results. Transfer
only these files if your cluster checkout was copied without Git. In Windows
PowerShell from your local project directory:

```powershell
scp -J shri@sjump.comp.nus.edu.sg scripts/parallel_finetune_eval.py scripts/run_parallel_finetune_eval.slurm scripts/submit_parallel_finetune_eval.sh shri@xlogin.comp.nus.edu.sg:/home/s/shri/cs4248-mt/scripts/
```

Stop the old evaluation jobs before launching replacements. For the reported
jobs, cancel the waiting test job first, then the active dev job:

```bash
cd ~/cs4248-mt
scancel 1096787 1096786
squeue -u "$USER"
```

Wait until both jobs have disappeared, including any `CG` cleanup state. If an
entire model/dataset generation has not yet been saved, cancelling repeats that
generation; the original script does not save each generation batch. To retain
that work too, wait for the model's JSON summary line before stopping. The next
model may already have started by then. Completed score checkpoints are reusable.

Then submit:

```bash
CS4248_EVAL_PARALLEL=4 bash scripts/submit_parallel_finetune_eval.sh ft702010_v1
```

This requests up to four simultaneous one-GPU workers. Actual concurrency depends
on your account's limits and available GPUs; raising this number cannot override
cluster quotas. Use `2` or up to `8` as appropriate. The old training-array
throttle does not parallelize the dev job.

The new dependency chain is:

1. Verify the run and all eight adapters; prepare private caches and metric downloads.
2. Sixteen generation tasks (eight models × two variants), skipping verified saved outputs.
3. Ten scoring tasks: eight member/variant tasks and two individual-model scoring tasks.
4. Validate and combine every result, then freeze the dev choice.
5. Repeat the parallel workflow on the three test sets, retaining that frozen choice.

Already-completed member evaluations are verified against their score checkpoints
and reused without rescoring. Unfinished workers reuse available checkpoints.
Each member has private candidate files and metric caches to avoid conflicting
writes. This may duplicate some peer-score work between members. Multiprocessing
temporary files use node-local `/tmp` instead of the network home directory.

The submit script automatically starts at test if `selection.json` already exists.
To explicitly restart only test:

```bash
CS4248_EVAL_PARALLEL=4 bash scripts/submit_parallel_finetune_eval.sh ft702010_v1 test
```

Jobs are held until the whole chain is submitted. A failed submission cancels
only jobs from that launch. A failed worker blocks collection and downstream test
jobs. Fix the failure, stop any remaining jobs from that launch, and resubmit;
completed candidates and scores are checked and reused. Do not run the sequential
launcher alongside this one. Do not change adapters, datasets or the new driver
after parallel preparation. Logs are `slurm-ft-eval-<job-id>_<task-id>.log`.

Final CSVs and `selection.json` remain in the original evaluation locations.
The combined dev file must have 240 rows, and test must have 720; no partial
result file is used to select the final system. Test scores never choose a winner.

## Give friends the trained adapters

No training checkpoint folders or base-model downloads need to be included. From
the cluster repository root, package the completed adapters and exact dev split:

```bash
tar --exclude='__pycache__' -czf ft702010-dev-adapters.tar.gz \
  scripts requirements-cluster.txt requirements-finetune.txt combinations*.json \
  artifacts/finetune/ft702010_v1/adapters \
  data/splits_70_20_10/dev_train20.zh \
  data/splits_70_20_10/dev_train20.en
```

On your Windows computer, download it:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:/home/s/shri/cs4248-mt/ft702010-dev-adapters.tar.gz .
```

Give the archive to your friends. It includes the exact script bytes used by this
run so line-ending conversions in another checkout cannot invalidate saved
generation identities. They need the repository, the
dependencies from `requirements-finetune.txt`, and a compatible CUDA environment.
The tested training/inference setup uses BF16-capable GPUs. Decode batch size 4
may exceed a smaller GPU's available memory; lower only the generation batch
size if necessary and record that difference. The original model weights will
download when first used; the adapters alone are not standalone full models.
Extract the archive from their repository root:

```bash
tar -xzf ft702010-dev-adapters.tar.gz
```

Do not run `finetune_pipeline.py`
with a copied `run.json`: that manifest refers to the original cluster paths.

On a Bash shell, choose the member and run:

```bash
OWNER="Reet"                    # Reet, Shri, Theodore, or Ming Zhe
COMBINATIONS=combinations_reet.json
export CS4248_MODEL_CACHE="$HOME/cs4248-model-cache"
python scripts/run_ensembles.py \
  --owner "$OWNER" --combinations "$COMBINATIONS" \
  --adapter-root artifacts/finetune/ft702010_v1/adapters \
  --model-outputs-dir artifacts/finetune/ft702010_v1/candidates/finetuned \
  --results-dir results/shared_ft702010_dev \
  --data-dir data/splits_70_20_10 --datasets dev_train20 \
  --selector all --comet --bertscore --bertscore-device cuda \
  --bertscore-batch-size 16 --generation-batch-size 4 --max-new-tokens 512
```

| Member | `OWNER` | `COMBINATIONS` |
|---|---|---|
| Reet | `Reet` | `combinations_reet.json` |
| Shri | `Shri` | `combinations_shri.json` |
| Theodore | `Theodore` | `combinations_theodore.json` |
| Ming Zhe | `Ming Zhe` | `combinations_ming_zhe.json` |

This uses the saved fine-tuned adapters and all three selectors. It does not
retrain. Each member returns the whole `results/shared_ft702010_dev/<OWNER>/`
directory, including hypotheses and score checkpoints, not just a screenshot.
These independent results require validation and collection before freezing a
final system; do not select a winner from only one person's ten combinations.
Use separate `--results-dir` paths for concurrent processes on the same machine.

## Share already-generated candidates to save time

After all eight fine-tuned dev outputs are complete, include their candidate
files as well. Do not archive actively written output directories. Verify they
are ready first using the progress command (8/8 fine-tuned dev candidates).

```bash
tar --exclude='__pycache__' -czf ft702010-dev-ready.tar.gz \
  scripts requirements-cluster.txt requirements-finetune.txt combinations*.json \
  artifacts/finetune/ft702010_v1/adapters \
  artifacts/finetune/ft702010_v1/candidates/finetuned \
  data/splits_70_20_10/dev_train20.zh \
  data/splits_70_20_10/dev_train20.en
```

Friends extract this archive and add `--no-generate` to the same command above.
The translation models do not need to be loaded or downloaded; only the metric
models run. Adapter files remain included so generation provenance can be
verified. All eight candidate outputs can be reused for every member's triples.
Keep source files byte-identical when copying the repository: generation cache
identities include the recipe file hashes, including line endings.

On the cluster, completed generation timing and simple BLEU/chrF++ scores are in
each model's `candidates/finetuned/<model>/output/scores.jsonl`. A null COMET value
there is expected: COMET is computed in the later evaluation stage.

[Slurm job arrays and concurrency limits](https://slurm.schedmd.com/job_array.html)
