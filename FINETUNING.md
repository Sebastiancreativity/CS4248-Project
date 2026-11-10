# Fine-tune, compare on development data, then evaluate on tests

The earlier experiments used pretrained checkpoints without project-specific
weight updates. This pipeline **actually trains LoRA adapters** using the cleaned
course training data. Fine-tuning may improve translation, but it can also
overfit or reduce quality. The pipeline therefore measures both pretrained and
adapted systems under matching conditions, and can select the pretrained system.

## Run the complete workflow on the cluster

From your existing repository checkout on the Slurm login node, with Python
3.10–3.12 available:

```bash
git switch MT
git pull --ff-only
bash scripts/submit_finetune_pipeline.sh ft702010_v1
```

This submits dependent jobs automatically:

1. Install the environment and verify the data split.
2. Fine-tune all **eight assigned models**, one model per GPU task.
3. Evaluate all eight individual models and all four members' **40 triples ×
   three selectors**, both before and after adaptation, on `dev_train20`.
4. Freeze the highest dev COMET22 configuration, then evaluate all those systems
   on the internal 10% holdout, official WMT22, and official Tatoeba tests.

No model weights are downloaded or trained on your laptop by these commands.
The scheduler allocates **one A100 40 GB, 8 CPUs and 64 GB RAM per task**, with a
48-hour limit per task. By default only one training task runs at a time. If your
GPU quota permits two concurrent tasks:

```bash
CS4248_FT_PARALLEL=2 bash scripts/submit_finetune_pipeline.sh ft702010_v1
```

Use a new run name when changing data, code or training settings. Optional
settings, which must be chosen before looking at test results:

```bash
CS4248_FT_EPOCHS=2 CS4248_FT_LR=0.00005 \
  bash scripts/submit_finetune_pipeline.sh ft702010_lr5e5
```

Your site may use different resource names. Set `CS4248_FT_PARTITION`,
`CS4248_FT_GPUS` (for example your site's H100 allocation name), and
`CS4248_FT_TIME` before submission to override the defaults. The GPU must support
BF16. Set `CS4248_CLUSTER_DIR` to an **absolute directory on shared cluster
storage** if your home quota cannot hold the environments and downloaded weights.
The repository and runtime must be accessible from every allocated node.

The setup job creates `$CS4248_CLUSTER_DIR/venv-finetune`; the default runtime is
`$HOME/cs4248-mt-runtime`. It reuses the existing model download cache. It does
not use the old inference environment. If Hugging Face requests access to a
checkpoint such as COMETKiwi, accept its conditions and authenticate on the
cluster before retrying. Downloads require network access from the allocated node.

## What is trained and what is held out?

The fixed seed is `4248`; splits are 70/20/10 after cleaning, deduplication, and
normalized-source exclusion against official test sources. Current counts:

| Data | Pairs | Use |
|---|---:|---|
| Training partition | 10,739 | Gradient updates to LoRA parameters |
| `dev_train20` | 3,068 | Checkpoint and model/ensemble selection |
| `test_train10` | 1,535 | Internal held-out evaluation only |
| Official WMT22 | 1,875 | Final evaluation only |
| Official Tatoeba | 469 | Final evaluation only |

These use the cleaned `train.zh-en.zh` / `train.zh-en.en` files. The old external
`dev_wmt`, `dev_short` and `dev_alma` datasets are **not** used in this workflow.
Training labels contain only the English targets. For decoder-only models,
Chinese source/prompt tokens and padding have ignored loss labels. NLLB uses
Chinese source and English target language tokens; MADLAD uses its `<2en>` prefix.
The two encoder-decoder models train adapters in both encoder and decoder blocks.

| Array index | Model | Adaptation |
|---:|---|---|
| 0 | Qwen2.5-1.5B-Instruct | 4-bit QLoRA |
| 1 | Qwen2.5-3B-Instruct | 4-bit QLoRA |
| 2 | Qwen2.5-7B-Instruct | 4-bit QLoRA |
| 3 | Hunyuan-MT-7B | 4-bit QLoRA |
| 4 | TowerInstruct-7B-v0.2 | 4-bit QLoRA |
| 5 | TowerInstruct-Mistral-7B-v0.2 | 4-bit QLoRA |
| 6 | MADLAD400-3B-MT | BF16 LoRA |
| 7 | NLLB-200-distilled-1.3B | BF16 LoRA |

ALMA is excluded because it is not in the current 40 assigned combinations.
LoRA updates additional trainable matrices while the original base weights stay
frozen; this is parameter-efficient fine-tuning, not full-weight fine-tuning.
COMETKiwi, COMET22 and the BERTScore encoder remain fixed evaluation/selection
models; they are not trained on the course data.

Default training: 3 epochs, learning rate `1e-4`, rank 16, alpha 32, dropout 0.05,
all linear transformer layers, AdamW, cosine schedule, 5% warmup, BF16,
gradient checkpointing, microbatch 1, gradient accumulation 16. Checkpoints are
saved each epoch; the checkpoint with minimum dev translation loss is restored.
There is no claim that these initial settings are optimal for every model.

Maximum training source/prompt and target lengths are 512 tokens each.
Overlength pairs are skipped **without truncating either side**, and their exact
row numbers/counts are recorded in `training/<model>/tokenization.json`.
Final dev/test generation evaluates every sentence, including rows excluded
from loss evaluation by the training length limit.

## How to tell whether fine-tuning helped

Both variants use the same pinned base checkpoint revision, tokenizer, prompt,
precision, greedy decoding and 512-token output cap. Decoder-only pretrained
controls use the same 4-bit precision as QLoRA. This is a fresh comparison;
older spreadsheet scores used a 256-token cap and are not its matched controls.

Every ensemble runs COMETKiwi, chrF++ agreement and BERTScore agreement.
Candidate selection never uses gold English translations. Reference-based
BLEU (13a), chrF++, COMET22 and BERTScore are computed afterward.

The primary system is selected from **dev** scores, prioritizing COMET22, with
BLEU and then chrF++ as tie-breakers. If all those rounded scores tie, the
pretrained version wins before assignment order breaks remaining ties. The
best configuration for each selector is also saved. These are dev choices,
not guarantees of statistically significant improvements.

Results live under `artifacts/finetune/ft702010_v1/`:

| File/folder | Contents |
|---|---|
| `run.json` | Data/code fingerprints, settings and counts |
| `adapters/<model>/` | Adapter weights, tokenizer and training completion record |
| `training/<model>/` | Resumable checkpoints, training history and tokenization counts |
| `evaluation/dev/individual_deltas.csv` | Eight matched before/after model comparisons |
| `evaluation/dev/ensemble_deltas.csv` | 120 matched before/after ensemble comparisons |
| `evaluation/dev/scores_comparison.csv` | 240 rows: 40 triples × 3 selectors × 2 variants |
| `selection.json` | Frozen dev-selected primary system and best per selector |
| `evaluation/test/final_selected_scores.csv` | Three test rows for the frozen primary system |
| `evaluation/test/scores_comparison.csv` | All 720 test rows: 40 × 3 selectors × 2 variants × 3 datasets |
| `evaluation/test/individual_scores.csv` | All 48 individual-model test rows |

In delta files, **positive means fine-tuning scored higher**. Compare each model
or triple against itself. Test results are final measurements; do not use them
to replace the frozen winner or tune training settings. Both variants' outputs
and caches are isolated from each other and from earlier experiments.

## Monitor and resume

For parallel dev/test evaluation that resumes the existing run, progress counts,
and commands for sharing trained adapters or saved candidates with teammates,
see [PARALLEL_EVALUATION.md](PARALLEL_EVALUATION.md). The parallel launcher keeps
the original recipe and run fingerprints unchanged.

```bash
squeue -u "$USER"
```

The submit command prints the setup, training, dev and test job IDs. Logs are
`slurm-finetune-<job-or-array-id>_<task-id>.log` in the checkout. Each stage starts
only after its prerequisites succeed. A failed prerequisite prevents dependent
jobs from starting; check the failed stage's log.

After correcting an external issue such as checkpoint access, rerun the same
submission command with the **same run name and training settings**. Completed
adapters are verified and skipped, interrupted training resumes from the last
saved epoch checkpoint, and generation/evaluation caches are reused when valid.
An interrupted epoch may need to be repeated. A whole-dataset generation that
was interrupted before saving also needs to be repeated. Do not submit a second
copy while the first is still running.

To resubmit just a stage after its prerequisites have already finished:

```bash
# Example: resume only MADLAD training (array index 6).
sbatch --array=6 scripts/run_finetune.slurm train ft702010_v1

# After all eight adapters are complete:
sbatch scripts/run_finetune.slurm dev ft702010_v1

# After development evaluation and selection are complete:
sbatch scripts/run_finetune.slurm test ft702010_v1
```

Do not edit code/data or replace adapters under a completed run. Integrity checks
will reject mismatches instead of silently combining incompatible experiments.

If an older launcher reports `flock: 9: Bad file descriptor` immediately in every
training task, update `scripts/run_finetune.slurm`. Its environment lock must open
descriptor 9 for both reading and writing (`exec 9<>...`), since shared locks on
network storage may require a readable descriptor. This launcher-only correction
does not change the training recipe: reuse the same run name, prepared split and
installed environment, then resubmit the complete workflow.

## Local verification and method references

The dependency-light checks run with `python -B -m unittest discover -s tests -v`.
They exercise source/target masking, split separation, adapter integrity,
pretrained-vs-adapted comparisons, and dev-only selection before test evaluation.
`tests/smoke_finetune_models.py` additionally checks tiny randomly initialized
models using PyTorch, Transformers and PEFT; it does not download full weights.
Neither check substitutes for running the full CUDA jobs on the cluster.

- [PEFT quantization and QLoRA guidance](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization)
- [Transformers Trainer](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer)
- [ALMA translation fine-tuning paper](https://arxiv.org/abs/2309.11674): evidence
  that adaptation with high-quality parallel data can help in its studied
  setting, not a promise of gains for our eight already-trained checkpoints.
