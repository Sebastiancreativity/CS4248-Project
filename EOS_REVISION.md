# Tower-Mistral stopping correction

The cluster A/B test on eight fine-tuned dev sentences measured 67.03 seconds
with the published generation EOS (`2`), versus 5.91 seconds with the tokenizer's
`<|im_end|>` EOS (`32000`): **11.34x faster**. Both repeats produced identical
cleaned translations on those eight sentences. All original generations reached
512 tokens, averaging 473 tokens after the message-end token. None of the fixed
generations reached the cap. These are sample measurements, not a full-test ETA.

The correction applies to **both pretrained and fine-tuned Tower-Mistral**.
Model weights, quantization, prompts, greedy decoding, batch size and the
512-token maximum remain the same. No model is retrained.

## Replace the pending test job

On the cluster, with the original dev job still running and its test job pending:

```bash
cd ~/cs4248-mt
git pull --ff-only origin MT
bash scripts/submit_eos_revision.sh ft702010_v1 ft702010_eos_v1 1096786 1096787
```

The arguments are the original training run, a new evaluation directory name,
the original dev job ID, and the pending test job ID. A test job already held
with `scontrol hold` is supported. A running test job is refused. The pending
test must still have its original `afterok` dependency on the given dev job.

The helper holds the old test, queues all replacement jobs on hold, then cancels
**only the old pending test** and releases the new jobs. It never cancels the
running dev job. A submission failure restores the old test's prior hold state
and cancels only this launch's new held jobs. A failure releasing the completed
replacement prints the exact release command; it preserves the new chain.

By default, generation/scoring arrays allow four concurrent GPU workers,
subject to the account's quota and cluster availability. To request two:

```bash
CS4248_EVAL_PARALLEL=2 bash scripts/submit_eos_revision.sh ft702010_v1 ft702010_eos_v1 1096786 1096787
```

Run one submission command once. Do not run both examples or resume the old
test launcher on the revision directory.

### If git pull refuses to overwrite local files

A cluster checkout populated by earlier SCP uploads can contain locally modified
or untracked recipe files. A refused pull does not install the new launcher.
Do not stash, reset, clean or force-checkout the active training/evaluation files.
The preferred recovery is a **separate checkout**:

```bash
cd ~/cs4248-mt
scontrol hold 1096787
git fetch origin MT
git worktree add --detach ~/cs4248-mt-eos origin/MT
cd ~/cs4248-mt-eos
python3 -B scripts/prepare_eos_checkout.py --source-repo ~/cs4248-mt --run-name ft702010_v1 &&
bash scripts/submit_eos_revision.sh ft702010_v1 ft702010_eos_v1 1096786 1096787
```

This leaves the active checkout intact. The preparation step copies the exact
original recipe and cleaned inputs into the fresh checkout, retains the new EOS
drivers, and links to the original run's adapters/results. A bare clone without
this step might contain different cleaned data or fail the run's code checks.
Every new worker verifies the copied inputs against the original run before
starting. Existing adapters and the shared model cache are reused; there is no
retraining or full model copy. The new outputs and logs live under
`~/cs4248-mt-eos`, and the original dev job continues under `~/cs4248-mt`.

Changing folders does not change the experiment. Corrected stopping remains the
only intended decoding difference and is checked on full dev before test. This
deployment method does **not** remove the existing dev-to-test dependency.

Alternatively, use the additive installer in the existing checkout; it verifies the prepared run's hashes and
installs only the five missing EOS/parallel/priority helpers. Existing files must
be byte-identical or installation stops before writing anything. Git HEAD and
the index are not changed.

```bash
cd ~/cs4248-mt
scontrol hold 1096787
git fetch origin MT
installer=$(mktemp /tmp/cs4248-eos-install.XXXXXXXX.py)
git show origin/MT:scripts/install_eos_additions.py > "$installer" &&
python3 "$installer" --commit origin/MT --run-dir artifacts/finetune/ft702010_v1 &&
bash scripts/submit_eos_revision.sh ft702010_v1 ft702010_eos_v1 1096786 1096787
```

The `&&` conditions prevent job submission if installation fails. If there is an
error, leave the old test held and inspect the message; the current dev job is
unaffected. This installs the same pipeline described below: test generation
still follows corrected dev selection.

## Work and dependencies

1. Snapshot the completed dev raw translations and link to the eight existing
   trained adapters. All new outputs have their own directory. Preparation does
   not wait for the original dev scoring job.
2. Regenerate only Tower-Mistral's full dev translations for both variants, in
   two GPU array tasks. This can overlap the original dev job's scoring.
3. After both validation generation and the original dev job finish, copy the
   original metric caches and completed score checkpoints. Other models' dev
   translations are reused.
4. Evaluate the corrected dev candidates with all three selectors. A completed
   owner's original scores are reused only when **every candidate translation
   for that owner's combinations is identical**. Changed inputs force selection
   to run again; content-keyed metric checkpoints reuse identical final outputs.
   Individual scoring reuses the seven unaffected models' checkpoints.
5. Compare all 3,068 Tower-Mistral dev sentences per variant and freeze a new
   configuration using dev COMET22, with the existing tie-breaking rule.
6. Generate all eight models' test translations, both variants, on all three
   test sets using up to four GPU workers. Tower-Mistral uses the corrected EOS.
7. Publish the selected ensemble and individual baseline first, then complete
   all 40 combinations x 3 selectors x 2 variants x 3 test sets: **720 rows**.

All test ensemble workers retain COMETKiwi, chrF++ agreement and BERTScore
agreement selection. Gold references are used for evaluation, not candidate
selection. Test results do not determine the final ensemble choice.

The original job's 7-hour Tower-Mistral dev generation is already completed;
the replacement does not repeat it with the old setting. Time spent scoring
other systems, loading models and waiting for GPUs is separate from the stopping
bug and still contributes to total runtime.

## Results and progress

```bash
python3 scripts/eos_revision.py status --revision-dir artifacts/finetune/ft702010_eos_v1
squeue -u "$USER"
```

Under `artifacts/finetune/ft702010_eos_v1/`:

- `dev_eos_comparison.json`: full-dev changed-translation counts for both variants.
- `selection.json`: the choice frozen from corrected dev results.
- `evaluation/priority_test/report_scores.csv`: 12 early report rows.
- `evaluation/test/scores_comparison.csv`: all 720 ensemble rows, including
  experiment, dev (dataset), models, selector, n, BLEU, chrF++, COMET22,
  BERTScore precision/recall/F1/hash and selection_counts, plus owner/variant.
- `evaluation/test/individual_scores.csv`: all 48 individual test evaluations.
- `evaluation/test/final_selected_scores.csv`: the three test rows for the
  dev-selected primary configuration.
- `evaluation/test/finetuned/<member>/scores_comparison.csv`: member-specific
  results for sharing.

Logs are `slurm-eos-<job-id>_<array-index>.log`; non-array jobs use the cluster's
normal non-array index representation. Candidate marker counts are progress
indicators, not proof that scoring is complete.

## Reproducibility

The original recipe files and training manifests remain unchanged. The revision
records its decoding policy, code hashes, original input hashes, adapter hashes
and the completed dev score snapshot it reused. It has no training `run.json`;
legacy launchers intentionally cannot operate on it. Runtime compatibility
bindings apply only inside `eos_revision.py` processes, including every scoring
and generation child. Only Tower-Mistral gets a new generation identity, so its
old candidates cannot be mistaken for corrected ones. Other models retain their
verified identities and reusable caches. Adapter links are only read by this
evaluation workflow; candidate files and scoring caches are separate copies.

If a replacement job fails, inspect its log and dependencies. Completed revision
candidates/checkpoints are reusable by rerunning the relevant action with
`run_eos_revision.slurm`; do not edit fingerprints or delete original results.
