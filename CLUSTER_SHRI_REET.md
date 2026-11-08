# Shri and Reet: training-derived development split

The ten combinations per owner follow the supplied assignment table, in its
experiment order. TM is `tower-mistral-7b`; no combination uses ALMA.

The new job uses all three selectors independently: BERTScore agreement,
chrF++ agreement and COMETKiwi quality estimation. Gold references are used
after selection for BLEU, chrF++, COMET-22 and BERTScore P/R/F1 evaluation.
Each owner produces 30 evaluation rows on `dev_train20` (10 combinations x
3 selectors; 60 rows for Shri and Reet together).
This does not fine-tune the pretrained models. Any later fine-tuning must use
only `train.zh`/`train.en`, not the original unsplit training corpus.

## Split

`python scripts/build_train_splits.py` reads the root `train.zh-en.zh` and
`train.zh-en.en`, removes empty pairs and duplicate normalized source sentences
(retaining the first reference), and excludes sources in `tatoeba.zh` and
`wmttest2022.zh`. Normalization uses Unicode NFKC, case folding and whitespace
removal. It does not detect semantic paraphrases or traditional/simplified variants.

With seed 4248, 15,342 eligible pairs give 10,739 train, 3,068 dev and 1,535 test
pairs. Thirty duplicate sources were removed. Train/dev counts are rounded
down; test gets the remainder. This dev set is larger than the former two dev
sets combined, so fewer experiments do not guarantee a shorter generation run.

Files are in `data/splits_70_20_10/`, with original line IDs and input hashes.
The builder refuses to overwrite a different existing split. The GPU job
reads only `dev_train20`; `test_train10` is reserved for final evaluation after
the winning configuration is fixed. Existing external test datasets stay intact.
If models were already fine-tuned on the entire original corpus, those models
must be retrained on the 70% partition for a valid held-out comparison.

## Upload from PowerShell

```powershell
cd C:\Users\Shricharan\VSCodeProjects\CS4248-Project
scp -J shri@sjump.comp.nus.edu.sg -r scripts combinations_shri.json combinations_reet.json requirements-cluster.txt train.zh-en.zh train.zh-en.en tatoeba.zh wmttest2022.zh shri@xlogin.comp.nus.edu.sg:/home/s/shri/cs4248-mt/
```

## Submit on the cluster

```bash
cd ~/cs4248-mt
sbatch scripts/run_shri_reet.slurm
```

Uses the existing cluster-only environment and model cache, A100 40 GB,
64 GB RAM and a 48-hour allocation. Shri runs first, then Reet, reusing shared
model generations. Do not submit concurrent copies sharing these caches.
The script attempts Reet even if Shri fails and exits nonzero if either fails.
Re-submitting reuses valid generations and completed evaluation checkpoints.
It never evaluates the train or test partitions. COMETKiwi requires Hugging Face access to `Unbabel/wmt22-cometkiwi-da`;
accept its access conditions and authenticate with `hf auth login` if needed.

Results:

```text
results/ensembles/Shri/train702010_seed4248_all/scores_comparison.csv
results/ensembles/Reet/train702010_seed4248_all/scores_comparison.csv
```

Check progress with `squeue -u "$USER"` and
`tail -f slurm-shri-reet-JOBID.log` (replace JOBID).

Choose the primary dev metric before reviewing results (for example COMET-22,
with BLEU and chrF++ as supporting metrics). These development commands are for
configuration comparison; test evaluation is described below. The old full-suite job references superseded
combination files; use the new job for these assignments.

## Each teammate's own job

After the changes are pushed, update the cluster checkout with `git pull --ff-only`.
Submit exactly the command for your name from the repository root:

```bash
sbatch scripts/run_team.slurm "Ming Zhe"
sbatch scripts/run_team.slurm Theodore
sbatch scripts/run_team.slurm Reet
sbatch scripts/run_team.slurm Shri
```

Each command runs only that person's ten assigned combinations. Separate users
can run in their own checkouts and runtime directories. Jobs sharing the same
runtime are protected by a nonblocking lock: a second job exits with a retry
message. To run several owners sequentially in one allocation, pass their names
as separate arguments, e.g. `sbatch scripts/run_team.slurm Shri Reet`.
Logs from the generic job use `slurm-team-JOBID.log`. Each owner's results are
under `results/ensembles/OWNER/train702010_seed4248_all/`.

## Local CUDA computer

Use an activated Python environment with the project's inference dependencies
and a working CUDA PyTorch installation. Set `CS4248_MODEL_CACHE` to a writable
folder (the repository default is `E:\huggingface`). In Windows PowerShell:

```powershell
$env:CS4248_MODEL_CACHE = "$HOME\huggingface"
python scripts/build_train_splits.py
python scripts/run_ensembles.py --owner Shri --combinations combinations_shri.json --data-dir data/splits_70_20_10 --dev dev_train20 --selector all --comet --bertscore --run-name train702010_seed4248_all
```

Replace the owner/file with Reet / combinations_reet.json,
Theodore / combinations_theodore.json, or "Ming Zhe" / combinations_ming_zhe.json.
BERTScore uses CPU by default; append `--bertscore-device cuda` if GPU memory
permits. Translation, COMETKiwi and COMET-22 require CUDA in this runner.

## Test evaluation for all members

Keep the final ensemble fixed using development results. The test suite evaluates
all assigned combinations for comparison. Choosing a new winner from these
scores makes the evaluated data an additional validation set, and that winner
would require a separate untouched test set for an unbiased final estimate.

The official suite reads the repository's `wmttest2022.zh`,
`wmttest2022.AnnA.en`, `tatoeba.zh` and `tatoeba.en`. The English files are the
provided references, including Annotator A's references for WMT22.
Every source/reference pair must have matching nonzero line counts and no blank
sentences. The runner checks all inputs before loading any models. It creates
derived copies under `data/test_suites/` and records their paths and hashes.

From the repository root in a local CUDA Python environment:

```bash
python scripts/run_test_suite.py --dry-run
python scripts/run_test_suite.py --bertscore-device cuda
```

The first command only checks inputs and prints the planned commands. The second
runs all four members sequentially, each with ten combinations and all three
selectors, on both official sets: 240 evaluation rows. BLEU, chrF++, COMET-22
and BERTScore P/R/F1 are reported. Omit `--bertscore-device cuda` to compute
BERTScore on CPU. Translation and COMET still use CUDA. Use the model-cache
environment setting described above if the default location is unsuitable.

To evaluate just one member, append `--owners "Ming Zhe"`, `--owners Theodore`,
`--owners Reet` or `--owners Shri`. Existing valid model generations and
per-experiment checkpoints are reused. Run one suite at a time in a checkout;
avoid running it alongside the old development runner, which shares generation
and scoring caches. A file lock prevents two copies of the new test runner from
using the same checkout concurrently.

For the internal 10% split, or all three test datasets:

```bash
python scripts/build_train_splits.py
python scripts/run_test_suite.py --test-set internal --bertscore-device cuda
python scripts/run_test_suite.py --test-set both --bertscore-device cuda
```

`internal` produces 120 rows and `both` produces 360 rows for all four members.
Choose one of these commands according to the intended evaluation scope.
Each dataset has distinct filenames and results are kept separate from dev runs.

On the cluster, use:

```bash
sbatch scripts/run_team.slurm --test-set official
# Optional subset of members:
sbatch scripts/run_team.slurm --test-set official Shri Reet
```

Replace `official` with `internal` or `both` if needed. This uses the existing
GPU environment and locking setup. The original command with only member names
continues to run the development experiments.

For the default official suite, outputs are:

```text
results/ensembles/OWNER/test_official_all/scores_comparison.csv
results/ensembles/test_suites/official/scores_comparison.csv
results/ensembles/test_suites/official/manifest.json
```

The combined CSV includes an `owner` column and retains the existing `dev` column
name for compatibility; its values are `test_wmt22` and `test_tatoeba`.
The manifest lists the requested members, exact model combinations, dataset
hashes, expected row count and completion status. A combined table is published
only after every requested member completes and all expected rows are verified.
If any member fails, the suite exits nonzero and preserves completed individual
checkpoints. Rerun the same command to finish and assemble the complete table.
Test outputs are recorded in experiment order without automatically selecting a
new final configuration.
