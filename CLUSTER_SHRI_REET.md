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
with BLEU and chrF++ as supporting metrics). No automatic winner selection or
test evaluation is included. The old full-suite job references superseded
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
