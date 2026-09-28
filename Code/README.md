# NUCLE grammatical error correction with T5

This project can score the plain `t5-small` model as a baseline, then fine-tune
the same model and score it on the **same held-out NUCLE sentences**. The
pipeline uses ERRANT to report precision, recall and F0.5. This is an internal
NUCLE split, not an official CoNLL or BEA test score.

Run all commands below in PowerShell from
`C:\Users\Ming Zhe\Desktop\CS4248 Project\Code`.

## One-time setup

Use this project's `.venv` Python for every command. Plain `python` may point
to another installation when the environment is not activated.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-scoring.txt
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

On this machine, `.venv` already has a CUDA-enabled PyTorch build. If setting
up elsewhere, select the appropriate PyTorch install command at
<https://pytorch.org/get-started/locally/>. Check this environment:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`True` means `--device cuda` is available. Scoring itself runs on the CPU.
The first model run may download `t5-small` from Hugging Face.

## 1. Score plain T5 without training

```powershell
.\.venv\Scripts\python.exe pipeline.py baseline --device cuda
```

This command makes a reproducible split of the existing
`data\nucle-error-only\nucle.train.gold.bea19.src` and `.tgt` files:
20,953 train, 200 dev and
200 held-out test pairs by default. It runs plain `t5-small` on only those
200 test inputs and scores its output with ERRANT. The score is saved to
`pipeline_runs\nucle\scores\baseline.score.txt`.

Plain T5 has not been trained specifically for grammar correction, so a low
baseline score is expected. The pipeline uses the `grammar: ` prefix for both
the baseline and the fine-tuned model.

## 2. Train and score the fine-tuned model

After the baseline, run:

```powershell
.\.venv\Scripts\python.exe pipeline.py train-score --device cuda --fp16
```

This trains on 1,000 examples from the **train split** by default, uses the
200 dev examples for training-time evaluation, then predicts and scores the
same 200 held-out test examples used for the baseline. The score is saved to
`pipeline_runs\nucle\scores\trained.score.txt`. The model is saved in
`pipeline_runs\nucle\model`.

For training with every available training pair, add `--train-subset-size 0`.
This takes much longer. If CUDA runs out of memory, add
`--train-batch-size 2 --gradient-accumulation-steps 4`.

For a fresh run that does both stages in one command, use:

```powershell
.\.venv\Scripts\python.exe pipeline.py all --device cuda --fp16
```

Use `all` before running `baseline` or `train-score` in that work directory.
The pipeline protects existing model and prediction files. Pass `--overwrite`
if you deliberately want to replace them, or choose a new `--work-dir`.

## Full NUCLE dataset, including unchanged sentences

The `data\nucle-error-only\nucle.train.gold.bea19.src/.tgt` pair excludes unchanged
sentences. A separate full pair has been generated in `data\nucle-full` from
`..\gec\nucle\bea2019\nucle.train.gold.bea19.m2` **without**
`--erroneous_only`: 57,151 aligned pairs, including 35,798 unchanged ones.
The error-only files are kept in their own folder, and the old experiment is preserved.

Its recommended pipeline folder is `pipeline_runs\nucle-full-grouped`. The
`--group-by-source` setting keeps identical source sentences in the same
split, so none appear in both training and test. Its split and plain-T5
baseline have already been prepared. To train on **all 56,751
training pairs** for three epochs and score on that run's 200 test pairs:

```powershell
.\.venv\Scripts\python.exe pipeline.py train-score --full-nucle --train-subset-size 0 --epochs 3 --device cuda --fp16
```

The pipeline automatically writes that run's model, predictions and scores
inside `pipeline_runs\nucle-full-grouped`. Its test set contains unchanged sentences,
so compare its trained score with its own baseline score in
`pipeline_runs\nucle-full-grouped\scores\baseline.score.txt`, not the old
error-only run's score.

An earlier exploratory folder, `pipeline_runs\nucle-full`, was created before
checking for repeated source sentences across splits. Keep it for reference;
use `nucle-full-grouped` for this experiment.

For a new dataset path, the converter now accepts explicit output locations.
This command describes how the full pair above was made; it refuses to
overwrite those files if they already exist:

```powershell
.\.venv\Scripts\python.exe ..\gec\m2_to_parallel.py --data ..\gec\nucle\bea2019\nucle.train.gold.bea19.m2 --src-output data\nucle-full\nucle.train.gold.bea19.src --tgt-output data\nucle-full\nucle.train.gold.bea19.tgt
```

## Useful options and outputs

| Option | Default | Purpose |
| --- | --- | --- |
| `--test-size` | `200` | Number of held-out examples to predict and score |
| `--dev-size` | `200` | Number of development examples |
| `--train-subset-size` | `1000` | Number of train examples used; `0` uses all train pairs |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--fp16` | Off | Half precision for CUDA training; optional |
| `--epochs` | `1` | Number of training epochs |
| `--train-batch-size` | `8` | Training examples per batch |
| `--infer-batch-size` | `16` | Sentences per inference batch |
| `--seed` | `42` | Keeps the split reproducible |
| `--work-dir` | `pipeline_runs\nucle` | Where split, predictions, model and scores go |
| `--source`, `--target` | Error-only NUCLE files in `data\nucle-error-only` | Matching parallel files |
| `--group-by-source` | Off | Keep identical source sentences in the same split |
| `--full-nucle` | Off | Select the full-data pair, grouped split and `nucle-full-grouped` run folder |

To try a different test size after the default split already exists, give it
a different work directory, for example:

```powershell
.\.venv\Scripts\python.exe pipeline.py baseline --test-size 50 --work-dir pipeline_runs\nucle-50 --device cuda
```

The split files are in `pipeline_runs\nucle\data`. The `split.json` file
records the original data hashes, split sizes and seed. Predictions go in
`pipeline_runs\nucle\predictions`, and ERRANT annotations and text scores go
in `pipeline_runs\nucle\scores`.

## Run the individual scripts

`train_gec.py` and `run_gec.py` still work on their own. See their complete
argument lists with:

```powershell
.\.venv\Scripts\python.exe train_gec.py --help
.\.venv\Scripts\python.exe run_gec.py --help
.\.venv\Scripts\python.exe score_gec.py --help
```

For an existing prediction file, `score_gec.py` scores it against matching
source and target files:

```powershell
.\.venv\Scripts\python.exe score_gec.py --source pipeline_runs\nucle\data\test.src --reference pipeline_runs\nucle\data\test.tgt --predictions pipeline_runs\nucle\predictions\baseline.txt --output-prefix pipeline_runs\nucle\scores\baseline
```

All three input files must have the same number of lines in the same order.
The scorer tokenizes both the reference corrections and model predictions
consistently before comparing edits. This matters for NUCLE: its parallel
files put spaces around punctuation, while T5 often does not. Use
`--pretokenized` only if all three input files already have identical
tokenization. Previous scores made without this normalization are preserved
under `scores\legacy-pretokenized` in the `nucle` and `nucle-full-grouped`
run folders; do not compare those directly with the current scores.
Do not pass the original
`..\gec\nucle\bea2019\nucle.train.gold.bea19.m2` directly with the prepared
NUCLE `.src`: that M2 has 57,151 sentences while the prepared `.src` has
21,353. The original NUS M2 scorer also requires Python 2; this pipeline
uses the Python 3 compatible [ERRANT scorer](https://github.com/chrisjbryant/errant).
