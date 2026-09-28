# Selective Multi-Pass Decoding for NUCLE GEC

This folder is a self-contained extension of the existing T5 grammatical error
correction project. It asks whether a small T5 model benefits from correcting
its own output, and whether a lightweight selector can avoid harmful second
passes.

The experiment uses only the existing, grouped NUCLE split and trained model:

- training data/model: `../pipeline_runs/nucle-full-grouped/model`
- selector training data: `../pipeline_runs/nucle-full-grouped/data/dev.*`
- final evaluation data: `../pipeline_runs/nucle-full-grouped/data/test.*`

The test references are used only for scoring and for an explicitly labelled
oracle upper bound. They are never used to fit the selector.

## Systems compared

1. `pass1`: normal one-pass T5 correction.
2. `pass2`: apply the same model to the first correction.
3. `pass3`: apply it once more.
4. `stable`: stop at the first repeated output, up to three passes.
5. `selective`: use a logistic selector trained on development data to choose
   pass 1 or pass 2.
6. `distance_oracle`: choose whichever of pass 1/pass 2 has lower token-level
   edit distance to the gold correction. This is diagnostic only, not an
   ERRANT-F0.5 upper bound or a deployable system.

The selector uses generation-confidence differences, token edit distances,
length changes, copying rates, and whether the two passes are already stable.

## Run

From the parent `Code` directory, using its existing virtual environment:

```powershell
.\.venv\Scripts\python.exe selective_mpd\run_experiment.py --device cuda --fp16
```

`--fp16` affects inference only when CUDA is selected. It can be omitted.
Outputs are written to `selective_mpd/outputs` by default. Existing outputs are
protected; use `--overwrite` only when deliberately repeating a run.

For a quick smoke test before the full experiment:

```powershell
.\.venv\Scripts\python.exe selective_mpd\run_experiment.py --device cuda --limit 20 --output-dir selective_mpd\outputs-smoke
```

The limited run is for debugging only and should not be reported as a result.

## Main outputs

- `dev/candidates.jsonl`, `test/candidates.jsonl`: all passes and features.
- `selector.json`: fitted selector parameters and development diagnostics.
- `predictions/*.txt`: one prediction per system.
- `scores/*.score.txt`: ERRANT precision, recall, and F0.5.
- `summary.json` and `summary.md`: consolidated results and pass analysis.

## Suggested report framing

The main comparison is `pass1` versus `selective`. Fixed pass 2 and pass 3
show whether naive iteration helps, while `distance_oracle` measures whether
second-pass cases can become textually closer to the reference. Report average
passes and inference time alongside F0.5.

This is a NUCLE-only development experiment. The course's final evaluation
still requires CoNLL-2014 and BEA-2019 when those datasets become available.

## Paper being adapted

Wang et al. (2024), *Multi-pass Decoding for Grammatical Error Correction*,
EMNLP 2024: <https://aclanthology.org/2024.emnlp-main.553/>

The paper trains a neural early-stop classifier from decoder hidden states and
uses source-information fusion. This project instead evaluates a lightweight,
model-agnostic selector based on observable confidence and edit features.

