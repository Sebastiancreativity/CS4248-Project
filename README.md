# Chinese-to-English MT: Ensemble Selection with an Encoder–Decoder Auditing Framework

CS4248 (AY2026/27 Sem 1), Group G20. Task: Chinese → English translation, evaluated on WMT22 general zh–en (reference A) and Tatoeba with SacreBLEU (13a) and COMET-22 (`Unbabel/wmt22-comet-da`).

The system has two parts that communicate only through text files:

1. **Candidate generation** (team). Six open LLMs are adapted with QLoRA and combined in three-model ensembles. The final ensemble is fine-tuned Hunyuan-MT-7B, Qwen2.5-7B-Instruct and TowerInstruct-Mistral-7B-v0.2.
2. **Selection** (this repository). One candidate is chosen per sentence without the reference. We compare multi-model voting (chrF++ consensus), COMETKiwi quality estimation and our **auditing framework**. The framework adds a rule guard and forward/reverse likelihood scores from small encoder–decoder models (OPUS-MT, NLLB) to consensus. It reads only text, so it works with candidates from any model.

**Final submission:** `results/submission/final_cometkiwi/wmt22.en` and `tatoeba.en` (COMETKiwi selection over the fine-tuned ensemble).

## Data

| Use | Data |
| --- | --- |
| Official test sets | `data/test/wmttest2022.zh`, `wmttest2022.AnnA.en` (1,875), `tatoeba.zh` (469, no reference) |
| Training corpus | `data/train/train.zh-en.{zh,en}` (15,406 pairs, course-provided) |
| Ensemble development / internal test | `data/dev/dev_train20` (3,068) and `data/test/test_train10` (1,535): seed-4248 70/20/10 split of the revised corpus |
| Framework development | `data/dev/dev_wmt` (1,033, WMT23 news + user review), `dev_alma` (1,002, ALMA validation), `dev_short` (992, Tatoeba) |

The two parts use different data. **Time was short for the candidate pool**, so the ensemble was trained and selected **only on the course-provided corpus**. To make selection as reliable as possible, **the framework uses all available data**: its own development sets (WMT23, the ALMA validation split and Tatoeba sentences) as well as `dev_train20`. No development sentence overlaps the test sets or the training corpus (checked after script conversion and whitespace removal). Test references are used only for scoring, never for tuning or selection.

## Repository layout

```
data/            dev/, test/, train/
framework/       selection framework (python package) and configs/ for every experiment
scripts/
  1_data/        build_dev_sets.py, rebuild_split.py, audit_train_alignment.py
  2_candidates/  sample_pool.py (+ llm.py), nmt_pool.py, finetuned_trio.py
  3_selection/   cometkiwi_select.py, run_framework.sh
  4_evaluation/  score_test_sets.py
results/
  candidates/         all candidate translations used in the paper
  framework/          development experiments (metrics.json, selected outputs, decision records)
  ensemble_selectors/ team selector outputs (chrF++, BERTScore) and COMETKiwi picks
  cometkiwi_scores/   sentence-level COMETKiwi scores
  submission/         test outputs: final_cometkiwi, ensemble_framework, ensemble_consensus, qwen3b_selfpool_test
  evaluation/         WMT22 and internal-test tables with bootstrap results
  train_audit/        misaligned training pairs found by the auditors
report/paper/    main_v5_full.tex (full paper), main_v5_4page.tex (4-page version), custom_additions.bib
```

## Reproducing

Run every command from this folder. Requirements: `pip install -r requirements.txt`; a CUDA GPU is needed for generation and COMET.

```bash
python scripts/1_data/build_dev_sets.py
python scripts/1_data/rebuild_split.py
python scripts/1_data/audit_train_alignment.py --gpu
python scripts/2_candidates/sample_pool.py --model qwen2.5-1.5b --src data/dev/dev_wmt.zh --name dev_wmt --n 8
python scripts/2_candidates/nmt_pool.py --system nllb-600m --dev dev_wmt dev_alma
python scripts/2_candidates/finetuned_trio.py --team-dir <ensemble repository>
python scripts/3_selection/cometkiwi_select.py
bash scripts/3_selection/run_framework.sh
python scripts/4_evaluation/score_test_sets.py
```

`sample_pool.py` is run once per model and set (Qwen2.5-1.5B: 8 samples; Qwen2.5-3B: 4 samples, including the test sets). `nmt_pool.py` is also run for `opus-mt-zh-en`. `finetuned_trio.py` needs the team's LoRA adapters, prompts and cleaner. The team's `dev_train20` outputs are in `results/candidates/ft.*.dev_train20.txt`. Auditor and COMET scores are cached in `cache/scores.sqlite`, so the selection steps run on CPU.

## Development

| Date | Step |
| --- | --- |
| Early Oct | Proposal: NLLB-1.3B + LoRA, with a coverage penalty in beam search. Changed after we screened the models: the translations of open LLMs are fluent, and their remaining errors are about information. |
| 7–10 Oct | New plan: consensus (MBR) over LLM candidates first, then an encoder–decoder auditor. Built the development sets and screened Qwen2.5. |
| 20 Oct | Framework: rule guard + consensus + forward/reverse likelihood, with every score cached by content. The team took over candidate models. |
| 29 Oct | Mixed pools (Qwen + OPUS-MT + NLLB) showed that auditors favour their own model family; added per-system calibration. Controlled error injection showed which signal detects which error. |
| 8–11 Nov | Team delivered fine-tuned ensembles. The final trio replaced NLLB-1.3B with Qwen2.5-7B. Added system priors, then ran COMETKiwi locally and reproduced the team's selection exactly (3,068/3,068 sentences). |
| 12 Nov | Rebuilt the internal test split and evaluated every selector on it. |

## Results

COMET-22 (×100), with BLEU where useful. Significance uses paired bootstrap with 1,000 resamples.

### 1. The framework improves weak single-model pools

The pools are one greedy translation plus samples from small Qwen models. Results are on the evaluation halves.

| Pool (dev set) | Greedy | Consensus | Framework | Gain over greedy |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B (dev_wmt) | 14.09 / 77.14 | 14.22 / 77.58 | **15.11 / 77.89** | +1.02 BLEU, +0.75 COMET |
| Qwen2.5-1.5B (dev_alma) | 16.00 / 82.96 | 15.92 / 83.08 | **16.60 / 83.52** | +0.60, +0.56 |
| Qwen2.5-3B (dev_wmt) | 17.85 / 77.95 | 17.96 / 78.38 | **18.30 / 78.86** | +0.45, +0.91 |
| Qwen2.5-3B (dev_alma) | 19.16 / 83.78 | 18.74 / 83.91 | **19.66 / 84.19** | +0.50, +0.41 |

- **The framework beats consensus in every case** (+0.28 to +0.48 COMET), winning at least 99.9% of resamples, and it also raises BLEU everywhere.
- **Why it helps:** a weak model's errors are mostly errors of information, and the two likelihood scores catch them. The forward score always prefers the intact translation when words are inserted (100%). The reverse score does so for at least 97.5% of deletions. Both detect a swapped name or a changed number in at least 95% of pairs.
- **It works on the test set too.** On WMT22 with Qwen2.5-3B, the framework raises BLEU from 18.65 to 19.60 and COMET-22 from 80.26 to 80.79 over the greedy output.
- **It removes Chinese from the output.** A guard fallback cuts the number of outputs that still contain Chinese from 44 to 1.

### 2. Multi-model voting

**On a mixed pool, voting beats every member.** The pool has four systems: Qwen2.5-1.5B, Qwen2.5-3B, OPUS-MT and NLLB-600M. Consensus over them reaches 78.82 on dev_wmt and 84.01 on dev_alma. That is +0.87 and +0.23 over the best member (Qwen2.5-3B: 77.95 / 83.78). It is also far above the weak members: OPUS-MT scores 72.06 / 80.56, NLLB-600M 73.25 / 82.00 and Qwen2.5-1.5B 77.14 / 82.96.

**On the fine-tuned ensemble, voting lifts the pool well above its weakest member.**

| System | Internal test BLEU / COMET | WMT22 BLEU / COMET |
| --- | --- | --- |
| Hunyuan-MT-7B alone | **36.81** / 85.26 | 28.40 / 81.91 |
| Qwen2.5-7B alone | 30.02 / 84.26 | 24.71 / 81.32 |
| Tower-Mistral-7B alone | 32.44 / 84.91 | **30.48** / 82.64 |
| chrF++ voting (consensus) | 34.16 / 85.16 | 29.42 / 82.54 |
| Auditing framework | 36.57 / 85.31 | 28.80 / 82.11 |
| **COMETKiwi selection (final)** | 33.72 / **85.44** | 28.44 / **83.01** |

- **Voting versus the weakest member.** Voting is +0.90 (internal) and +1.22 (WMT22) above Qwen2.5-7B alone. On WMT22 it is also above Hunyuan-MT-7B in 100% of resamples.
- **Why voting does not beat the best member.** In each domain one member is much stronger than the others. On the internal test, Hunyuan-MT-7B leads the others by 4–7 BLEU. On WMT22, Tower-Mistral-7B leads. Voting gives every member an equal vote, so it still picks the weaker members' translations often. It ends 0.10 below the best member on both sets, and on WMT22 the gap is not significant (Tower-Mistral is higher in only 85% of resamples).
- **The framework fixes this in-domain.** Its learned system prior gives Hunyuan-MT-7B the default vote. On the internal test this lifts the framework above voting (85.31, ahead in 98.3% of resamples) and recovers almost all of Hunyuan's BLEU (36.57). On WMT22 the same prior chose Hunyuan for 1,727 of 1,875 sentences. There, Tower-Mistral is the strongest model (Hunyuan scores 0.73 COMET lower), so the prior costs 0.43 COMET against voting. Priors learned in one domain do not transfer to another.
- **COMETKiwi is the best selector on every set.** Its COMET-22 is highest on the development half (85.62), the internal test (85.44) and WMT22 (83.01). It beats every single model and every other selector in at least 94.9% of resamples. On WMT22 its BLEU is 2.0 lower than Tower-Mistral alone. COMETKiwi and COMET-22 are trained on similar human judgements, so part of this gain may reflect shared preferences between the two metrics.

### 3. What each signal detects

| Injected error | Forward | Reverse | Coverage |
| --- | --- | --- | --- |
| Delete 30% of words | 96.8–99.0 | 97.5–99.8 | 99.5–100 |
| Insert 5 words | 100 | 96.8–98.6 | 0.2–1.0 |
| Swap a name | 97.9–100 | 95.3–98.4 | 36.9–40.7 |
| Change a number | 98.9–100 | 95.6–100 | 19.8–33.5 |

The table shows how often each signal prefers the intact translation. Attention coverage detects deletions only, so it is used to explain and repair omissions, not to rank candidates.

**Calibration matters in mixed pools.** Without it, the NLLB-1.3B auditor chose same-family NLLB-600M outputs for 60.6% of dev_alma sentences, and COMET-22 fell by 0.72. Per-system calibration removes this loss.

### 4. Training-data audit

The two auditors flagged 60 of 15,406 training lines. They include a run of 24 lines (954–977) whose English is shifted by one row. The team deleted these pairs before training (`results/train_audit/`).
