# Candidate Screening Report · Qwen2.5-1.5B-Instruct

**Date**: 2026-10-07 · **Stage**: MBR candidate-pool screening (first model, the reference line for later models)

## 1. Summary

Qwen2.5-1.5B-Instruct scores **15.64 BLEU / 77.81 COMET** on the WMT22-like dev set and **37.81 BLEU / 90.12 COMET** on the short-sentence dev set. Output hygiene is good (0.77% of outputs contain Chinese, none are empty) and peak GPU memory is only 3.15 GB.

It suits the pool as a **reference line and a low-cost candidate**, but cannot carry the pool alone. Its weak spots are **proper names** (only 6 of the 22 sentences about "范明政" get the name right) and **colloquial user reviews** (9.96 BLEU, 9.4 below news). Later models should be compared on exactly these two points.

## 2. Setup

| Item | Setting |
| --- | --- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, bf16, one RTX 5070 Ti Laptop GPU (12 GB) |
| Prompt | Chat template; user message: `Translate the following Chinese text into English. Output only the English translation.` + `Chinese: {src}` |
| Decoding | Greedy (num_beams=1), max_new_tokens=256, batch 16 |
| Output cleaning | Drop prefix lines such as "Translation:" or "Here's the English translation:", keep the first non-empty line; strip wrapping quotes unless the source itself starts with a quote |
| Metrics | sacreBLEU (tokenize 13a), chrF++, COMET-22 (`Unbabel/wmt22-comet-da`) |
| Dev sets | `dev_wmt` 1,033 sentences (WMT23 zh-en, news 527 + user_review 506); `dev_short` 992 sentences (Tatoeba cmn-eng, converted to simplified). Both deduplicated against the official test sets and the training data |

The test sets (WMT22, Tatoeba) were not used in any computation here.

## 3. Overall results

| Dev set | Sentences | BLEU | chrF++ | COMET | Chinese in output | Empty | Repetition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev_wmt | 1,033 | 15.64 | 42.06 | 77.81 | 0.77% (8) | 0 | 0.19% (2) |
| dev_short | 992 | 37.81 | 57.67 | 90.12 | 0.10% (1) | 0 | 0 |

**Consistent with the earlier WMT22 baseline**: the same model previously scored 16.83 BLEU / 77.87 COMET on the WMT22 test set (with a different prompt). dev_wmt lands close to that, so its difficulty matches WMT22 and it is a sound basis for screening.

## 4. By domain and length

| Dev set | Group | Sentences | BLEU | chrF++ | COMET |
| --- | --- | --- | --- | --- | --- |
| dev_wmt | news | 527 | 19.36 | 46.20 | 79.60 |
| dev_wmt | user_review | 506 | 9.96 | 35.62 | 75.94 |
| dev_wmt | ≤15 chars | 227 | 11.33 | 31.23 | 75.79 |
| dev_wmt | 16–40 chars | 364 | 15.11 | 39.43 | 79.18 |
| dev_wmt | 41+ chars | 442 | 16.06 | 43.71 | 77.72 |
| dev_short | ≤15 chars | 865 | 40.43 | 58.84 | 90.40 |
| dev_short | 16–40 chars | 123 | 30.92 | 54.85 | 88.48 |

- **User reviews are hardest**: 9.4 BLEU and 3.7 COMET below news. They stand in for WMT22's e-commerce and social domains (54% of WMT22), so this is where candidates matter most.
- **Short sentences in dev_wmt score worst** (75.79 COMET): they are mostly colloquial review fragments ("就酱紫", "奸商啊"), unlike Tatoeba's well-formed short sentences.

## 5. Error analysis

**Proper names.** 22 dev_wmt sentences mention Vietnam's Prime Minister "范明政". The model wrote Feng Mingzheng 4 times, Fan Mingzheng 2, Nguyen Xuan Phuc 2, and Nguyen Xuan Thanh, Pham Gia Khiem and Wen Jiabao once each; 8 sentences gave no name and only 6 were correct (Pham/Phan Minh Chinh). It either transliterates from pinyin or substitutes a different real Vietnamese leader. This is the entity error diagnosed earlier; larger or translation-tuned models are needed to cover it.

**Colloquial language and idioms.** The lowest-scoring sentences cluster here:

| Source | Output | Reference |
| --- | --- | --- |
| 我的没有你的好。 | My lack of you is not as good. | Mine is not so good as yours. |
| 左看右看 | Left and right look at | Look to your left, look to your right. |
| 强宾不压主。 | Power does not subdue virtue. | A guest should not try to make himself superior to the host. |
| 撇开网文这个文化现象不谈，全职不失为一本不错的书。 | Cutting to the chase is not bad as a book either. | Full-time is still a terrific book even when … internet writing is disregarded. |

**Chinese left in output.** 8 outputs keep a Chinese word inside the English, e.g. "above阿拉斯加", "keep your初心", "High Sky轩 said". Rules catch these, so the pre-MBR filter stays.

**Formatting.** In 3 outputs the model first wrote a line "Translation:" or "Here's the English translation:" and then the translation; the original cleaning kept the label or an empty line. The cleaning rule is fixed and the outputs were re-scored with `--rescore`; all numbers here are after the fix.

## 6. Problems in the dev set itself

Some WMT23 references are poor, which depresses BLEU and muddies the lowest-scoring cases:

- Of 21 reference renderings of "范明政", 12 say **Pham Minh Triet** (a different Vietnamese leader); only 9 give the correct Phan Minh Chinh.
- "就酱紫" ("that's how it is") is referenced as "The sauce alone."; the reference for "应对新冠疫情冲击" contains "unused crown plague".

So on dev_wmt, compare models **mainly by COMET, with BLEU as secondary**, and only by relative differences between models.

## 7. What this means for the pool

1. **Reference line**: a later model needs more than 77.81 COMET on dev_wmt, or a clear gain on user_review, to earn a place.
2. **Where to look**: proper names and the user-review domain, the 1.5B model's clearest gaps and where heterogeneous candidates are most likely to complement it.
3. **Cost**: 3.15 GB of GPU memory, so it fits in 12 GB alongside one 4-bit 7B model; a good low-cost MBR candidate.
4. **Speed not yet comparable**: the GPU was software-capped at 30 W (default 80 W) during this run, giving 0.92 sentences/s on dev_wmt and 6.3 on dev_short. Re-measure once power is restored.

## 8. Reproduce

```bash
python official/scripts/build_dev_sets.py
python official/scripts/screen_model.py --model qwen2.5-1.5b
python official/scripts/screen_model.py --model qwen2.5-1.5b --rescore
python official/scripts/breakdown.py --model qwen2.5-1.5b
```

Result files: `official/results/screening/screening.jsonl` (summary) and `official/results/screening/qwen2.5-1.5b/` (translations, raw outputs, sentence-level COMET, `breakdown.json`).
