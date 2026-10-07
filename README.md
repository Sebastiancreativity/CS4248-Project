# official/ — 正式结果、代码与过程记录

## 原始数据（Canvas，保持原样，不修改）

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `train.zh-en.{zh,en}` | 15,406 | 课程训练数据（与 HF 上 ALMA-Human-Parallel zh-en 一致，仅 16 行尾部空格不同） |
| `wmttest2022.zh` / `wmttest2022.AnnA.en` | 1,875 | WMT22 测试集，Annotator A 参考（与 sacrebleu 版仅 4 行多一个空格） |
| `tatoeba.zh` | 469 | Tatoeba 测试集，**只有中文，无英文参考**；全部为简体 |

## 开发集（用于确定 MBR 候选池；测试集不参与任何选择）

由 `scripts/build_dev_sets.py` 生成，随机种子 4248，输出在 `data/dev/`，统计见 `data/dev/stats.json`。
所有开发句都与两个官方测试集和训练集做过去重（按简体、去空白后精确匹配）。

| 开发集 | 句数 | 来源 | 对标 | 平均字数 | ≤15 字占比 |
| --- | --- | --- | --- | --- | --- |
| `dev_wmt` | 1,033 | WMT23 zh-en general test：news 527 + user_review 506，按整篇文档抽样（119 篇）；去掉 manuals | WMT22 测试（39.2 字，23%） | 41.5 | 22% |
| `dev_short` | 992 | Tatoeba cmn-eng（MTEB 镜像），繁→简；去掉与官方测试重合 7 句、重复 1 句 | Tatoeba 测试（8.8 字，93%） | 11.0 | 87% |
| `dev_alma` | 1,002 | ALMA-Human-Parallel zh-en validation（新闻） | LoRA 早停用，不用于候选筛选 | 47.9 | 3% |

每个开发集有 `.zh`、`.en`、`.meta.jsonl`（id、来源、领域、docid、是否做过繁简转换）。

## 模型筛选

`scripts/screen_model.py --model <名字>`：统一提示词、贪心解码、统一清洗，在 `dev_wmt` 和 `dev_short` 上生成并评测。
输出：`results/screening/<模型>/<开发集>.hyp`（清洗后）、`.raw.jsonl`（原始输出），汇总追加到 `results/screening/screening.jsonl`；日志在 `logs/`。
每行结果记录了当时显卡的功率上限，`sent_per_s` 只在功率上限相同时可比。
修改清洗规则后用 `--rescore` 重新清洗已保存的原始输出并重新评分，不重新生成；`screening.jsonl` 中同一模型、同一开发集以最新一行为准。

| 模型 | 开发集 | BLEU | chrF++ | COMET | 夹中文 | 空输出 | 复读 | 句/秒（功率上限） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | dev_wmt | 15.64 | 42.06 | 77.81 | 0.77% | 0 | 0.19% | 0.92（30 W） |
| Qwen2.5-1.5B-Instruct | dev_short | 37.81 | 57.67 | 90.12 | 0.10% | 0 | 0 | 6.3（30 W） |

## 过程记录

- 2026-10-07：建立三个开发集（见上）。
- 2026-10-07：Qwen2.5-1.5B 开始筛选。显卡当时被软件限功到 30 W（默认 80 W），速度约为正常的 1/10，质量指标不受影响。
- 2026-10-07：清洗规则补上 "Here's the English translation:" 一类前缀，Qwen2.5-1.5B 重新评分；`scripts/breakdown.py` 给出分领域、分句长结果。报告见 `report/screening_qwen2.5-1.5b_{zh,en}.md`。
