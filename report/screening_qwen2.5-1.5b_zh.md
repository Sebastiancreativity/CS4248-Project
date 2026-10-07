# 候选模型筛选报告 · Qwen2.5-1.5B-Instruct

**日期**：2026-10-07 · **阶段**：MBR 候选池筛选（第一个模型，作为后续模型的参照线）

## 1. 结论

Qwen2.5-1.5B-Instruct 在 WMT22 风格的开发集上得到 **15.64 BLEU / 77.81 COMET**，在短句开发集上得到 **37.81 BLEU / 90.12 COMET**；输出卫生良好（夹中文 0.77%，无空输出），显存峰值仅 3.15 GB。

它适合作为**参照线和低成本候选**，但不足以单独撑起候选池：主要短板是**专有名词**（"范明政"相关的 22 句中只有 6 句译对人名）和**口语化的用户评论**（BLEU 9.96，比新闻低 9.4）。后续模型应重点比较这两处。

## 2. 实验设置

| 项目 | 设置 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct`，bf16，单卡 RTX 5070 Ti Laptop 12 GB |
| 提示词 | 对话模板，用户消息：`Translate the following Chinese text into English. Output only the English translation.` + `Chinese: {src}` |
| 解码 | 贪心（num_beams=1），max_new_tokens=256，batch 16 |
| 输出清洗 | 去掉 "Translation:" / "Here's the English translation:" 等前缀行，取第一行非空内容；源句不以引号开头时去掉包裹译文的引号 |
| 评测 | sacreBLEU（tokenize 13a）、chrF++、COMET-22（`Unbabel/wmt22-comet-da`） |
| 开发集 | `dev_wmt` 1,033 句（WMT23 zh-en 的 news 527 + user_review 506）；`dev_short` 992 句（Tatoeba cmn-eng，已转简体）。均与官方测试集和训练集去重 |

测试集（WMT22、Tatoeba）没有参与本次任何计算。

## 3. 总体结果

| 开发集 | 句数 | BLEU | chrF++ | COMET | 夹中文 | 空输出 | 复读 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev_wmt | 1,033 | 15.64 | 42.06 | 77.81 | 0.77%（8 句） | 0 | 0.19%（2 句） |
| dev_short | 992 | 37.81 | 57.67 | 90.12 | 0.10%（1 句） | 0 | 0 |

**与此前 WMT22 基线一致**：同一模型此前在 WMT22 测试集上是 16.83 BLEU / 77.87 COMET（另一提示词）。dev_wmt 的分数与之接近，说明 dev_wmt 的难度与 WMT22 相当，可作为筛选依据。

## 4. 分领域与分句长

| 开发集 | 分组 | 句数 | BLEU | chrF++ | COMET |
| --- | --- | --- | --- | --- | --- |
| dev_wmt | news | 527 | 19.36 | 46.20 | 79.60 |
| dev_wmt | user_review | 506 | 9.96 | 35.62 | 75.94 |
| dev_wmt | ≤15 字 | 227 | 11.33 | 31.23 | 75.79 |
| dev_wmt | 16–40 字 | 364 | 15.11 | 39.43 | 79.18 |
| dev_wmt | 41+ 字 | 442 | 16.06 | 43.71 | 77.72 |
| dev_short | ≤15 字 | 865 | 40.43 | 58.84 | 90.40 |
| dev_short | 16–40 字 | 123 | 30.92 | 54.85 | 88.48 |

- **用户评论最难**：比新闻低 9.4 BLEU、3.7 COMET。它对应 WMT22 中的电商和社交领域（共占 WMT22 的 54%），是候选模型最该比较的地方。
- **dev_wmt 中的短句反而最差**（75.79 COMET）：这些多是评论里的口语短句（"就酱紫""奸商啊"），而不是 Tatoeba 那种规范短句。

## 5. 错误分析

**专有名词。** dev_wmt 中有 22 句提到越南总理"范明政"。模型输出的人名：Feng Mingzheng 4 次、Fan Mingzheng 2 次、Nguyen Xuan Phuc 2 次、Nguyen Xuan Thanh / Pham Gia Khiem / Wen Jiabao 各 1 次，8 句没有给出人名，只有 6 句写对（Pham/Phan Minh Chinh）。错法有两种：按拼音音译，或换成另一个真实的越南领导人。这正是此前诊断出的"实体错误"，需要更大的模型或翻译专用模型来补。

**口语与习语。** 最低分的句子集中在口语和习语：

| 源句 | 输出 | 参考 |
| --- | --- | --- |
| 我的没有你的好。 | My lack of you is not as good. | Mine is not so good as yours. |
| 左看右看 | Left and right look at | Look to your left, look to your right. |
| 强宾不压主。 | Power does not subdue virtue. | A guest should not try to make himself superior to the host. |
| 撇开网文这个文化现象不谈，全职不失为一本不错的书。 | Cutting to the chase is not bad as a book either. | Full-time is still a terrific book even when … internet writing is disregarded. |

**夹中文。** 8 句把个别中文词直接留在英文里，例如 "above阿拉斯加"、"keep your初心"、"High Sky轩 said"。这类错误规则可以检出，MBR 前的过滤需要保留。

**格式。** 3 句模型先回一行 "Translation:" 或 "Here's the English translation:" 再给译文，原清洗规则会留下空行或这行字。已修正清洗规则，并用 `--rescore` 重新评分（本报告数字均为修正后）。

## 6. 开发集本身的问题

WMT23 的部分参考译文质量偏低，会压低 BLEU，对最低分句子的判断也有干扰：

- "范明政"的 21 处参考译名中，12 处写成了 **Pham Minh Triet**（另一位越南领导人），只有 9 处是正确的 Phan Minh Chinh。
- "就酱紫"（"就这样子"）的参考是 "The sauce alone."；"应对新冠疫情冲击"的参考里有 "unused crown plague"。

因此在 dev_wmt 上比较模型时，**以 COMET 为主、BLEU 为辅**，且只看模型之间的相对差距。

## 7. 对候选池的意义

1. **参照线**：后续模型在 dev_wmt 上至少要超过 77.81 COMET，或在 user_review 上明显更好，才有入池价值。
2. **看点**：专有名词（人名、机构名）和用户评论领域，这两处是 1.5B 的明显短板，也是异质候选最可能互补的地方。
3. **成本**：显存 3.15 GB，可以和一个 4-bit 7B 模型同时放进 12 GB，适合作为 MBR 中的低成本候选。
4. **速度暂不可比**：本次显卡被软件限功到 30 W（默认 80 W），dev_wmt 只有 0.92 句/秒、dev_short 6.3 句/秒。恢复功率后需重测。

## 8. 复现

```bash
python official/scripts/build_dev_sets.py
python official/scripts/screen_model.py --model qwen2.5-1.5b
python official/scripts/screen_model.py --model qwen2.5-1.5b --rescore
python official/scripts/breakdown.py --model qwen2.5-1.5b
```

结果文件：`official/results/screening/screening.jsonl`（汇总）、`official/results/screening/qwen2.5-1.5b/`（译文、原始输出、逐句 COMET、`breakdown.json`）。
