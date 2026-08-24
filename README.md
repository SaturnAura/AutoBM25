# AutoBM25：基于语料统计特征的 BM25 超参数自适应选择

> **零标注、零调参：只通过分析文档集自身的统计特征，预测最优 BM25 超参数（k1, b, δ, IDF 类型），在 8 个 BEIR 数据集上平均提升 NDCG@10 ≈ 16%，并在同口径查询子集上达到 grid search 上界的 55%–100%（多数数据集 ≥ 75%）。**

## 摘要

BM25 是信息检索中最经典、应用最广的词袋排序模型，但其检索性能对超参数（词频饱和点 k1、长度归一化强度 b、BM25+ 补偿项 δ、IDF 变体）高度敏感：不同数据集的参数最优值差异极大，而实际工程中几乎总是使用固定默认值（k1=1.2, b=0.75），原因是最优参数需要带相关性标注的验证集做网格搜索，新数据集往往没有标注。

本项目研究一个更根本的问题：**能否在不使用任何相关性标注的前提下，仅通过分析语料本身的统计特征（文档长度分布、词频冗余度、词汇增长曲线等）直接预测该数据集最合适的 BM25 超参数？** 我们设计了 16 维统计特征体系（P0/P1/P2 分层），以启发式规则将特征映射到参数，并用数据增强与 grid search 标定规则系数。在 8 个覆盖金融、医学、科学文献、社区问答、论点检索等领域的 BEIR 数据集上，零标注规则预测在 7/8 数据集上优于固定默认参数，且在同口径查询子集上达到 grid search 上界的 55%–100%（多数数据集 ≥ 75%）。

---

## 1. 研究背景与动机

### 1.1 BM25 与默认参数的"普适性假设"

BM25 对查询 q 在文档 D 上的打分形式为：

```
score(D, Q) = Σ_{q∈Q} IDF(q) · f(q,D)·(k1+1) / (f(q,D) + k1·(1 − b + b·|D|/avgdl))
```

其中 `f(q,D)` 是查询词在文档中的词频，`avgdl` 是语料平均文档长度。k1 控制词频的饱和速度（k1 越大，词频带来的收益越不饱和）；b 控制文档长度归一化的强度（b=0 完全不归一化，长文档天然占优；b=1 完全归一化）。BM25+ [3] 在此基础上为每个匹配词增加常数补偿项 δ，用于缓解短文档被长度归一化过度惩罚的问题。IDF 亦有多种变体，最常见的是 RSJ 形式 `ln((N−df+0.5)/(df+0.5))` 与平滑形式 `ln((N+1)/(df+1))`，后者对稀有词的惩罚更温和。

经验上，k1=1.2、b=0.75 来自早期 TREC 语料的调参结果 [1][2]，但不同领域、不同文本类型的最优参数差异显著：短而精炼的标题/推文语料偏好小 k1，代码库、法律文书等重复度高的长文本偏好大 k1；文档长度差异极大的语料应降低 b，否则短文档被系统性过度惩罚。固定默认参数等于假设所有语料具有相同统计结构——这一假设在跨领域应用中并不成立。

### 1.2 零标注自适应：为什么统计特征可以预测最优参数

调参的障碍在于需要标注。但如果"最优参数"本质上由语料的**统计结构**决定，那么我们可以直接测量这种结构：

- **文档长度分布**决定 b 的合理取值：长度差异越大（cv_len 高），长度归一化越容易把短文档打到接近零分，因此最优 b 应越小；文档长度与词频的共变关系（length_tf_corr）进一步区分"长文档只是更长"与"长文档携带更多独特内容"两种情形。
- **文档内词频冗余度**决定 k1 的合理取值：类型-标记比（TTR）接近 1 说明文档精炼、词频接近伯努利噪声，词频信号很快饱和，k1 应小；TTR 低说明词频有大量冗余，需要更大的 k1 让高频词持续贡献。
- **词汇增长与稀疏性**决定 IDF 变体：Heaps' Law 的指数 β [5] 刻画词汇量随文本量的增长速率，hapax ratio（只出现一次的词占比）刻画词汇稀疏性；多领域、词汇稀疏的语料中稀有词携带更强判别信号，平滑 IDF 的温和惩罚更合适。

这些假设构成了本项目启发式规则的设计依据，也是后续用数据标定系数时的先验方向。

---

## 2. 问题定义

给定文档集 D、查询集 Q（与可选标注），定义统计特征提取器 Φ(D, Q) → **f** ∈ ℝ^16，以及参数预测函数 g(**f**) → **θ** = (k1, b, δ, idf_type)。目标是在**不使用相关性标注**的条件下，使零标注预测参数的检索性能逼近"上帝视角"的上界：

```
θ* = argmax_{θ ∈ Θ} NDCG@10(D, Q, qrels; θ)          （grid search 上界）
minimize  NDCG@10(θ*) − NDCG@10(g(Φ(D, Q)))            （零标注预测与上界的差距）
```

搜索空间：k1 ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0}，b ∈ {0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0}，δ ∈ {0.0, 0.3, 0.5, 0.7, 1.0, 1.5}，idf_type ∈ {rsj, smoothed}，共 8×11×6×2 = **1056 个组合**。

---

## 3. 方法

### 3.1 统计特征体系（feature_extractor.py）

特征按实现优先级分为 P0（必须）、P1（建议）、P2（可选），按预测目标分为四个维度。分词方式统一为空格分词 + 小写化，不去停用词（停用词仅用于单独的密度统计）。

**维度一：文档长度与分布特征（决定 b）**

| 特征 | 优先级 | 定义 | 机制假设 |
|---|---|---|---|
| doc_count | P0 | 文档总数 | 样本量 |
| avgdl | P0 | 平均文档长度（词数） | 长度归一化的参照量 |
| std_len | P0 | 文档长度标准差 | 长度离散度 |
| cv_len | P0 | 变异系数 = std_len / avgdl | **cv_len 高 → b 应小**（避免短文档被过度惩罚） |
| len_skew | P1 | 长度分布偏度（scipy.stats.skew） | 偏度高 → 少量超长文档 → 需进一步降 b 或切片 |
| len_p90 / len_p10 | P1 | 长度 90/10 分位数 | 稳健的长度位置量 |
| len_ratio_p90_p10 | P1 | p90 / p10 | 长度长尾程度 |

**维度二：文档内词频与冗余度特征（决定 k1）**

| 特征 | 优先级 | 定义 | 机制假设 |
|---|---|---|---|
| avg_ttr | P0 | 平均类型-标记比（每篇 unique/总词数，再平均） | **TTR 高 → 文档精炼 → k1 应小**；TTR 低 → 词汇重复 → k1 应大 |
| avg_max_tf | P0 | 每篇文档去停用词后的最大词频，再平均 | 反映词频自然饱和的天花板；值高 → k1 应大 |
| length_tf_corr | P0 | 文档长度与文档内平均词频（总词数/唯一词数）的皮尔逊相关 | **corr 高 → 长文档冗余多 → b 应大**；corr 低 → 长文档有独特内容 → b 应小 |
| hapax_ratio | P1 | 全文只出现一次的词数 / 词汇量 | 词汇稀疏性与噪音水平 |

**维度三：全局词汇分布特征（决定 IDF 类型）**

| 特征 | 优先级 | 定义 | 机制假设 |
|---|---|---|---|
| vocab_size | P1 | 词汇表大小 | 语料规模 |
| heaps_beta | P1 | Heaps' Law 拟合 β：log(V)=log(K)+β·log(n) | 刻画词汇增长速率，β 高 → 多领域/词汇开放 |
| stopword_density | P1 | 停用词占总词数比例 | 密度极高时长度归一化会被停用词污染，avgdl 应相应调整 |
| zipf_alpha | P2 | Zipf 律拟合指数：log(freq) ~ −α·log(rank) | 词频分布陡峭程度，与 heaps_beta 交叉验证 |

**维度四：查询侧特征（可选，有查询集时计算）**

| 特征 | 优先级 | 定义 |
|---|---|---|
| avg_query_len | P2 | 平均查询长度 |
| query_idf_mean / query_idf_std | P2 | 查询词在语料中 IDF 的均值/标准差 |
| query_oov_ratio | P2 | 查询中 OOV 词（不在语料词汇表中）的比例 |

### 3.2 启发式参数预测规则（rule_predictor.py）

系数全部写在 `config.yaml` 中，便于标定：

```
k1    = clip(k1_base − alpha_ttr·avg_ttr + alpha_maxtf·min(avg_max_tf/20, 1), 0.5, 3.0)
b     = clip(b_base + beta_corr·length_tf_corr − beta_cv·cv_len, 0, 1)
δ     = clip(gamma_delta·cv_len, 0, 2.0)
idf   = smoothed（若 heaps_beta > 0.7 或 hapax_ratio > 0.6），否则 rsj
```

默认系数：k1_base=1.5, alpha_ttr=1.5, alpha_maxtf=0.3, b_base=0.5, beta_corr=0.4, beta_cv=0.3, gamma_delta=1.0。符号方向由 3.1 节的机制假设决定；例如 b 公式中 length_tf_corr 取正号、cv_len 取负号。δ>0 时模型变体为 BM25+，否则退化为标准 BM25。

### 3.3 数据增强（data_augmentation.py）

公开标注数据集数量有限（如 BEIR 仅十几个），而本方法只关心统计结构、不关心语义内容，因此可以通过**变换文档集**扩充"（特征向量, 最优参数）"样本：

1. **random_subsample**：按比例（0.1/0.3/0.5/0.7）× 3 个随机种子采样文档，制造不同规模的语料；
2. **length_bucket_split**：按文档长度分桶（桶内 cv_len 小）与首尾桶组合（cv_len 大），显式控制长度分布形态；
3. **truncate_docs**：截断到 max_len（50/100/200/500）词，doc_id 与标注不变；
4. **sliding_window**：长文档滑动切片（50/100/200 词窗），原相关文档的所有切片均标为相关。

每种变换生成标准格式子数据集，保存到 `dataset/augmented/<name>/`。

### 3.4 系数标定（evaluator.calibrate）

1. 对每个数据集（原始 + 增强）提取特征，grid search 得到最优参数作为 ground truth；
2. 收集全部（特征向量, 最优参数）配对；
3. 用**非负最小二乘**（NNLS）按公式结构拟合系数（符号方向由公式保证），并在小网格上搜索 IDF 判断阈值；
4. 更新 `config.yaml`；
5. 在未参与标定的数据集上输出 default / predicted / grid_search 三组对比报告。

---

## 4. 实验设置

### 4.1 数据集

使用 8 个 BEIR [7] 数据集，覆盖多种文本类型与规模（3.6k–52 万文档、短查询到长查询）。

| 数据集 | 文档数 | 查询数 | 评估查询数 | grid 查询数 | 领域 |
|---|---:|---:|---:|---:|---|
| arguana | 8,674 | 1,406 | 1,406 | 20 | 议论文论点检索（超长查询） |
| fiqa | 57,638 | 6,648 | 6,648 | 50 | 金融领域问答 |
| nfcorpus | 3,633 | 3,237 | 3,237 | 100 | 医学信息检索 |
| quora | 522,931 | 15,000 | 1,000* | — | 社区问答去重 |
| scidocs | 25,657 | 1,000 | 1,000 | — | 科学文献检索 |
| scifact | 5,183 | 1,109 | 1,109 | 100 | 科学论断证据检索 |
| trec-covid | 171,332 | 50 | 50 | 20 | COVID-19 学术文献 |
| webis-touche2020 | 382,545 | 49 | 49 | 20 | 论点检索 |

\* quora 评估取前 1000 条查询采样（全量 1.5 万条在单机 Python 实现下计算量过大）。grid 查询数为该数据集上 grid search 使用的查询子集大小（用于同口径上界对照）；quora 与 scidocs 未跑 grid。

### 4.2 评估协议

- 分词：空格 + 小写，不去停用词；
- 指标：MRR@10、NDCG@10（支持分级相关性）、Recall@100，对每个查询取 top-100 结果计算；
- 基线：**default**（k1=1.2, b=0.75, δ=0, rsj）；**predicted**（本项目零标注规则）；**grid_search**（1056 组合 grid search 最优，上界）；
- 引擎：自实现倒排索引 BM25/BM25+（numpy 向量化打分，见 7 节性能说明）；
- 复现：`python run_experiments.py --all --eval-queries -1 ...`，原始结果存于 `results/benchmark_results.json`，表格由 `python summarize_results.py` 生成。

---

## 5. 实验结果与结论

### 5.1 语料统计特征（节选）

**长度/词频维度**

| 数据集 | avgdl | cv_len | len_skew | p90/p10 | avg_ttr | avg_max_tf | length_tf_corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| arguana | 167.2 | 0.531 | 1.96 | 3.50 | 0.690 | 4.17 | 0.723 |
| fiqa | 132.9 | 0.969 | 3.80 | 6.14 | 0.740 | 3.32 | 0.715 |
| nfcorpus | 233.8 | 0.357 | 2.99 | 2.28 | 0.606 | 7.10 | 0.616 |
| quora | 11.5 | 0.548 | 2.38 | 3.17 | 0.970 | 1.08 | 0.572 |
| scidocs | 176.7 | 0.665 | 7.11 | 2.78 | 0.648 | 5.69 | 0.592 |
| scifact | 214.6 | 0.407 | 2.98 | 2.43 | 0.616 | 7.16 | 0.598 |
| trec-covid | 161.2 | 0.837 | 27.62 | 33.00 | 0.715 | 4.88 | 0.764 |
| webis-touche2020 | 293.8 | 1.316 | 2.49 | 63.50 | 0.690 | 5.86 | 0.135 |

**词汇/查询维度**

| 数据集 | hapax_ratio | heaps_beta | stopword_density | zipf_alpha | avg_query_len | query_idf_mean | query_oov |
|---|---:|---:|---:|---:|---:|---:|---:|
| arguana | 0.540 | 0.696 | 0.420 | 1.239 | 193.6 | 2.10 | 0.000 |
| fiqa | 0.607 | 0.704 | 0.465 | 1.232 | 10.8 | 2.70 | 0.025 |
| nfcorpus | 0.567 | 0.741 | 0.334 | 1.152 | 3.3 | 3.29 | 0.107 |
| quora | 0.580 | 0.759 | 0.488 | 1.265 | 9.5 | 4.50 | 0.007 |
| scidocs | 0.601 | 0.727 | 0.362 | 1.217 | 9.4 | 2.99 | 0.039 |
| scifact | 0.571 | 0.748 | 0.336 | 1.146 | 12.4 | 2.32 | 0.051 |
| trec-covid | 0.569 | 0.721 | 0.340 | 1.214 | 10.6 | 2.49 | 0.008 |
| webis-touche2020 | 0.658 | 0.714 | 0.464 | 1.143 | 6.6 | 3.26 | 0.003 |

### 5.2 零标注预测参数

| 数据集 | k1 | b | δ | IDF | 模型变体 |
|---|---:|---:|---:|---|---|
| arguana | 0.528 | 0.630 | 0.531 | rsj | BM25+ |
| fiqa | 0.500 | 0.495 | 0.969 | smoothed | BM25+ |
| nfcorpus | 0.697 | 0.639 | 0.357 | smoothed | BM25+ |
| quora | 0.500 | 0.564 | 0.548 | smoothed | BM25+ |
| scidocs | 0.614 | 0.537 | 0.665 | smoothed | BM25+ |
| scifact | 0.683 | 0.617 | 0.407 | smoothed | BM25+ |
| trec-covid | 0.501 | 0.555 | 0.837 | smoothed | BM25+ |
| webis-touche2020 | 0.553 | 0.159 | 1.316 | smoothed | BM25+ |

三个可观察的规律：**(1)** 所有数据集 b 均低于默认 0.75，且 webis（cv_len 最大，1.32）b 最低（0.16），与"长度差异大 → 降 b"的假设一致；**(2)** 全部启用 BM25+（δ>0），δ 与 cv_len 正相关；**(3)** 7/8 数据集选择 smoothed IDF（arguana 因 heaps_beta 与 hapax 均低于阈值而选 rsj）。

### 5.3 零标注预测 vs 默认参数（全量评估）

| 数据集 | MRR@10 default→predicted | NDCG@10 default→predicted | Recall@100 default→predicted |
|---|---:|---:|---:|
| arguana | 0.0469 → 0.0642 (**+37.1%**) | 0.0718 → 0.1022 (**+42.4%**) | 0.5014 → 0.6494 (**+29.5%**) |
| fiqa | 0.1648 → 0.1810 (+9.8%) | 0.1322 → 0.1425 (+7.8%) | 0.3270 → 0.3456 (+5.7%) |
| nfcorpus | 0.4583 → 0.4674 (+2.0%) | 0.2626 → 0.2684 (+2.2%) | 0.1982 → 0.2021 (+1.9%) |
| quora | 0.6565 → 0.6410 (−2.4%) | 0.6399 → 0.6272 (−2.0%) | 0.9028 → 0.8884 (−1.6%) |
| scidocs | 0.2189 → 0.2342 (+7.0%) | 0.1191 → 0.1306 (+9.7%) | 0.2888 → 0.3064 (+6.1%) |
| scifact | 0.4781 → 0.5472 (+14.4%) | 0.5111 → 0.5824 (+13.9%) | 0.7997 → 0.8305 (+3.8%) |
| trec-covid | 0.5754 → 0.6794 (+18.1%) | 0.3202 → 0.4079 (**+27.4%**) | 0.0544 → 0.0646 (+18.8%) |
| webis-touche2020 | 0.4931 → 0.5727 (+16.1%) | 0.2227 → 0.2820 (+26.6%) | 0.4349 → 0.4302 (−1.1%) |

**结论 1：零标注规则预测在 7/8 数据集上优于固定默认参数**，8 集平均 MRR@10 +12.8%、NDCG@10 +16.0%、Recall@100 +7.9%；剔除唯一负例 quora 后，NDCG@10 平均提升 +18.6%。提升最大的三类场景是：超长查询的论点检索（arguana）、词汇/长度极度长尾的学术语料（trec-covid、webis）、以及论断级证据检索（scifact）。

**结论 2：quora 是唯一的负例（约 −2%）**。quora 文档为短问答（avgdl≈11，avg_ttr≈0.97，词频几乎无冗余），基线本身已很高（MRR@10≈0.66、Recall@100≈0.90），规则预测的 BM25+ 与 smoothed IDF 在该场景下没有带来收益；这提示在"短文档 + 高 TTR + 大规模"场景下，规则预测的收益边际很小，需要查询侧特征（如 query_idf 分布）进一步刻画。

### 5.4 与 grid search 上界对照（同口径查询子集）

为避免"上界用前 N 条查询、评估用全量查询"的口径错位，下表 default/predicted/grid 三者均在同一查询子集上计算。

| 数据集 | 指标 | default | predicted | grid 上界 | 达上界 |
|---|---|--:|--:|--:|--:|
| arguana (20q) | MRR@10 / NDCG@10 / R@100 | 0.086 / 0.137 / 0.950 | 0.113 / 0.202 / 0.900 | 0.235 / 0.369 / 1.000 | 48% / 55% / 90% |
| fiqa (50q) | MRR@10 / NDCG@10 / R@100 | 0.074 / 0.059 / 0.288 | 0.082 / 0.071 / 0.357 | 0.124 / 0.095 / 0.378 | 66% / 75% / 95% |
| nfcorpus (100q) | MRR@10 / NDCG@10 / R@100 | 0.469 / 0.249 / 0.140 | 0.496 / 0.264 / 0.144 | 0.534 / 0.284 / 0.150 | 93% / 93% / 97% |
| scifact (100q) | MRR@10 / NDCG@10 / R@100 | 0.428 / 0.465 / 0.795 | 0.511 / 0.562 / 0.830 | 0.548 / 0.588 / 0.830 | 93% / 96% / 100% |
| trec-covid (20q) | MRR@10 / NDCG@10 / R@100 | 0.554 / 0.278 / 0.046 | 0.652 / 0.342 / 0.055 | 0.852 / 0.499 / 0.069 | 77% / 69% / 80% |
| webis-touche2020 (20q) | MRR@10 / NDCG@10 / R@100 | 0.501 / 0.225 / 0.377 | 0.634 / 0.290 / 0.387 | 0.700 / 0.359 / 0.473 | 91% / 81% / 82% |

**结论 3：在相同查询子集上，零标注预测达到 grid search 上界的 65%–100%**（NDCG@10）。nfcorpus 与 scifact 达 93%–96%，webis 达 81%，trec-covid 达 69%–80%，fiqa 达 75%–95%。唯一的明显短板是 arguana（NDCG@10 仅达 55%）——其查询本身就是超长议论文（平均 194 词），参数最优值对上界极为敏感，且规则 k1 预测（0.53）与 grid 最优（3.0）方向相反。

**结论 4：IDF 变体的规则预测与上界高度一致**——grid search 在全部 6 个采样数据集上均选择 smoothed IDF，而规则在 7/8 数据集上预测 smoothed。唯一例外是 arguana（heaps_beta=0.696、hapax_ratio=0.540 略低于阈值而判为 rsj），但其 grid 上界同样为 smoothed，说明当前 IDF 阈值在该边界样本上判反了方向。总体上支持"多领域/稀有词多的语料应使用平滑 IDF"的假设。

**结论 5（主要局限）：当前 k1 规则系统性偏低**。规则预测 k1 ∈ [0.50, 0.70]，而 grid 上界在 4/6 采样数据集上偏好 k1=3.0。可能原因是 avg_ttr 主导了 k1 公式，而缺少对文档内词频分布形状（如高频词集中度、重复词间隔）的刻画；这也是后续标定与特征扩展的重点。

---

## 6. 局限性与未来工作

- **k1 规则偏差**：见结论 5，需要引入词频分布形状特征（如 top-k 高频词占比、文档内重复词的集中度），或让标定流程用真实数据集重新拟合系数。
- **长查询场景**：arguana 表明超长查询下参数敏感性极高，可考虑查询侧特征（query_idf_std、avg_query_len）进入规则，或对长查询做 term 级加权/剪枝。
- **大规模短文本**：quora（52 万文档、1.5 万查询）显示收益边际小且可能为负，需要针对"高 TTR 短文档"的专门处理。
- **上界估计**：grid search 在本工作中对大部分数据集采用查询子集采样（表注），且未对 quora/scidocs 计算上界；完整上界与更细网格（如 k1 0.1 步长）可作为后续实验。
- **标定流程未在本结论中使用**：当前结果全部来自 `config.yaml` 的默认启发式系数；`--calibrate` 会结合增强数据重拟合系数，可在更大规模上验证（需要较多计算时间）。
- **可扩展方向**：线性规则 → 小型回归/树模型；特征选择与消融（P0/P1/P2 逐级）；BM25 之外的排序模型迁移；按查询聚类自适应参数。

---

## 7. 快速开始（复现）

```bash
pip install -r requirements.txt

# 单数据集：预测参数并保存特征/参数 json
python main.py --dataset dataset/XXX --predict

# 单数据集：评估（default vs predicted，可加 --grid 画上界）
python main.py --dataset dataset/XXX --eval
python main.py --dataset dataset/XXX --eval --grid --limit-queries 100

# 生成增强数据集（dataset/augmented/XXX/）
python main.py --dataset dataset/XXX --augment

# 系数标定（所有原始数据集；--include-augmented 纳入增强集）
python main.py --calibrate --include-augmented --limit-queries 200

# 批量实验 + 结果汇总（本 README 表格的复现入口）
python run_experiments.py --all --eval-queries -1
python summarize_results.py
```

数据格式支持两种：项目标准格式（`docs.jsonl` / `queries.jsonl` / `qrels.jsonl`）与 BEIR 格式（`corpus.jsonl` / `queries.jsonl` / `qrels/*.tsv`，自动识别 `_id`/`qid` 与 `query-id/corpus-id` 表头）。

### 性能说明

检索引擎使用倒排索引 + numpy 向量化打分；grid search 采用"查询计划"预计算（每条查询的倒排条目只拼接一次，每个参数组合一次向量化累加），单机可跑 50 万级文档。完整 1056 组合 grid 在大型语料上仍较耗时，建议用 `--limit-queries` 采样查询；万级文档 + 千级查询的单次评估约 1 分钟内。

---

## 8. 项目结构

```
autobm25/
├── main.py                 # 命令行入口（predict / eval / grid / augment / calibrate）
├── config.yaml             # 启发式系数、默认参数、grid 空间、增强参数
├── data_loader.py          # 数据集加载/保存（标准格式 + BEIR 格式）
├── feature_extractor.py    # 16 维统计特征（P0/P1/P2 分层）
├── rule_predictor.py       # 特征 → 参数 的启发式规则
├── bm25_engine.py          # 自实现 BM25/BM25+（倒排索引 + numpy 向量化）
├── data_augmentation.py    # 数据增强（子采样/分桶/截断/滑动窗口）
├── evaluator.py            # 评估指标、grid search、系数标定、对比报告
├── run_experiments.py      # 批量实验（结果写入 results/benchmark_results.json）
├── summarize_results.py    # 汇总实验结果为 Markdown 表格
├── stopwords.txt
└── requirements.txt
```

## 参考文献

1. Robertson, S. E., & Walker, S. (1994). *Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval*. SIGIR.
2. Robertson, S. E., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
3. Lv, Y., & Zhai, C. (2011). *Lower-Bounding Term Frequency Normalization* (BM25+). CIKM.
4. Fang, H., Tao, T., & Zhai, C. (2004). *A Formal Study of Information Retrieval Heuristics*. SIGIR.
5. Heaps, H. S. (1978). *Information Retrieval: Computational and Theoretical Aspects*. Academic Press.
6. Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*. Addison-Wesley.
7. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS Datasets and Benchmarks.
8. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
