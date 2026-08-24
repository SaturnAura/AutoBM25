# AutoBM25

基于数据集统计特征的 BM25 自适应超参数选择：不依赖任何相关性标注，仅通过分析
文档集本身的统计特征（长度分布、词频冗余度、词汇分布等），预测最适合该数据集的
BM25 超参数（k1, b, δ, IDF 类型），并在零标注条件下逼近 grid search 的上界。

## 项目结构

```
autobm25/
├── main.py                # 命令行入口
├── config.yaml            # 启发式公式系数、默认参数、grid search 空间、增强参数
├── data_loader.py         # 数据集加载/保存（标准格式 + BEIR 格式）
├── feature_extractor.py   # 统计特征提取
├── rule_predictor.py      # 特征 -> 参数 的启发式预测
├── bm25_engine.py         # 自实现 BM25 / BM25+（倒排索引 + numpy 向量化打分）
├── data_augmentation.py   # 数据增强（扩充“特征, 最优参数”样本）
├── evaluator.py           # 评估指标、grid search、系数标定、对比报告
├── stopwords.txt          # 内置英文停用词表
└── requirements.txt
```

## 数据格式

每个数据集一个目录，支持两种格式：

1. 标准格式（增强输出也使用这种）：
   ```
   dataset/XXX/docs.jsonl      # 每行 {"id": "...", "text": "..."}
   dataset/XXX/queries.jsonl   # 每行 {"qid": "...", "query": "..."}
   dataset/XXX/qrels.jsonl     # 每行 {"qid": "...", "doc_id": "...", "relevance": int}
   ```
2. BEIR 格式（如 dataset/fiqa）：
   ```
   dataset/XXX/corpus.jsonl    # 每行 {"_id": "...", "title": "...", "text": "..."}
   dataset/XXX/queries.jsonl   # 每行 {"_id": "...", "text": "..."}
   dataset/XXX/qrels/*.tsv     # 表头 query-id / corpus-id / score（自动兼容 qid/corpus_id）
   ```

分词方式：空格分词 + 小写化，不去停用词（停用词只用于单独统计密度和 avg_max_tf）。

## 用法

```bash
pip install -r requirements.txt

# 只预测参数（保存特征与预测结果到 results/）
python main.py --dataset dataset/XXX --predict

# 预测并评估（default vs predicted）
python main.py --dataset dataset/XXX --eval

# 评估并额外做 grid search（上界；大数据集建议 --limit-queries）
python main.py --dataset dataset/XXX --eval --grid --limit-queries 100

# 生成增强数据集（保存到 dataset/augmented/XXX/）
python main.py --dataset dataset/XXX --augment

# 用 dataset/ 下所有原始数据集标定 config.yaml 中的系数
python main.py --calibrate

# 标定时也纳入增强数据集（样本多、耗时更长）
python main.py --calibrate --include-augmented --limit-queries 200
```

所有中间结果（特征、预测参数、grid search 结果、标定 ground truth 与对比报告）
都以 json 保存到 `results/` 目录，方便后续分析。

## 特征 -> 参数 的启发式规则

```text
k1    = clip(k1_base - alpha_ttr*avg_ttr + alpha_maxtf*min(avg_max_tf/20, 1), 0.5, 3.0)
b     = clip(b_base + beta_corr*length_tf_corr - beta_cv*cv_len, 0, 1)
delta = clip(gamma_delta*cv_len, 0, 2.0)
idf_type = smoothed（若 heaps_beta>0.7 或 hapax_ratio>0.6），否则 rsj
```

系数默认值写在 `config.yaml` 中，`--calibrate` 会用非负最小二乘拟合更新它们
（同时在小网格上搜索 IDF 阈值），并在留出的测试数据集上输出
default / predicted / grid_search 三组对比与提升比例。

## 说明与提示

- BM25 引擎对打分做了 numpy 向量化，万级文档集 + 千级查询的评估约 1 分钟；
  完整 grid search（8×11×6×2 = 1056 组合）在大型数据集上耗时较长，
  建议配合 `--limit-queries` 采样查询数使用。
- 研究阶段代码，未做工程级异常处理与单元测试。
