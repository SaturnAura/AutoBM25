# AutoBM25 —— 零标注、零调参的 BM25 超参数自适应

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 🇬🇧 English version: [README.md](README.md)

![AutoBM25 logo](logo/logo.png)

> 输入任意语料的**统计特征** → **O(1) 大字典查表**直接得到最优 BM25 超参数（k1 / b / k3 / δ / IDF）；查不到就启发式兜底。16 个 BEIR 数据集验证：**16/16 不劣于默认参数、平均 NDCG@10 +11.6%、词典命中率 92.0%**。

> 😄 或许我们应该叫 **BM26**？——参数都能自适应了，版本号也自适一下（开个玩笑，BM25 还是 BM25）。

---

## 为什么需要它

BM25 是信息检索最常用的排序模型，但它的**最优超参数随语料变化极大**——短文本、长文本、代码、法律文书、问答社区，各自的最优 k1/b 完全不同。实际工程却永远用 `k1=1.2, b=0.75` 的默认值，因为调参需要带标注的验证集，而新语料往往**没有标注**。

AutoBM25 换了一条路：**最优参数是语料统计结构的函数**，而统计结构不需要标注就能测量。测量出 18 维统计特征，查表即得参数——零标注、零调参、常数时间。

## 亮点

- 🗂️ **18 维统计特征**（P0/P1/P2）：文档长度分布、词频冗余度、词汇增长曲线、查询侧特征——每个特征都有明确的"机制假设"（为什么它能预测某个参数）
- ⚡ **O(1) 参数大字典**：`特征向量 → 最优参数` 最近邻查表，含**分布外（OOD）防护**与**启发式回退**；30 个条目随仓库提交，每新增一个做过 grid search 的数据集就多一条经验
- 🔧 **k3 查询词频饱和项**：消融试验确认有效，长查询场景相对提升最高 **+19.9%**
- 🧩 **多语料拼接（Mixing）**：把已有语料组合拼接生成"跨域样本"，主动在特征空间补点，词典命中率 **70.6% → 92.0%**
- 🚀 **规模验证到 884 万文档**（msmarco），全自研 numpy 向量化 BM25/BM25+ 引擎，超大数据集流式子采样、相关标注全保留

## 数值提升（16 个 BEIR 数据集）

**原版 BM25（默认参数）vs AutoBM25（自适应）—— MRR@10 / NDCG@10 / Recall@100**

每个单元格是 `原版 BM25 → AutoBM25`，在同一批查询上计算（默认参数：k1=1.2, b=0.75, k3=0, δ=0, rsj IDF）。

| 数据集 | MRR@10 | NDCG@10 | Recall@100 |
|---|---:|---:|---:|
| arguana | 0.2036 → 0.1672 | 0.3158 → 0.2610 | 0.9090 → 0.8563 |
| climate-fever | 0.2480 → 0.1982 | 0.1778 → 0.1375 | 0.4551 → 0.4339 |
| dbpedia-entity | 0.2724 → 0.2972 | 0.2330 → 0.2510 | 0.4638 → 0.4868 |
| fever | 0.5983 → 0.7263 | 0.6105 → 0.7277 | 0.9007 → 0.9276 |
| fiqa | 0.1727 → 0.1818 | 0.1387 → 0.1429 | 0.3429 → 0.3465 |
| hotpotqa | 0.8131 → 0.8116 | 0.6478 → 0.6582 | 0.7700 → 0.7740 |
| msmarco | 0.4024 → 0.4116 | 0.4400 → 0.4506 | 0.7490 → 0.7603 |
| nfcorpus | 0.4600 → 0.4674 | 0.2633 → 0.2684 | 0.1984 → 0.2018 |
| nq | 0.1594 → 0.1796 | 0.1774 → 0.2083 | 0.5937 → 0.6227 |
| scidocs | 0.2228 → 0.2360 | 0.1215 → 0.1307 | 0.2936 → 0.3084 |
| scifact | 0.5033 → 0.5471 | 0.5388 → 0.5829 | 0.8116 → 0.8312 |
| trec-covid | 0.5936 → 0.6794 | 0.3322 → 0.4060 | 0.0573 → 0.0640 |
| trec-covid-beir | 0.6107 → 0.6894 | 0.3394 → 0.4147 | 0.0575 → 0.0640 |
| trec-covid-v2 | 0.4079 → 0.7130 | 0.2127 → 0.4121 | 0.0494 → 0.0755 |
| vihealthqa | 0.3824 → 0.3495 | 0.3676 → 0.3373 | 0.6248 → 0.5910 |
| webis-touche2020 | 0.4931 → 0.5727 | 0.2226 → 0.2816 | 0.4336 → 0.4302 |
| **平均** | **0.4090 → 0.4518** | **0.3212 → 0.3544** | **0.4819 → 0.4859** |
## 快速开始

```bash
pip install .

# 批量检索：跑完 dataset/XXX/queries.jsonl 的全部查询，
# 结果写回 dataset/XXX/results.jsonl；有 qrels 时自动附带评测（evaluation.json）
autobm25 --dataset dataset/XXX

# 交互式检索：显示输入栏，输入查询 → 回车出结果，exit 退出
autobm25 --dataset dataset/XXX --interactive
```

也可以当库用：

```python
from autobm25 import AutoBM25

corpus = [
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Automatic parameter tuning improves retrieval effectiveness.",
]
retriever = AutoBM25()
retriever.index(corpus)                  # 直接传字符串列表即可，不依赖任何文件
print(retriever.params)                  # 该语料自动选出的超参数
retriever.search("parameter tuning", top_k=5)
```

想用数据集目录？`AutoBM25.from_dataset("dataset/fiqa")` 支持标准格式 / BEIR 格式，效果一样。

启动时会打印一条日志，显示本数据集自适应选择的 BM25 超参数（如 `k1=0.5 b=0.495 k3=0.7 δ=0.97 idf=smoothed`）。

## 数据格式

数据集放在 `dataset/XXX/` 目录下，支持两种格式（自动识别）：

**标准格式**

```
docs.jsonl      # 每行 {"id": "...", "text": "..."}
queries.jsonl   # 每行 {"qid": "...", "query": "..."}
qrels.jsonl     # 每行 {"qid": "...", "doc_id": "...", "relevance": 1}
```

**BEIR 格式**

```
corpus.jsonl    # 每行 {"_id": "...", "title": "...", "text": "..."}
queries.jsonl   # 每行 {"_id": "...", "text": "..."}
qrels/*.tsv     # 表头 query-id \t corpus-id \t score（也兼容 qid / corpus_id 命名）
```

纯检索时只需要 `docs.jsonl`（或 `corpus.jsonl`）即可，查询可以交互式输入，qrels 可缺省。大字典随包内置（`autobm25/data/param_dictionary.json`），`pip install .` 后开箱即用，无需先跑实验。

## 局限

- **分布外（OOD）**：当新语料特征超出词典训练覆盖（超长查询、跨域混合语料等）时，查表自动回退启发式——极端/混合场景下规则仍可能弱于默认参数。
- **非英文语料**：特征含英文停用词密度等语言相关项，非英文语料（如越南语 vihealthqa）需要语言无关特征或独立分册。

## 项目结构

```
autobm25/
├── autobm25/              # pip 包：检索接口（引擎、特征、词典、CLI）+ data/（大字典）
├── logo/                 # 项目 logo
├── config.yaml           # 启发式系数（本地 research/ 工具用）
├── pyproject.toml        # pip 打包配置
├── requirements.txt
└── README.md
```

## License

本项目采用 [MIT License](LICENSE)：代码可**自由复制、修改、商用**，唯一要求是在副本中**保留版权声明与本许可文本**（即"使用请注明出处"）。引用本项目时附上仓库链接即可满足要求。
