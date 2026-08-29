# AutoBM25 — Zero-Label, Zero-Tuning BM25 Hyperparameter Adaptation

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![AutoBM25 logo](logo/logo.png)

> Feed the corpus's **statistical features** into an **O(1) parameter dictionary** and get the optimal BM25 hyperparameters (k1 / b / k3 / δ / IDF) directly; fall back to interpretable heuristics when nothing matches. Verified on 16 BEIR datasets: **16/16 never worse than defaults, average NDCG@10 +11.6%, dictionary hit rate 92.0%**.

> 😄 Maybe we should call it **BM26**? — if the hyperparameters can adapt, so can the version number (just kidding, BM25 is BM25).

---

## Why

BM25 is the most widely used ranking model in information retrieval, but its **optimal hyperparameters vary dramatically across corpora** — short texts, long texts, code, legal documents, Q&A communities each need different k1/b values. In practice, everyone uses the fixed defaults `k1=1.2, b=0.75`, because tuning requires a labeled validation set, and **new corpora rarely have labels**.

AutoBM25 takes a different path: **optimal parameters are a function of the corpus's statistical structure, and statistical structure can be measured without any labels.** Measure 18 features, look up the parameters — zero labels, zero tuning, constant time.

## Highlights

- 🗂️ **18 statistical features** (P0/P1/P2): document length distribution, term-frequency redundancy, vocabulary growth, query-side statistics — each with an explicit "mechanism hypothesis" (why it predicts a given parameter)
- ⚡ **O(1) parameter dictionary**: nearest-neighbor lookup from feature vectors to optimal parameters, with **out-of-distribution (OOD) guard** and **heuristic fallback**; 44 entries shipped with the repo, growing with every new grid-searched dataset
- 🔧 **k3 query-term-frequency saturation**: ablation-verified, up to **+19.9%** on long-query scenarios
- 🧩 **Corpus mixing**: combine existing corpora into cross-domain blends to fill feature-space gaps; dictionary hit rate **70.6% → 92.0%**
- 🚀 **Scales to 8.84M documents** (msmarco); self-built numpy-vectorized BM25/BM25+ engine with streaming stratified subsampling that keeps all relevant annotations

## Results (16 BEIR datasets)

**Original BM25 (defaults) vs AutoBM25 (adaptive) — MRR@10 / NDCG@10 / Recall@100**

Each cell shows `original BM25 → AutoBM25` on the same queries (defaults: k1=1.2, b=0.75, k3=0, δ=0, rsj IDF).

| Dataset | MRR@10 | NDCG@10 | Recall@100 |
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
| **Average** | **0.4090 → 0.4518** | **0.3212 → 0.3544** | **0.4819 → 0.4859** |
## Quick Start

```bash
pip install .

# Batch retrieval: run all queries in dataset/XXX/queries.jsonl,
# write results back to dataset/XXX/results.jsonl; auto-evaluates if qrels exist
autobm25 --dataset dataset/XXX

# Interactive retrieval: type a query, press Enter, exit with "exit"
autobm25 --dataset dataset/XXX --interactive
```

Or use it as a library:

```python
from autobm25 import AutoBM25

corpus = [
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Automatic parameter tuning improves retrieval effectiveness.",
]
retriever = AutoBM25()
retriever.index(corpus)                 # index a plain list of strings — no files needed
print(retriever.params)                 # hyperparameters chosen for this corpus
retriever.search("parameter tuning", top_k=5)
```

Working with dataset directories instead? `AutoBM25.from_dataset("dataset/fiqa")` builds the same thing from standard/BEIR-format folders.

On startup a log line shows the hyperparameters chosen for this dataset (e.g., `k1=0.5 b=0.495 k3=0.7 δ=0.97 idf=smoothed`).

The parameter dictionary is bundled inside the package (`autobm25/data/param_dictionary.json`), so retrieval works out of the box after `pip install .` — no need to run any experiments first.

## Data Formats

Datasets live in `dataset/XXX/`; both formats are auto-detected:

**Standard format**

```
docs.jsonl      # one line per doc: {"id": "...", "text": "..."}
queries.jsonl   # {"qid": "...", "query": "..."}
qrels.jsonl     # {"qid": "...", "doc_id": "...", "relevance": 1}
```

**BEIR format**

```
corpus.jsonl    # {"_id": "...", "title": "...", "text": "..."}
queries.jsonl   # {"_id": "...", "text": "..."}
qrels/*.tsv     # header: query-id \t corpus-id \t score (also accepts qid / corpus_id)
```

For retrieval only, `docs.jsonl` (or `corpus.jsonl`) is enough — queries can be typed interactively and qrels are optional.

## Limitations

- **Out-of-distribution (OOD)**: when a new corpus's features fall outside the dictionary's training coverage (e.g., extremely long queries or cross-domain blends), lookup falls back to heuristics — rules can still underperform defaults in extreme/mixed scenarios.
- **Non-English corpora**: features include English-stopword-density and other language-dependent terms; non-English corpora (e.g., Vietnamese vihealthqa) need language-agnostic features or a separate section of the dictionary.

## Project Structure

```
autobm25/
├── autobm25/              # pip package: retrieval API (engine, features, dictionary, CLI) + data/ (dictionary)
├── logo/                 # Project logo
├── config.yaml           # Heuristic coefficients (used by local research/ tooling)
├── pyproject.toml        # pip packaging
├── requirements.txt
└── README.md
```

## License

[MIT License](LICENSE): the code can be freely **copied, modified, and used commercially**; the only requirement is to **retain the copyright notice and this license text** in copies (i.e., attribution). Linking the repository when using the project satisfies this requirement.

---

🇨🇳 中文版：[README.md.cn](README.md.cn)
