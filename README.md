# AutoBM25 — Zero-Label, Zero-Tuning BM25 Hyperparameter Adaptation

![AutoBM25 logo](logo/logo.png)

> Feed the corpus's **statistical features** into an **O(1) parameter dictionary** and get the optimal BM25 hyperparameters (k1 / b / k3 / δ / IDF) directly; fall back to interpretable heuristics when nothing matches. Verified on 16 BEIR datasets: **16/16 never worse than defaults, average NDCG@10 +11.6%, dictionary hit rate 88.9%**.

> 😄 Maybe we should call it **BM26**? — if the hyperparameters can adapt, so can the version number (just kidding, BM25 is BM25).

---

## Why

BM25 is the most widely used ranking model in information retrieval, but its **optimal hyperparameters vary dramatically across corpora** — short texts, long texts, code, legal documents, Q&A communities each need different k1/b values. In practice, everyone uses the fixed defaults `k1=1.2, b=0.75`, because tuning requires a labeled validation set, and **new corpora rarely have labels**.

AutoBM25 takes a different path: **optimal parameters are a function of the corpus's statistical structure, and statistical structure can be measured without any labels.** Measure 18 features, look up the parameters — zero labels, zero tuning, constant time.

## Highlights

- 🗂️ **18 statistical features** (P0/P1/P2): document length distribution, term-frequency redundancy, vocabulary growth, query-side statistics — each with an explicit "mechanism hypothesis" (why it predicts a given parameter)
- ⚡ **O(1) parameter dictionary**: nearest-neighbor lookup from feature vectors to optimal parameters, with **out-of-distribution (OOD) guard** and **heuristic fallback**; 30 entries shipped with the repo, growing with every new grid-searched dataset
- 🔧 **k3 query-term-frequency saturation**: ablation-verified, up to **+19.9%** on long-query scenarios
- 🧩 **Corpus mixing**: combine existing corpora into cross-domain blends to fill feature-space gaps; dictionary hit rate **70.6% → 88.9%**
- 🚀 **Scales to 8.84M documents** (msmarco); self-built numpy-vectorized BM25/BM25+ engine with streaming stratified subsampling that keeps all relevant annotations

## Results (16 BEIR datasets)

**Representative improvements (BEIR datasets, NDCG@10)**

| Dataset | Domain | NDCG@10 gain |
|---|---|---:|
| trec-covid-v2 | COVID-19 scientific literature (129K docs) | **+93.7%** |
| webis-touche2020 | Argument retrieval (383K docs) | **+26.5%** |
| trec-covid / trec-covid-beir | COVID-19 scientific literature (171K docs) | +22.2% |
| fever | Fact verification (5.4M docs) | +19.2% |
| nq | Open-domain QA (2.7M docs) | +17.5% |
| scifact | Scientific claim evidence | +8.2% |
| dbpedia-entity | Entity linking retrieval (4.6M docs) | +7.7% |
| msmarco | Web search (8.8M docs) | +2.4% |
| **Average (16 datasets)** | — | **MRR@10 +9.4% · NDCG@10 +11.6% · Recall@100 +5.6%** |

## Quick Start

```bash
pip install -r requirements.txt

# Batch retrieval: run all queries in dataset/XXX/queries.jsonl,
# write results back to dataset/XXX/results.jsonl; auto-evaluates if qrels exist
python src/main.py --dataset dataset/XXX

# Interactive retrieval: type a query, press Enter, exit with "exit"
python src/main.py --dataset dataset/XXX --interactive
```

On startup a log line shows the hyperparameters chosen for this dataset (e.g., `k1=0.5 b=0.495 k3=0.7 δ=0.97 idf=smoothed`).

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
├── src/                  # User-facing retrieval modules (CLI, engine, features, dictionary, evaluation)
├── dictionary/           # Parameter dictionary param_dictionary.json (30 entries, shipped with the repo)
├── logo/                 # Project logo
├── config.yaml           # Heuristic coefficients, defaults, grid/augmentation/dictionary settings
├── requirements.txt
└── README.md
```

## License

[MIT License](LICENSE): the code can be freely **copied, modified, and used commercially**; the only requirement is to **retain the copyright notice and this license text** in copies (i.e., attribution). Linking the repository when using the project satisfies this requirement.

---

🇨🇳 中文版：[README.md.cn](README.md.cn)
