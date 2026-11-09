# AutoBM25 — Zero-Label, Zero-Tuning BM25 Hyperparameter Adaptation

[![PyPI](https://img.shields.io/pypi/v/autobm25.svg)](https://pypi.org/project/autobm25/)
[![Python](https://img.shields.io/pypi/pyversions/autobm25.svg)](https://pypi.org/project/autobm25/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/SaturnAura/AutoBM25/blob/main/LICENSE)
[![Rust MSRV](https://img.shields.io/badge/rust-1.71%2B-orange.svg)](https://github.com/SaturnAura/AutoBM25/blob/main/rust/Cargo.toml)
[![CI](https://github.com/SaturnAura/AutoBM25/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SaturnAura/AutoBM25/actions/workflows/ci.yml)

> [Chinese README](https://github.com/SaturnAura/AutoBM25/blob/main/README_CN.md)

![AutoBM25 logo](https://raw.githubusercontent.com/SaturnAura/AutoBM25/main/logo/logo.png)

> Feed the corpus's **statistical features** into an **O(1) parameter dictionary** and get the optimal BM25 hyperparameters (k1 / b / k3 / δ / IDF) directly; fall back to interpretable heuristics when nothing matches. Verified on 16 BEIR datasets: **16/16 never worse than the BM25 defaults, NDCG@10 +18.0% median / +30.5% mean, 92.0% leave-one-corpus-out dictionary hit rate**.

> 😄 Maybe we should call it **BM26**? — if the hyperparameters can adapt, so can the version number (just kidding, BM25 is BM25).

---

## Why

BM25 is the most widely used ranking model in information retrieval, but its **optimal hyperparameters vary dramatically across corpora** — short texts, long texts, code, legal documents, Q&A communities each need different k1/b values. In practice, everyone uses the fixed defaults `k1=1.2, b=0.75`, because tuning requires a labeled validation set, and **new corpora rarely have labels**.

AutoBM25 takes a different path: **optimal parameters are a function of the corpus's statistical structure, and statistical structure can be measured without any labels.** Measure 21 normalized dictionary features, look up the parameters — zero labels, zero tuning, constant time.

## Highlights

- 🗂️ **21 dictionary features** (15 corpus + 6 query-side) across length distribution, term-frequency redundancy, vocabulary growth, and query statistics — each with an explicit mechanism hypothesis
- ⚡ **O(1) parameter dictionary**: nearest-neighbor lookup from feature vectors to optimal parameters, with **out-of-distribution (OOD) guard** and **heuristic fallback**; 48 entries ship with the repo (16 benchmark corpora + 32 cross-domain blends), each validated to be no worse than the defaults on the corpus it came from
- 🔧 **k3 query-term-frequency saturation**: ablation-verified, up to **+19.9%** on long-query scenarios
- 🧩 **Corpus mixing**: combine existing corpora into cross-domain blends to fill feature-space gaps; dictionary hit rate **70.6% → 92.0%**
- 🚀 **Scales to 8.84M documents** (msmarco); self-built numpy-vectorized BM25/BM25+ engine with streaming stratified subsampling that keeps all relevant annotations
- 🦀 **Rust/PyO3 acceleration**: the tokenize-and-index stage is available as a compiled extension, **5.2–12.4× faster one-pass end-to-end** on the benchmark below, with identical scores (optional; pure Python still works)

## Results (16 BEIR datasets)

**BM25 defaults vs AutoBM25 (adaptive) — MRR@10 / NDCG@10 / Recall@100**

Each cell shows `BM25 defaults → AutoBM25` on the same corpus, queries and relevance
judgements (defaults: k1=1.2, b=0.75, k3=0, δ=0, rsj IDF). All 16 datasets improve;
none is worse than the default it is compared with.

| Dataset          | MRR@10              | NDCG@10             | Recall@100          |
| ---------------- | ------------------: | ------------------: | ------------------: |
| arguana          | 0.2036 → 0.2411     | 0.3158 → 0.3654     | 0.9090 → 0.9573     |
| climate-fever    | 0.2480 → 0.3049     | 0.1778 → 0.2241     | 0.4551 → 0.5306     |
| dbpedia-entity   | 0.6674 → 0.6919     | 0.3863 → 0.4158     | 0.4779 → 0.5152     |
| fever            | 0.5983 → 0.8467     | 0.6105 → 0.8322     | 0.9007 → 0.9536     |
| fiqa             | 0.1727 → 0.2050     | 0.1387 → 0.1624     | 0.3429 → 0.3799     |
| hotpotqa         | 0.8131 → 0.8376     | 0.6478 → 0.6803     | 0.7700 → 0.7830     |
| msmarco          | 0.4024 → 0.4162     | 0.4400 → 0.4548     | 0.7490 → 0.7530     |
| nfcorpus         | 0.4600 → 0.4740     | 0.2633 → 0.2738     | 0.1983 → 0.2027     |
| nq               | 0.1594 → 0.1957     | 0.1774 → 0.2266     | 0.5937 → 0.6587     |
| scidocs          | 0.2228 → 0.2596     | 0.1215 → 0.1444     | 0.2936 → 0.3260     |
| scifact          | 0.5033 → 0.5574     | 0.5388 → 0.5945     | 0.8116 → 0.8334     |
| trec-covid       | 0.5936 → 0.7832     | 0.3322 → 0.5011     | 0.0573 → 0.0764     |
| trec-covid-beir  | 0.6107 → 0.7982     | 0.3394 → 0.5113     | 0.0575 → 0.0761     |
| trec-covid-v2    | 0.4079 → 0.7941     | 0.2127 → 0.5041     | 0.0495 → 0.0933     |
| vihealthqa       | 0.3824 → 0.3921     | 0.3676 → 0.3758     | 0.6247 → 0.6289     |
| webis-touche2020 | 0.4931 → 0.7170     | 0.2226 → 0.3901     | 0.4336 → 0.5250     |
| **Average**      | **0.4337 → 0.5322** | **0.3308 → 0.4160** | **0.4828 → 0.5183** |

Relative NDCG@10 gain over the 16 corpora: **+18.0% median, +30.5% mean** (worst
+2.2%, best +137.0%).

How those numbers are produced, because it matters when reading them:

- Each dictionary entry is selected by evaluating a candidate pool on the **full**
  query set of its source corpus, and the BM25 defaults are always in that pool. An
  entry therefore cannot be worse than the defaults on the corpus it was built from,
  and the table measures the shipped prediction path end to end (dictionary lookup,
  heuristic fallback on a miss).
- Five corpora (dbpedia-entity, climate-fever, fever, hotpotqa, msmarco) are
  subsampled to 200k–300k documents. The sampler keeps **every** annotated relevant
  document, so absolute values are not comparable with full-corpus BEIR leaderboards;
  both numbers in a cell come from the same sample, queries and judgements.
- For a corpus that is *not* in the dictionary, parameters come from the nearest
  entry or the heuristics. Leaving the target's own entry out, the dictionary still
  answers **92.0%** of the 50 targets (the 16 corpora plus 34 cross-domain blends).

## Quick Start

```bash
pip install autobm25
```

For a source checkout, use `pip install .`. If Cargo is available, the installer
tries to build the optional Rust extension; otherwise it installs the fully
functional pure-Python backend.

Then hand it a corpus - a plain list of strings is all it takes, there is no file format to learn:

```python
from autobm25 import AutoBM25

corpus = [
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Automatic parameter tuning improves retrieval effectiveness.",
]
retriever = AutoBM25()
retriever.index(corpus)                 # a plain list of strings is enough — no files needed
print(retriever.params)                 # hyperparameters chosen for this corpus
retriever.search("parameter tuning", top_k=5)
```

`corpus` can be anything list-shaped: strings, `{"id", "text"}` records, a column of
a dataframe, documents you fetched from an API. Nothing is written to disk and no
directory layout is required. Pass `queries=[...]` to `index()` as well and the
query-side statistics (query length, query/term overlap, idf) join the same feature
vector.

Documents already in files? `autobm25 --dataset dataset/XXX` runs the same pipeline
over a directory, and `--interactive` turns it into a prompt - the CLI is a
convenience wrapper over this API, not a requirement. `autobm25 --demo` needs no files
at all.

The CLI prints the hyperparameters chosen for the dataset (for example,
`k1=0.5 b=0.495 k3=0.7 δ=0.97 idf=smoothed`).

The parameter dictionary is bundled inside the package (`autobm25/data/param_dictionary.json`), so retrieval works out of the box after installation — no need to run any experiments first.

Want to see it run before wiring up your own data? `python -m autobm25 --demo` (or the
`autobm25 --demo` console script) indexes a small built-in corpus - ten invented
documents about brewing, bicycles and retrieval - and prints the hyperparameters it
chose for that corpus together with the ranking for a handful of queries, with no
dataset directory involved. `autobm25 --demo --interactive` then drops you into a
prompt where you can type your own queries against the same corpus, and
`examples/quickstart.py` is the same thing as a copy-paste script over a made-up
support-desk corpus.

## Rust acceleration (PyO3)

The retrieval core is also implemented in Rust and exposed as a **PyO3** extension
module, so the hot loops run as compiled code while the API, the feature
definitions and the rankings stay exactly as documented above.

The extension is optional. Published platform wheels include it, while a source
install compiles it when Cargo is available and falls back to the pure-Python
implementation when it is not. `autobm25.BACKEND` reports which one is live, and
`AUTOBM25_NO_RUST=1` forces the Python path.

### Which stages are covered

The whole pipeline is: load -> tokenize + index -> features -> predict -> search.
The extension covers the three middle stages plus scoring; the two Python stages are
the ones where Python is simply the right tool.

| Stage                                                 | Implementation | Cost                                                                                                        |
| ----------------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------- |
| loading documents and queries                         | Python         | I/O + `json` parsing, seconds to tens of seconds on multi-GB files; this is why the loader is *not* in Rust |
| tokenize + inverted index (`build_index`)             | **Rust**       | one pass over the corpus; also collects the feature statistics on the way                                   |
| feature vector (`extract_features`)                   | **Rust**       | read off that same pass instead of re-tokenizing                                                            |
| parameter prediction (dictionary lookup + heuristics) | Python         | O(1): 48 entries x 21 features, **0.1-0.5 ms per corpus**, cached config/dictionary                         |
| `search` / `search_many`                              | **Rust**       | posting-list accumulation, GIL released, one scorer per worker thread                                       |
| evaluation metrics (CLI batch mode)                   | Python         | numpy over the returned rankings, milliseconds                                                              |

The loader stays in Python because it is I/O plus JSON parsing, and on the multi-GB
corpora it is the dominant wall-clock cost - a Rust rewrite would not change the
disk reads. Prediction stays in Python for a different reason: it is a constant-time
lookup (0.1-0.5 ms against a 3 s index build), so moving it would buy nothing.

### What the Rust extension replaces

The bottleneck is not scoring — that part is already vectorized numpy. It is a Python
loop over every token in the corpus with a dictionary operation per token, **run
twice**: once to collect the feature statistics and once to build the inverted index.

| Stage                    | Python                                                                                   | Rust                                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| tokenize + term counting | `str.lower().split()` plus a dict operation per token                                    | one pass over the corpus; terms live in a flat table, so only the first occurrence of a term hashes its string |
| feature statistics       | re-tokenizes every document, and keeps a per-document dict for **all** of them in memory | read off the index build — the corpus is tokenized **once**, not twice                                         |
| inverted index           | one `np.asarray` per distinct term over Python-built posting lists                       | postings are built during that same pass                                                                       |
| BM25 scoring             | vectorized, but `np.add.at` scatter-add plus a full-array sort per query                 | posting-list accumulation, no per-query allocation, GIL released, one scorer per worker thread                 |

`index()` therefore tokenizes the corpus once instead of twice, which is the single
largest change. The pure-Python `BM25Engine` is unchanged and still available.

### Do end users need Rust? No.

PyO3 is a **build-time** tool: the Rust code is compiled once into an ordinary
Python extension module, and the compiled result is what ships. Anyone who
installs the package just imports it — there is no compiler, no toolchain and no
extra step on their side:

```python
import autobm25                       # picks up the compiled extension if present

retriever = autobm25.AutoBM25()       # Rust-backed
retriever.index(corpus)               # one tokenizing pass
retriever.search("parameter tuning", top_k=5)

print(autobm25.BACKEND)               # "rust"
```

That is what a wheel gives you: `pip install <wheel>` (or `pip install autobm25` from
an index, where pip picks the wheel matching the OS/CPU) and nothing else, because the
extension is already inside. `abi3-py39` is what makes the matrix manageable - one
binary per **platform** serves CPython 3.9 and every later version, so wheels are
built per OS/architecture (`win_amd64`, `manylinux_x86_64`, `macosx_arm64`, …) rather
than per Python version. A single universal wheel is not possible: a compiled
extension is platform-specific by definition, which is also why a source checkout
with no compiled extension present builds a `py3-none-any` pure-Python wheel.

Nothing here is ever fatal. `autobm25/_accel.py` imports the extension when it is
there and falls back to the Python engine when it is not, so `import autobm25`
cannot fail because of a missing or mismatched binary — it only runs ~7× slower.
`AUTOBM25_NO_RUST=1` forces the Python backend at runtime,
`AUTOBM25_SKIP_RUST_BUILD=1` skips the compile step during an install, and
`autobm25.backend_info()` explains which backend is live.

From a source checkout it comes down to `cargo`: with a toolchain on PATH,
`pip install .` compiles the extension for you, and `python rust/build_pyext.py`
does the same thing by hand (`--check` also verifies that it imports). Without a
toolchain you still get a fully working pure-Python package. Set
`AUTOBM25_REQUIRE_RUST=1` for release builds that must fail rather than silently
omit the accelerator. Toolchains are tried in order - the default first, then any
installed `*-windows-gnu` one - and `--toolchain <name>` pins one explicitly.

### Measured speedup

Measured on 2026-11-09 on Windows with 12 logical cores, CPython 3.11.5,
numpy 1.26.4 and scipy 1.12.0. Same machine, same data and process,
`top_k=100`; the Python column is the same package with `AUTOBM25_NO_RUST=1`.

| Dataset       | Docs    | Tokens | Queries | Build index + features      | Search (all queries)        | End-to-end                  |
| ------------- | ------: | -----: | ------: | --------------------------: | --------------------------: | --------------------------: |
| arguana       | 8,674   | 1.45M  | 1,406   | 2.27 s → 0.24 s (**9.6×**)  | 3.87 s → 0.22 s (**17.6×**) | 6.38 s → 0.52 s (**12.4×**) |
| scifact       | 5,183   | 1.11M  | 1,109   | 1.16 s → 0.13 s (**8.8×**)  | 0.39 s → 0.06 s (**6.2×**)  | 1.59 s → 0.23 s (**6.8×**)  |
| nfcorpus      | 3,633   | 0.85M  | 3,237   | 0.97 s → 0.11 s (**9.3×**)  | 0.35 s → 0.07 s (**4.9×**)  | 1.50 s → 0.29 s (**5.2×**)  |
| trec-covid-v2 | 129,179 | 27.2M  | 50      | 41.27 s → 4.22 s (**9.8×**) | 0.37 s → 0.05 s (**7.1×**)  | 42.88 s → 5.26 s (**8.1×**) |

*End-to-end* = load + parameter prediction + index + features + search. The split
path (`extract_features` and `build_index` called separately, each tokenizing the
corpus) is **3.7–8.9×** faster.

Reading the table: the gap generally widens with corpus size because Python's
per-token dictionary work is linear in tokens. Scoring gains most on long-query
workloads (arguana, 17.6×) where the old `np.add.at` scatter-add is slowest, and
the batch path uses every core with the GIL released.

### Parity

Running both backends over the same corpus and diffing the feature vector, the chosen
hyperparameters, the full ranked list of every query and MRR@10 / NDCG@10 /
Recall@100 gives the same answer either way. On scifact, nfcorpus, arguana and
trec-covid-v2:

- feature values agree to about `1e-11` — they come from different floating-point summation
  orders over identical numbers — and the chosen hyperparameters are identical;
- every ranked list is bitwise identical, except where documents **tie** at the
  `top_k` cutoff: `np.argpartition` returns an arbitrary member of a tie group,
  while Rust applies the tie-break the Python code documents (lowest document index
  wins). Scores stay bitwise equal, MRR@10 and NDCG@10 are unaffected, and
  Recall@100 can move in the 4th decimal when the tied document at rank 100 is
  judged relevant in one list but not the other.

## Limitations

- **Out-of-distribution (OOD)**: an entry is validated on the corpus it was built
  from, not on yours. With the target's own entry removed, the dictionary still
  answers 75% of the 16 corpora above — but it is then predicting from a *different*
  corpus, and that is not free: in this leave-one-corpus-out setting 12/16 improve
  (mean +17.0%, median +7.7%) while 4 get worse than the defaults (-8% to -23%:
  fiqa, arguana, climate-fever, vihealthqa — three of which miss the dictionary
  entirely and land on the heuristics). The OOD guard and the heuristic fallback
  limit the damage; they do not remove it. If your corpus is not represented in the
  dictionary, validate the returned parameters before trusting them.
- **Non-English corpora**: a few features are language-dependent (English stopword
  density, stopword-filtered term statistics). Vietnamese vihealthqa is in the
  dictionary and lands at +2.2%, but for a language the dictionary has never seen,
  validate the returned parameters before trusting them.

## Project Structure

```
autobm25/
├── autobm25/             # the pip package: retrieval API (engine, features, dictionary, CLI) + data/
├── rust/                 # Rust core + PyO3 bindings (python rust/build_pyext.py)
├── tests/                # pytest API, loader, metrics, dictionary and parity tests
├── .github/workflows/    # three-platform CI and Trusted Publishing
├── examples/             # runnable library examples (quickstart.py, ...)
├── logo/
├── config.yaml           # heuristic coefficients
├── setup.py              # compiles the extension into the wheel (metadata is in pyproject.toml)
├── pyproject.toml        # pip packaging
├── requirements.txt
└── README.md / README_CN.md
```

## License

[MIT License](LICENSE): the code can be freely **copied, modified, and used commercially**; the only requirement is to **retain the copyright notice and this license text** in copies (i.e., attribution). Linking the repository when using the project satisfies this requirement.

---

[Chinese README](https://github.com/SaturnAura/AutoBM25/blob/main/README_CN.md)
