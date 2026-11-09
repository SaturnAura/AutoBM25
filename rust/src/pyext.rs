//! PyO3 bindings: the `autobm25._autobm25_rs` extension module.
//!
//! The Python package keeps its pure-Python implementation as a fallback; this
//! module only replaces the three stages that actually cost time:
//!
//! 1. `Index::new` - one tokenizing pass that builds the inverted index *and*
//!    the per-document statistics the features are derived from. The Python
//!    pipeline tokenizes the whole corpus twice (`extract_features` and then
//!    `BM25Engine.build_index`); here it happens once.
//! 2. `Index::features` - the 21 dictionary features (15 corpus + 6 query-side),
//!    plus three document-count/token diagnostics, read off that same pass
//!    instead of re-walking every document with a Python-level dict.
//! 3. `Index::search` / `Index::search_many` - BM25 scoring straight over the
//!    posting lists, with the GIL released and one scorer per worker thread.
//!
//! Everything a caller can observe (feature values, tie-breaking, top-k order)
//! matches the Python implementation bit-for-bit; `_bench/compare.py` asserts it.

use std::collections::HashSet;
use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::corpus::Corpus;
use crate::engine::{Engine, Scorer};
use crate::features::{self, Features};
use crate::loader::{Doc, Query};
use crate::params::Params;
use crate::text::TokenBuffer;

/// Feature keys the Python dict stores as `int` rather than `float`.
const INT_FEATURE_KEYS: [&str; 4] = ["doc_count", "num_docs", "total_tokens", "vocab_size"];

fn features_to_dict<'py>(py: Python<'py>, f: &Features) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for (key, value) in &f.items {
        if INT_FEATURE_KEYS.contains(&key.as_str()) {
            dict.set_item(key.as_str(), *value as i64)?;
        } else {
            dict.set_item(key.as_str(), *value)?;
        }
    }
    Ok(dict)
}

fn build_corpus(
    py: Python<'_>,
    ids: Vec<String>,
    texts: Vec<String>,
    stopwords: Option<Vec<String>>,
) -> PyResult<Corpus> {
    if ids.len() != texts.len() {
        return Err(PyValueError::new_err(format!(
            "ids ({}) and texts ({}) must have the same length",
            ids.len(),
            texts.len()
        )));
    }
    if texts.is_empty() {
        return Err(PyValueError::new_err(
            "docs must not be empty (need at least one document)",
        ));
    }
    let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
    let docs: Vec<Doc> = ids
        .into_iter()
        .zip(texts)
        .map(|(id, text)| Doc { id, text })
        .collect();
    // The corpus build is the longest stage in the pipeline; drop the GIL so a
    // threaded Python caller is not serialized behind it.
    Ok(py.allow_threads(move || Corpus::build(&docs, &stop)))
}

/// Per-thread `Scorer` scratch space, parked between calls.
///
/// Allocating a scorer costs `O(n_docs)`, which is fine once but not once per
/// query on a multi-million-document corpus, so single-query calls recycle one
/// from here and the parallel batch path hands each worker thread its own.
struct ScorerPool {
    n_docs: usize,
    free: Mutex<Vec<Scorer>>,
}

impl ScorerPool {
    fn new(n_docs: usize) -> ScorerPool {
        ScorerPool { n_docs, free: Mutex::new(Vec::new()) }
    }

    fn take(&self) -> Scorer {
        match self.free.lock() {
            Ok(mut parked) => parked.pop().unwrap_or_else(|| Scorer::new(self.n_docs)),
            Err(_) => Scorer::new(self.n_docs),
        }
    }

    fn put(&self, scorer: Scorer) {
        if let Ok(mut parked) = self.free.lock() {
            // Bounded so a burst of threads does not pin memory forever.
            if parked.len() < 32 {
                parked.push(scorer);
            }
        }
    }
}

/// An indexed corpus plus the BM25 hyperparameters currently in effect.
#[pyclass(module = "autobm25._autobm25_rs", name = "Index")]
pub struct Index {
    corpus: Corpus,
    params: Params,
    pool: ScorerPool,
}

#[pymethods]
impl Index {
    /// `Index(ids, texts, stopwords=None)` - tokenize once and build the index.
    #[new]
    #[pyo3(signature = (ids, texts, stopwords=None))]
    fn new(
        py: Python<'_>,
        ids: Vec<String>,
        texts: Vec<String>,
        stopwords: Option<Vec<String>>,
    ) -> PyResult<Index> {
        let corpus = build_corpus(py, ids, texts, stopwords)?;
        let pool = ScorerPool::new(corpus.n_docs());
        Ok(Index {
            corpus,
            // Same defaults as `config.yaml`; `set_params` overwrites them.
            params: Params {
                k1: 1.2,
                b: 0.75,
                k3: 0.0,
                delta: 0.0,
                idf_type: "rsj".to_string(),
            },
            pool,
        })
    }

    /// Update the hyperparameters; `None` keeps the current value.
    #[pyo3(signature = (k1=None, b=None, k3=None, delta=None, idf_type=None))]
    fn set_params(
        &mut self,
        k1: Option<f64>,
        b: Option<f64>,
        k3: Option<f64>,
        delta: Option<f64>,
        idf_type: Option<String>,
    ) {
        if let Some(v) = k1 {
            self.params.k1 = v;
        }
        if let Some(v) = b {
            self.params.b = v;
        }
        if let Some(v) = k3 {
            self.params.k3 = v;
        }
        if let Some(v) = delta {
            self.params.delta = v;
        }
        if let Some(v) = idf_type {
            self.params.idf_type = v;
        }
    }

    /// Top-`top_k` `(doc_id, score)` pairs for one query, best first.
    #[pyo3(signature = (query, top_k=10))]
    fn search(&self, py: Python<'_>, query: &str, top_k: usize) -> Vec<(String, f64)> {
        let mut scorer = self.pool.take();
        let hits = {
            let corpus = &self.corpus;
            let params = self.params.clone();
            // Move the *borrow* into the closure so the scorer can go back in the
            // pool once the search returns.
            let scorer = &mut scorer;
            py.allow_threads(move || {
                let engine = Engine::new(corpus, params);
                let mut tb = TokenBuffer::new();
                tb.refill(query);
                engine.search(&tb, top_k, scorer)
            })
        };
        self.pool.put(scorer);
        self.collect(hits)
    }

    /// Top-`top_k` results for many queries, returned in input order.
    ///
    /// `threads` defaults to all logical cores; each worker owns a token buffer
    /// and a scorer, and the whole batch runs with the GIL released.
    #[pyo3(signature = (queries, top_k=10, threads=None))]
    fn search_many(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        top_k: usize,
        threads: Option<usize>,
    ) -> Vec<Vec<(String, f64)>> {
        let n = queries.len();
        if n == 0 {
            return Vec::new();
        }
        let default_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let threads = threads.unwrap_or(default_threads).max(1).min(n);

        let corpus = &self.corpus;
        let params = self.params.clone();
        let pool = &self.pool;
        let rankings: Vec<Vec<(u32, f64)>> = py.allow_threads(move || {
            let mut out: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
            let chunk = (n + threads - 1) / threads;
            std::thread::scope(|scope| {
                let mut rest: &mut [Vec<(u32, f64)>] = out.as_mut_slice();
                for batch in queries.chunks(chunk) {
                    let (head, tail) = rest.split_at_mut(batch.len());
                    rest = tail;
                    let worker_params = params.clone();
                    scope.spawn(move || {
                        let engine = Engine::new(corpus, worker_params);
                        let mut tb = TokenBuffer::new();
                        let mut scorer = pool.take();
                        for (query, slot) in batch.iter().zip(head.iter_mut()) {
                            tb.refill(query);
                            *slot = engine.search(&tb, top_k, &mut scorer);
                        }
                        pool.put(scorer);
                    });
                }
            });
            out
        });

        rankings
            .into_iter()
            .map(|hits| self.collect(hits))
            .collect()
    }

    /// The corpus feature vector, computed from the already-built index.
    ///
    /// `queries` adds the optional query-side features (dimension 4).
    #[pyo3(signature = (queries=None))]
    fn features<'py>(
        &self,
        py: Python<'py>,
        queries: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let qs: Vec<Query> = queries
            .unwrap_or_default()
            .into_iter()
            .map(|query| Query { qid: String::new(), query })
            .collect();
        let corpus = &self.corpus;
        let feats = py.allow_threads(move || features::extract(corpus, &qs));
        features_to_dict(py, &feats)
    }

    /// Number of indexed documents.
    #[getter]
    fn n_docs(&self) -> usize {
        self.corpus.n_docs()
    }

    /// Number of distinct terms (posting lists).
    #[getter]
    fn vocab_size(&self) -> usize {
        self.corpus.vocab_size()
    }

    /// Alias of `vocab_size`, matching `len(engine.postings)` on the Python engine.
    #[getter]
    fn n_terms(&self) -> usize {
        self.corpus.vocab_size()
    }

    /// Mean document length in tokens.
    #[getter]
    fn avgdl(&self) -> f64 {
        self.corpus.avgdl
    }

    /// Total tokens across the corpus.
    #[getter]
    fn total_tokens(&self) -> u64 {
        self.corpus.total_tokens
    }

    /// Document ids, in index order.
    #[getter]
    fn doc_ids(&self) -> Vec<String> {
        self.corpus.doc_ids.clone()
    }

    /// Document frequency of one term (`0` when the term is not in the index).
    fn doc_freq(&self, term: &str) -> u32 {
        self.corpus.df(term)
    }

    /// IDF of one term under the current `idf_type`.
    fn idf(&self, term: &str) -> f64 {
        Engine::new(&self.corpus, self.params.clone()).idf(self.corpus.df(term))
    }

    /// Score a single `(term, doc_idx)` pair - the debugging helper
    /// `BM25Engine.score_doc` exposes on the Python engine.
    fn score_doc(&self, term: &str, doc_idx: usize, idf_val: f64) -> f64 {
        if doc_idx >= self.corpus.doc_len.len() {
            return 0.0;
        }
        let term_id = match self.corpus.term_id(term) {
            Some(id) => id,
            None => return 0.0,
        };
        // Posting lists are built in ascending document order, so this is a lookup.
        let postings = &self.corpus.postings[term_id as usize];
        let tf = match postings.binary_search_by_key(&(doc_idx as u32), |&(doc, _)| doc) {
            Ok(pos) => postings[pos].1 as f64,
            Err(_) => return 0.0,
        };
        let dl = self.corpus.doc_len[doc_idx] as f64;
        let (k1, b, delta, avgdl) =
            (self.params.k1, self.params.b, self.params.delta, self.corpus.avgdl);
        let denom = if avgdl > 0.0 {
            tf + k1 * ((1.0 - b) + (b * dl) / avgdl)
        } else {
            tf + k1
        };
        let tf_part = tf * (k1 + 1.0) / denom;
        if delta > 0.0 {
            idf_val * (tf_part + delta)
        } else {
            idf_val * tf_part
        }
    }

    /// Token counts per document, in index order.
    fn doc_lengths(&self) -> Vec<u32> {
        self.corpus.doc_len.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "<autobm25 Rust Index docs={} terms={} avgdl={:.3} k1={} b={} k3={} delta={} idf={}>",
            self.corpus.n_docs(),
            self.corpus.vocab_size(),
            self.corpus.avgdl,
            self.params.k1,
            self.params.b,
            self.params.k3,
            self.params.delta,
            self.params.idf_type
        )
    }
}

impl Index {
    /// Map internal `(doc_idx, score)` pairs onto document ids.
    fn collect(&self, hits: Vec<(u32, f64)>) -> Vec<(String, f64)> {
        hits.into_iter()
            .map(|(doc, score)| (self.corpus.doc_ids[doc as usize].clone(), score))
            .collect()
    }
}

/// `extract_features(texts, queries=None, stopwords=None)` → feature dict.
///
/// Convenience wrapper for callers that only need the features and not the index;
/// the values are identical to `Index::features` on the same corpus.
#[pyfunction]
#[pyo3(signature = (texts, queries=None, stopwords=None, ids=None))]
fn extract_features<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    queries: Option<Vec<String>>,
    stopwords: Option<Vec<String>>,
    ids: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ids = ids.unwrap_or_else(|| (0..texts.len()).map(|i| i.to_string()).collect());
    let corpus = build_corpus(py, ids, texts, stopwords)?;
    let qs: Vec<Query> = queries
        .unwrap_or_default()
        .into_iter()
        .map(|query| Query { qid: String::new(), query })
        .collect();
    let corpus_ref = &corpus;
    let feats = py.allow_threads(move || features::extract(corpus_ref, &qs));
    features_to_dict(py, &feats)
}

/// `tokenize(text)` → the exact tokens the indexer uses.
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    let mut tb = TokenBuffer::new();
    tb.refill(text);
    tb.tokens().map(|t| t.to_string()).collect()
}

/// Read a stopword list (one word per line, blank lines skipped).
#[pyfunction]
fn load_stopwords(path: &str) -> PyResult<Vec<String>> {
    let set = crate::text::load_stopwords(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("{}: {}", path, e)))?;
    Ok(set.into_iter().collect())
}

/// The extension module itself.
#[pymodule]
fn _autobm25_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(load_stopwords, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
