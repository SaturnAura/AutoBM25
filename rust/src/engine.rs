//! Vectorised BM25 / BM25+ scoring over the inverted index, with the accumulator
//! machinery reused across queries (no per-query allocation, no full-array clear).

use std::cmp::Ordering;

use crate::corpus::Corpus;
use crate::params::Params;
use crate::text::TokenBuffer;

/// Scratch space for one query at a time; one per thread.
pub struct Scorer {
    pub scores: Vec<f64>,
    pub stamp: Vec<u32>,
    pub epoch: u32,
    pub touched: Vec<u32>,
    /// Reusable candidate buffer (doc indices) for ranked retrieval.
    pub cand: Vec<u32>,
}

impl Scorer {
    pub fn new(n: usize) -> Scorer {
        Scorer {
            scores: vec![0.0; n],
            stamp: vec![0; n],
            epoch: 0,
            touched: Vec::new(),
            cand: Vec::new(),
        }
    }

    /// Start a new query: zero the accumulator entries the previous query used
    /// (only those, so no O(n) clear per query) and bump the epoch so the stamp
    /// array no longer marks them.
    pub fn begin(&mut self) -> u32 {
        let n = self.touched.len();
        for i in 0..n {
            let doc = self.touched[i] as usize;
            self.scores[doc] = 0.0;
        }
        self.touched.clear();
        if self.epoch == u32::MAX {
            for s in self.stamp.iter_mut() {
                *s = 0;
            }
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
        self.epoch
    }
}

/// Ranking order: score descending, then doc index ascending (the Python
/// implementation's tie-break, applied here deterministically).
pub fn cmp_rank(a: &(u32, f64), b: &(u32, f64)) -> Ordering {
    let by_score = match b.1.partial_cmp(&a.1) {
        Some(o) => o,
        None => Ordering::Equal,
    };
    by_score.then(a.0.cmp(&b.0))
}

/// Reduce a candidate list to the best `k` in ranking order.
pub fn rank_top(cand: &mut Vec<(u32, f64)>, k: usize) {
    let k = k.min(cand.len());
    if k == 0 {
        cand.clear();
        return;
    }
    if cand.len() > k {
        cand.select_nth_unstable_by(k - 1, cmp_rank);
        cand.truncate(k);
    }
    cand.sort_unstable_by(cmp_rank);
}

/// Ranking order over doc indices, with scores looked up in a shared array.
pub fn cmp_scored(a: u32, b: u32, scores: &[f64]) -> Ordering {
    let sa = scores[a as usize];
    let sb = scores[b as usize];
    let by_score = match sb.partial_cmp(&sa) {
        Some(o) => o,
        None => Ordering::Equal,
    };
    by_score.then(a.cmp(&b))
}

/// In-place ranking of doc indices backed by a score array; the grid search reuses
/// one candidate buffer for every (query, parameter) pair, so nothing allocates here.
pub fn rank_docs(cand: &mut Vec<u32>, scores: &[f64], k: usize) {
    let k = k.min(cand.len());
    if k == 0 {
        cand.clear();
        return;
    }
    if cand.len() > k {
        cand.select_nth_unstable_by(k - 1, |&a, &b| cmp_scored(a, b, scores));
        cand.truncate(k);
    }
    cand.sort_unstable_by(|&a, &b| cmp_scored(a, b, scores));
}

pub struct Engine<'a> {
    pub corpus: &'a Corpus,
    pub params: Params,
}

impl<'a> Engine<'a> {
    pub fn new(corpus: &'a Corpus, params: Params) -> Engine<'a> {
        Engine { corpus, params }
    }

    pub fn set_params(&mut self, params: Params) {
        self.params = params;
    }

    pub fn idf(&self, df: u32) -> f64 {
        let n = self.corpus.n_docs() as f64;
        let df = df as f64;
        if self.params.idf_type == "smoothed" {
            ((n + 1.0) / (df + 1.0)).ln()
        } else {
            ((n - df + 0.5) / (df + 0.5)).ln()
        }
    }

    /// Accumulate the BM25 score of every posting of `query`'s terms into `sc`.
    ///
    /// Query terms are visited in first-occurrence order and posting lists in
    /// ascending doc order, which reproduces the floating-point summation order of
    /// the Python implementation (`qtf.items()` + `np.add.at`).
    pub fn accumulate(&self, tb: &TokenBuffer, sc: &mut Scorer) {
        let epoch = sc.begin();
        let (terms, counts) = tb.counted();
        let k1 = self.params.k1;
        let b = self.params.b;
        let k3 = self.params.k3;
        let delta = self.params.delta;
        let avgdl = self.corpus.avgdl;

        for (i, term) in terms.iter().enumerate() {
            let term_id = match self.corpus.term_index.get(*term) {
                Some(&id) => id,
                None => continue,
            };
            let postings = &self.corpus.postings[term_id as usize];
            let idf = self.idf(postings.len() as u32);
            let qcount = counts[i];
            let mut qmul = 1.0f64;
            if k3 > 0.0 && qcount > 1 {
                qmul = (qcount as f64 * (k3 + 1.0)) / (qcount as f64 + k3);
            }
            let boost = if delta > 0.0 { idf * delta } else { 0.0 };

            for &(doc, tf) in postings.iter() {
                let dl = self.corpus.doc_len[doc as usize] as f64;
                let tf_f = tf as f64;
                let denom = if avgdl > 0.0 {
                    tf_f + k1 * ((1.0 - b) + (b * dl) / avgdl)
                } else {
                    tf_f + k1
                };
                let mut part = ((idf * tf_f) * (k1 + 1.0)) / denom;
                if qmul != 1.0 {
                    part *= qmul;
                }
                if boost != 0.0 {
                    part += boost;
                }
                let di = doc as usize;
                if sc.stamp[di] != epoch {
                    sc.stamp[di] = epoch;
                    sc.touched.push(doc);
                }
                sc.scores[di] += part;
            }
        }
    }

    /// Top-`top_k` (doc_idx, score) for one query.
    pub fn search(&self, tb: &TokenBuffer, top_k: usize, sc: &mut Scorer) -> Vec<(u32, f64)> {
        if self.corpus.n_docs() == 0 || top_k == 0 || tb.is_empty() {
            return Vec::new();
        }
        self.accumulate(tb, sc);
        let mut cand: Vec<(u32, f64)> = sc
            .touched
            .iter()
            .filter_map(|&d| {
                let s = sc.scores[d as usize];
                if s != 0.0 {
                    Some((d, s))
                } else {
                    None
                }
            })
            .collect();
        rank_top(&mut cand, top_k);
        cand
    }
}
