//! Statistical feature extraction, mirroring
//! `autobm25/feature_extractor.py::extract_features`.
//!
//! Reads the per-document statistics produced by `Corpus::build`, so the corpus is
//! never tokenized a second time.

use crate::corpus::Corpus;
use crate::json::Json;
use crate::loader::Query;
use crate::stats;
use crate::text::TokenBuffer;

const HEAPS_POINTS: usize = 50;

/// Ordered feature vector (insertion order matches the Python dict order).
pub struct Features {
    pub items: Vec<(String, f64)>,
}

impl Features {
    pub fn new() -> Features {
        Features { items: Vec::with_capacity(24) }
    }

    pub fn push(&mut self, key: &str, value: f64) {
        self.items.push((key.to_string(), value));
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.items.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    pub fn get_or(&self, key: &str, default: f64) -> f64 {
        self.get(key).unwrap_or(default)
    }

    /// Same shape (and key order) as the Python feature dict.
    pub fn to_json(&self) -> Json {
        let int_keys = ["doc_count", "num_docs", "total_tokens", "vocab_size"];
        let fields = self
            .items
            .iter()
            .map(|(k, v)| {
                if int_keys.contains(&k.as_str()) {
                    (k.clone(), Json::Int(*v as i64))
                } else {
                    (k.clone(), Json::Num(*v))
                }
            })
            .collect();
        Json::Obj(fields)
    }
}

impl Default for Features {
    fn default() -> Self {
        Features::new()
    }
}

fn distinct_count(xs: &[f64]) -> usize {
    let mut bits: Vec<u64> = xs.iter().map(|x| x.to_bits()).collect();
    bits.sort_unstable();
    bits.dedup();
    bits.len()
}

pub fn extract(corpus: &Corpus, queries: &[Query]) -> Features {
    let n = corpus.n_docs();
    let n_f = n as f64;
    let mut f = Features::new();

    // ---- Dimension 1: document length distribution (drives b) ----
    let lens = corpus.doc_lens_f64();
    let avgdl = if lens.is_empty() { 0.0 } else { stats::mean(&lens) };
    let std_len = if lens.len() > 1 { stats::std_sample(&lens) } else { 0.0 };
    let cv_len = if avgdl > 0.0 { std_len / avgdl } else { 0.0 };
    let len_skew = if lens.len() > 2 { stats::skew_unbiased(&lens) } else { 0.0 };
    let (len_p90, len_p10) = if lens.is_empty() {
        (0.0, 0.0)
    } else {
        let mut sorted = lens.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (stats::percentile_sorted(&sorted, 0.9), stats::percentile_sorted(&sorted, 0.1))
    };
    let len_ratio_p90_p10 = len_p90 / len_p10.max(1.0);

    // ---- Dimension 2: in-document term frequency (drives k1) ----
    let avg_ttr = if corpus.ttrs.is_empty() { 0.0 } else { stats::mean(&corpus.ttrs) };
    let avg_max_tf = if corpus.max_tfs.is_empty() { 0.0 } else { stats::mean(&corpus.max_tfs) };
    let length_tf_corr = if lens.len() > 2
        && stats::std_pop(&lens) > 0.0
        && stats::std_pop(&corpus.mean_tfs) > 0.0
    {
        stats::pearson(&lens, &corpus.mean_tfs)
    } else {
        0.0
    };

    // ---- Dimension 3: global vocabulary distribution (drives the IDF variant) ----
    let vocab_size = corpus.vocab_size();
    let hapax_ratio = if vocab_size > 0 {
        let hapax = corpus.vocab_freq.iter().filter(|&&v| v == 1).count();
        hapax as f64 / vocab_size as f64
    } else {
        0.0
    };

    // Zipf: log(freq) = log(C) - alpha * log(rank); |slope| is alpha.
    let mut zipf_alpha = 0.0;
    if vocab_size >= 2 {
        let mut freqs: Vec<f64> = corpus.vocab_freq.iter().map(|&v| v as f64).collect();
        freqs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let ranks: Vec<f64> = (1..=freqs.len()).map(|r| r as f64).collect();
        let xs: Vec<f64> = ranks.iter().map(|r| r.ln()).collect();
        let ys: Vec<f64> = freqs.iter().map(|v| v.ln()).collect();
        let slope = stats::linregress_slope(&xs, &ys);
        zipf_alpha = if slope.is_finite() { slope.abs() } else { 0.0 };
    }

    // Heaps: log(V) = log(K) + beta * log(n), sampled on a log-spaced grid.
    let mut points: Vec<(u64, u64)> = corpus.heaps_points.clone();
    if points.len() > HEAPS_POINTS && points.len() > 1 {
        let grid = stats::logspace(0.0, (points.len() as f64).log10(), HEAPS_POINTS);
        let mut idxs: Vec<usize> = Vec::with_capacity(grid.len());
        for v in grid {
            let rounded = stats::round_half_even(v);
            let idx = rounded as i64 - 1;
            let clamped = idx.max(0).min(points.len() as i64 - 1);
            idxs.push(clamped as usize);
        }
        idxs.sort_unstable();
        idxs.dedup();
        points = idxs.iter().map(|&i| corpus.heaps_points[i]).collect();
    }
    let mut hx: Vec<f64> = Vec::with_capacity(points.len());
    let mut hy: Vec<f64> = Vec::with_capacity(points.len());
    for (tokens, vocab) in &points {
        if *tokens > 0 {
            hx.push((*tokens as f64).ln());
            hy.push((*vocab as f64).ln());
        }
    }
    let heaps_beta = if hx.len() >= 2 && distinct_count(&hx) >= 2 {
        let slope = stats::linregress_slope(&hx, &hy);
        if slope.is_finite() { slope } else { 0.0 }
    } else {
        0.0
    };

    let stopword_density = if corpus.total_tokens > 0 {
        corpus.stopword_tokens as f64 / corpus.total_tokens as f64
    } else {
        0.0
    };

    f.push("doc_count", n_f);
    f.push("num_docs", n_f);
    f.push("total_tokens", corpus.total_tokens as f64);
    f.push("avgdl", avgdl);
    f.push("std_len", std_len);
    f.push("cv_len", cv_len);
    f.push("len_skew", len_skew);
    f.push("len_p90", len_p90);
    f.push("len_p10", len_p10);
    f.push("len_ratio_p90_p10", len_ratio_p90_p10);
    f.push("avg_ttr", avg_ttr);
    f.push("avg_max_tf", avg_max_tf);
    f.push("length_tf_corr", length_tf_corr);
    f.push("hapax_ratio", hapax_ratio);
    f.push("vocab_size", vocab_size as f64);
    f.push("heaps_beta", heaps_beta);
    f.push("stopword_density", stopword_density);
    f.push("zipf_alpha", zipf_alpha);

    // ---- Dimension 4: query-side features (only when queries are present) ----
    if !queries.is_empty() {
        let mut q_lens: Vec<f64> = Vec::new();
        let mut q_max_tfs: Vec<f64> = Vec::new();
        let mut q_idfs: Vec<f64> = Vec::new();
        let mut q_with_repeat = 0usize;
        let mut q_count = 0usize;
        let mut q_oov = 0usize;
        let mut total_q_tokens = 0usize;

        let mut tb = TokenBuffer::new();
        for q in queries {
            tb.refill(&q.query);
            if tb.is_empty() {
                continue;
            }
            q_lens.push(tb.len() as f64);
            q_count += 1;
            total_q_tokens += tb.len();

            let (_terms, counts) = tb.counted();
            let mut max_count = 0u32;
            for &c in &counts {
                if c > max_count {
                    max_count = c;
                }
            }
            q_max_tfs.push(max_count as f64);
            if max_count > 1 {
                q_with_repeat += 1;
            }

            for tok in tb.tokens() {
                let df = corpus.df(tok);
                if df > 0 {
                    let df_f = df as f64;
                    q_idfs.push(((n_f - df_f + 0.5) / (df_f + 0.5)).ln());
                } else {
                    q_oov += 1;
                }
            }
        }

        f.push("avg_query_len", if q_lens.is_empty() { 0.0 } else { stats::mean(&q_lens) });
        f.push(
            "avg_query_max_tf",
            if q_max_tfs.is_empty() { 0.0 } else { stats::mean(&q_max_tfs) },
        );
        f.push(
            "query_repeat_ratio",
            if q_count > 0 { q_with_repeat as f64 / q_count as f64 } else { 0.0 },
        );
        f.push(
            "query_oov_ratio",
            if total_q_tokens > 0 { q_oov as f64 / total_q_tokens as f64 } else { 0.0 },
        );
        f.push("query_idf_mean", if q_idfs.is_empty() { 0.0 } else { stats::mean(&q_idfs) });
        f.push(
            "query_idf_std",
            if q_idfs.len() > 1 { stats::std_sample(&q_idfs) } else { 0.0 },
        );
    }

    f
}
