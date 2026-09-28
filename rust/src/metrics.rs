//! Retrieval metrics (MRR@10 / NDCG@10 / Recall@100), the parallel query runner,
//! and the relevance map used by both evaluation and the grid search.
//!
//! Relevance is keyed by *document index* rather than document id: the grid search
//! scores millions of ranked positions, and integer lookups keep that off the
//! profile. The ideal-DCG term of every query is precomputed once for the same
//! reason.

use std::collections::HashMap;

use crate::engine::{rank_top, Engine, Scorer};
use crate::fx::FxHashMap;
use crate::json::{self, Json};
use crate::loader::{Qrel, Query};
use crate::stats;
use crate::text::TokenBuffer;

pub struct QueryRel {
    /// doc_idx -> graded relevance (only judged-relevant documents)
    pub rels: FxHashMap<u32, i32>,
    /// sum(gain / log2(i+2)) over the ten best possible gains
    pub idcg: f64,
}

impl QueryRel {
    pub fn size(&self) -> usize {
        self.rels.len()
    }
}

pub struct RelMap {
    pub by_qid: HashMap<String, QueryRel>,
}

impl RelMap {
    /// Join qrels onto document indices (`doc_index` maps doc id -> doc_idx).
    pub fn from_qrels(qrels: &[Qrel], doc_index: &FxHashMap<&str, u32>) -> RelMap {
        let mut by_qid: HashMap<String, QueryRel> = HashMap::new();
        for r in qrels {
            if r.relevance <= 0 {
                continue;
            }
            let doc_idx = match doc_index.get(r.doc_id.as_str()) {
                Some(&idx) => idx,
                None => continue,
            };
            by_qid
                .entry(r.qid.clone())
                .or_insert_with(|| QueryRel { rels: FxHashMap::default(), idcg: 0.0 })
                .rels
                .insert(doc_idx, r.relevance);
        }
        for entry in by_qid.values_mut() {
            let mut gains: Vec<i32> = entry.rels.values().copied().collect();
            gains.sort_unstable_by(|a, b| b.cmp(a));
            let mut idcg = 0.0;
            for (i, gain) in gains.iter().take(10).enumerate() {
                idcg += *gain as f64 / (i as f64 + 2.0).log2();
            }
            entry.idcg = idcg;
        }
        RelMap { by_qid }
    }

    pub fn get(&self, qid: &str) -> Option<&QueryRel> {
        self.by_qid.get(qid)
    }

    pub fn is_empty(&self) -> bool {
        self.by_qid.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Metrics {
    pub mrr10: f64,
    pub ndcg10: f64,
    pub recall100: f64,
    pub num_queries: usize,
}

impl Metrics {
    pub fn to_json(&self) -> Json {
        json::obj(vec![
            ("mrr@10", Json::num(self.mrr10)),
            ("ndcg@10", Json::num(self.ndcg10)),
            ("recall@100", Json::num(self.recall100)),
            ("num_queries_evaluated", Json::Int(self.num_queries as i64)),
        ])
    }
}

/// Per-query ranked (doc_idx, score) lists, in query order.
///
/// `threads` chunks the query set; each thread keeps its own token buffer and
/// scorer, which is where most of the parallelism in this project comes from.
pub fn run_queries(
    engine: &Engine,
    queries: &[Query],
    top_k: usize,
    threads: usize,
) -> Vec<Vec<(u32, f64)>> {
    let n = queries.len();
    let mut out: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    if n == 0 {
        return out;
    }
    let threads = threads.max(1).min(n);
    let chunk = (n + threads - 1) / threads;
    std::thread::scope(|scope| {
        let mut rest: &mut [Vec<(u32, f64)>] = out.as_mut_slice();
        for qs in queries.chunks(chunk) {
            let (head, tail) = rest.split_at_mut(qs.len());
            rest = tail;
            scope.spawn(move || {
                let mut tb = TokenBuffer::new();
                let mut sc = Scorer::new(engine.corpus.n_docs());
                for (q, slot) in qs.iter().zip(head.iter_mut()) {
                    tb.refill(&q.query);
                    *slot = engine.search(&tb, top_k, &mut sc);
                }
            });
        }
    });
    out
}

/// MRR@10, NDCG@10 (graded) and Recall@100 for one ranked list of doc indices.
pub fn query_metrics(ranked: &[u32], rel: &QueryRel) -> (f64, f64, f64) {
    let mut rr = 0.0;
    for (i, doc) in ranked.iter().take(10).enumerate() {
        if rel.rels.contains_key(doc) {
            rr = 1.0 / (i as f64 + 1.0);
            break;
        }
    }

    let mut dcg = 0.0;
    for (i, doc) in ranked.iter().take(10).enumerate() {
        let gain = rel.rels.get(doc).copied().unwrap_or(0);
        if gain > 0 {
            dcg += gain as f64 / (i as f64 + 2.0).log2();
        }
    }
    let ndcg = if rel.idcg > 0.0 { dcg / rel.idcg } else { 0.0 };

    let size = rel.size();
    let hits = ranked
        .iter()
        .take(100)
        .filter(|doc| rel.rels.contains_key(doc))
        .count();
    let recall = if size > 0 { hits as f64 / size as f64 } else { 0.0 };
    (rr, ndcg, recall)
}

/// Average metrics over every query that has relevance judgements.
pub fn evaluate(
    engine: &Engine,
    queries: &[Query],
    rel: &RelMap,
    top_k: usize,
    threads: usize,
) -> Metrics {
    let rankings = run_queries(engine, queries, top_k, threads);
    average_metrics(queries, &rankings, rel)
}

pub fn average_metrics(
    queries: &[Query],
    rankings: &[Vec<(u32, f64)>],
    rel: &RelMap,
) -> Metrics {
    let mut mrr: Vec<f64> = Vec::new();
    let mut ndcg: Vec<f64> = Vec::new();
    let mut recall: Vec<f64> = Vec::new();
    for (q, ranked) in queries.iter().zip(rankings.iter()) {
        let query_rel = match rel.get(&q.qid) {
            Some(r) => r,
            None => continue,
        };
        if query_rel.rels.is_empty() {
            continue;
        }
        let docs: Vec<u32> = ranked.iter().map(|(doc, _)| *doc).collect();
        let (rr, nd, rc) = query_metrics(&docs, query_rel);
        mrr.push(rr);
        ndcg.push(nd);
        recall.push(rc);
    }
    Metrics {
        mrr10: if mrr.is_empty() { 0.0 } else { stats::mean(&mrr) },
        ndcg10: if ndcg.is_empty() { 0.0 } else { stats::mean(&ndcg) },
        recall100: if recall.is_empty() { 0.0 } else { stats::mean(&recall) },
        num_queries: mrr.len(),
    }
}

/// Re-rank a candidate list of doc indices (helper for the grid search).
pub fn rank_candidates(cand: &mut Vec<(u32, f64)>, top_k: usize) {
    rank_top(cand, top_k);
}
