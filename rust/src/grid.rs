//! Parallel grid search (the "upper bound" experiment from the research pipeline).
//!
//! Same plan-based design as `research/evaluator.py::grid_search`: the corpus is
//! indexed once and each query is flattened into contiguous score arrays, so a
//! parameter combination only costs one pass over those arrays. Here the
//! combinations are additionally spread over all cores.

use crate::corpus::Corpus;
use crate::engine::{rank_docs, Scorer};
use crate::json::{self, Json};
use crate::loader::Query;
use crate::metrics::{query_metrics, RelMap};
use crate::params::GridSpace;
use crate::stats;
use crate::text::TokenBuffer;

#[derive(Clone)]
pub struct Combo {
    pub k1: f64,
    pub b: f64,
    pub k3: f64,
    pub delta: f64,
    pub idf_type: String,
}

impl Combo {
    pub fn to_json(&self) -> Json {
        json::obj(vec![
            ("k1", Json::num(self.k1)),
            ("b", Json::num(self.b)),
            ("k3", Json::num(self.k3)),
            ("delta", Json::num(self.delta)),
            ("idf_type", Json::str(self.idf_type.clone())),
        ])
    }
}

#[derive(Clone)]
pub struct ComboResult {
    pub combo: Combo,
    pub mrr: f64,
    pub ndcg: f64,
    pub recall: f64,
    pub num_queries: usize,
}

impl ComboResult {
    pub fn metric(&self, name: &str) -> f64 {
        match name {
            "mrr@10" => self.mrr,
            "recall@100" => self.recall,
            _ => self.ndcg,
        }
    }

    pub fn to_json(&self) -> Json {
        json::obj(vec![
            ("params", self.combo.to_json()),
            ("mrr@10", Json::num(self.mrr)),
            ("ndcg@10", Json::num(self.ndcg)),
            ("recall@100", Json::num(self.recall)),
            ("num_queries_evaluated", Json::Int(self.num_queries as i64)),
        ])
    }
}

/// One query flattened into contiguous arrays: all postings of all of its terms,
/// with the per-term IDF already expanded.
pub struct Plan {
    pub docs: Vec<u32>,
    pub tf: Vec<f64>,
    pub dl: Vec<f64>,
    pub qtf: Vec<f64>,
    pub idf_rsj: Vec<f64>,
    pub idf_smooth: Vec<f64>,
}

pub fn build_plan(corpus: &Corpus, query: &str, tb: &mut TokenBuffer) -> Plan {
    tb.refill(query);
    let (terms, counts) = tb.counted();
    let n = corpus.n_docs() as f64;
    let mut plan = Plan {
        docs: Vec::new(),
        tf: Vec::new(),
        dl: Vec::new(),
        qtf: Vec::new(),
        idf_rsj: Vec::new(),
        idf_smooth: Vec::new(),
    };
    for (i, term) in terms.iter().enumerate() {
        let term_id = match corpus.term_index.get(*term) {
            Some(&id) => id,
            None => continue,
        };
        let postings = &corpus.postings[term_id as usize];
        let df = postings.len() as f64;
        let idf_rsj = ((n - df + 0.5) / (df + 0.5)).ln();
        let idf_smooth = ((n + 1.0) / (df + 1.0)).ln();
        let qcount = counts[i] as f64;
        for &(doc, tf) in postings.iter() {
            plan.docs.push(doc);
            plan.tf.push(tf as f64);
            plan.dl.push(corpus.doc_len[doc as usize] as f64);
            plan.qtf.push(qcount);
            plan.idf_rsj.push(idf_rsj);
            plan.idf_smooth.push(idf_smooth);
        }
    }
    plan
}

fn score_plan(plan: &Plan, combo: &Combo, avgdl: f64, sc: &mut Scorer) {
    let epoch = sc.begin();
    let (k1, b, k3, delta) = (combo.k1, combo.b, combo.k3, combo.delta);
    let idf = if combo.idf_type == "smoothed" { &plan.idf_smooth } else { &plan.idf_rsj };
    for i in 0..plan.docs.len() {
        let tf = plan.tf[i];
        let dl = plan.dl[i];
        let idf_v = idf[i];
        let denom = if avgdl > 0.0 {
            tf + k1 * ((1.0 - b) + (b * dl) / avgdl)
        } else {
            tf + k1
        };
        let mut part = ((idf_v * tf) * (k1 + 1.0)) / denom;
        if k3 > 0.0 {
            part *= (plan.qtf[i] * (k3 + 1.0)) / (plan.qtf[i] + k3);
        }
        if delta > 0.0 {
            part += delta * idf_v;
        }
        let di = plan.docs[i] as usize;
        if sc.stamp[di] != epoch {
            sc.stamp[di] = epoch;
            sc.touched.push(plan.docs[i]);
        }
        sc.scores[di] += part;
    }
}

fn evaluate_combo(
    combo: &Combo,
    plans: &[Plan],
    qs: &[Query],
    rel: &RelMap,
    avgdl: f64,
    top_k: usize,
    sc: &mut Scorer,
) -> ComboResult {
    let mut mrr: Vec<f64> = Vec::new();
    let mut ndcg: Vec<f64> = Vec::new();
    let mut recall: Vec<f64> = Vec::new();
    for (plan, q) in plans.iter().zip(qs.iter()) {
        let query_rel = match rel.get(&q.qid) {
            Some(r) => r,
            None => continue,
        };
        if query_rel.rels.is_empty() {
            continue;
        }
        score_plan(plan, combo, avgdl, sc);
        sc.cand.clear();
        for &doc in sc.touched.iter() {
            if sc.scores[doc as usize] != 0.0 {
                sc.cand.push(doc);
            }
        }
        rank_docs(&mut sc.cand, &sc.scores, top_k);
        let (rr, nd, rc) = query_metrics(&sc.cand, query_rel);
        mrr.push(rr);
        ndcg.push(nd);
        recall.push(rc);
    }
    ComboResult {
        combo: combo.clone(),
        mrr: if mrr.is_empty() { 0.0 } else { stats::mean(&mrr) },
        ndcg: if ndcg.is_empty() { 0.0 } else { stats::mean(&ndcg) },
        recall: if recall.is_empty() { 0.0 } else { stats::mean(&recall) },
        num_queries: mrr.len(),
    }
}

/// Parameter combinations in `itertools.product` order: the last axis (`idf_type`)
/// varies fastest, exactly like the Python grid search.
pub fn enumerate_combos(space: &GridSpace) -> Vec<Combo> {
    let mut out: Vec<Combo> = Vec::new();
    for &k1 in &space.k1 {
        for &b in &space.b {
            for &k3 in &space.k3 {
                for &delta in &space.delta {
                    for idf in &space.idf_type {
                        out.push(Combo {
                            k1,
                            b,
                            k3,
                            delta,
                            idf_type: idf.clone(),
                        });
                    }
                }
            }
        }
    }
    out
}

pub struct GridOutcome {
    pub json: Json,
    pub results: Vec<ComboResult>,
    pub best_index: usize,
    pub num_queries_used: usize,
}

pub fn grid_search(
    corpus: &Corpus,
    queries: &[Query],
    rel: &RelMap,
    space: &GridSpace,
    metric: &str,
    top_k: usize,
    limit_queries: Option<usize>,
    threads: usize,
) -> GridOutcome {
    let qs: &[Query] = match limit_queries {
        Some(l) if l < queries.len() => &queries[..l],
        _ => queries,
    };
    let avgdl = corpus.avgdl;
    let n_docs = corpus.n_docs();

    let mut tb = TokenBuffer::new();
    let mut plans: Vec<Plan> = Vec::with_capacity(qs.len());
    for q in qs {
        plans.push(build_plan(corpus, &q.query, &mut tb));
    }

    let combos = enumerate_combos(space);
    let total = combos.len();
    let mut results: Vec<ComboResult> = Vec::with_capacity(total);

    if total > 0 {
        let threads = threads.max(1).min(total);
        let chunk = (total + threads - 1) / threads;
        let n_chunks = (total + chunk - 1) / chunk;
        let mut slots: Vec<Vec<ComboResult>> = vec![Vec::new(); n_chunks];
        let plans_ref: &[Plan] = &plans;
        std::thread::scope(|scope| {
            let mut rest: &mut [Vec<ComboResult>] = slots.as_mut_slice();
            for chunk_combos in combos.chunks(chunk) {
                let (head, tail) = rest.split_at_mut(1);
                rest = tail;
                let slot = &mut head[0];
                scope.spawn(move || {
                    let mut sc = Scorer::new(n_docs);
                    for combo in chunk_combos {
                        let r = evaluate_combo(
                            combo, plans_ref, qs, rel, avgdl, top_k, &mut sc,
                        );
                        slot.push(r);
                    }
                });
            }
        });
        results = slots.into_iter().flatten().collect();
    }

    let mut best_index = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (i, r) in results.iter().enumerate() {
        let v = r.metric(metric);
        if v > best_value {
            best_value = v;
            best_index = i;
        }
    }

    let best_score = match results.get(best_index) {
        Some(b) => json::obj(vec![
            ("mrr@10", Json::num(b.mrr)),
            ("ndcg@10", Json::num(b.ndcg)),
            ("recall@100", Json::num(b.recall)),
        ]),
        None => json::obj(vec![
            ("mrr@10", Json::num(0.0)),
            ("ndcg@10", Json::num(0.0)),
            ("recall@100", Json::num(0.0)),
        ]),
    };
    let best_params = match results.get(best_index) {
        Some(b) => b.combo.to_json(),
        None => Json::Null,
    };

    let json = json::obj(vec![
        ("metric", Json::str(metric)),
        ("num_combos", Json::Int(results.len() as i64)),
        ("num_queries_used", Json::Int(qs.len() as i64)),
        ("best_params", best_params),
        ("best_score", best_score),
        (
            "results",
            Json::Arr(results.iter().map(|r| r.to_json()).collect()),
        ),
    ]);

    GridOutcome { json, results, best_index, num_queries_used: qs.len() }
}
