use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use autobm25::corpus::Corpus;
use autobm25::engine::{Engine, Scorer};
use autobm25::features::Features;
use autobm25::loader::{self, Doc, Query};
use autobm25::metrics::{average_metrics, QueryRel, RelMap};
use autobm25::params::{predict, Config, Params};
use autobm25::text::TokenBuffer;

fn docs() -> Vec<Doc> {
    vec![
        Doc { id: "d1".into(), text: "alpha alpha beta".into() },
        Doc { id: "d2".into(), text: "alpha gamma".into() },
        Doc { id: "d3".into(), text: "delta".into() },
    ]
}

fn params() -> Params {
    Params {
        k1: 1.2,
        b: 0.75,
        k3: 0.0,
        delta: 0.0,
        idf_type: "smoothed".into(),
    }
}

#[test]
fn tokenizer_lowercases_and_uses_python_whitespace() {
    let mut tb = TokenBuffer::new();
    tb.refill("  Alpha\tALPHA\nBeta\u{1c}Gamma  ");

    assert_eq!(tb.tokens().collect::<Vec<_>>(), vec!["alpha", "alpha", "beta", "gamma"]);
    assert_eq!(tb.counted().1, vec![2, 1, 1]);
}

#[test]
fn bm25_ranks_term_frequency_before_length_only_matches() {
    let corpus = Corpus::build(&docs(), &Default::default());
    let engine = Engine::new(&corpus, params());
    let mut tb = TokenBuffer::new();
    let mut scorer = Scorer::new(corpus.n_docs());
    tb.refill("alpha beta");

    let hits = engine.search(&tb, 3, &mut scorer);

    assert_eq!(hits.iter().map(|(doc, _)| *doc).collect::<Vec<_>>(), vec![0, 1]);
    assert!(hits[0].1 > hits[1].1);
}

#[test]
fn equal_scores_use_lowest_document_index_and_top_k_zero_is_empty() {
    let tied = vec![
        Doc { id: "a".into(), text: "alpha".into() },
        Doc { id: "b".into(), text: "alpha".into() },
        Doc { id: "c".into(), text: "gamma".into() },
    ];
    let corpus = Corpus::build(&tied, &Default::default());
    let engine = Engine::new(&corpus, params());
    let mut tb = TokenBuffer::new();
    let mut scorer = Scorer::new(corpus.n_docs());
    tb.refill("alpha");

    let hits = engine.search(&tb, 2, &mut scorer);
    assert_eq!(hits.iter().map(|(doc, _)| *doc).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(hits[0].1, hits[1].1);
    assert!(engine.search(&tb, 0, &mut scorer).is_empty());
}

#[test]
fn heuristic_prediction_is_bounded_and_rounded() {
    let mut features = Features::new();
    features.push("avg_ttr", 1.0);
    features.push("avg_max_tf", 20.0);
    features.push("length_tf_corr", 0.0);
    features.push("cv_len", 0.5);
    features.push("avg_query_max_tf", 2.0);
    features.push("query_repeat_ratio", 0.0);
    features.push("heaps_beta", 0.8);
    features.push("hapax_ratio", 0.0);

    let predicted = predict(&features, &Config::default());

    assert_eq!(predicted.k1, 0.5);
    assert_eq!(predicted.b, 0.35);
    assert_eq!(predicted.delta, 0.5);
    assert_eq!(predicted.k3, 0.5);
    assert_eq!(predicted.idf_type, "smoothed");
}

#[test]
fn metrics_match_their_ranked_lists() {
    let queries = vec![Query { qid: "q1".into(), query: "alpha".into() }];
    let mut rels = autobm25::fx::FxHashMap::default();
    rels.insert(1, 1);
    let mut by_qid = HashMap::new();
    by_qid.insert("q1".into(), QueryRel { rels, idcg: 1.0 });
    let rel = RelMap { by_qid };
    let rankings = vec![vec![(0, 0.9), (1, 0.8)]];

    let metrics = average_metrics(&queries, &rankings, &rel);

    assert_eq!(metrics.mrr10, 0.5);
    assert!((metrics.ndcg10 - 1.0 / 3.0f64.log2()).abs() < 1e-12);
    assert_eq!(metrics.recall100, 1.0);
    assert_eq!(metrics.num_queries, 1);
}

#[test]
fn loader_reads_beir_corpus_and_qrels() {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("autobm25-core-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(dir.join("qrels")).unwrap();
    std::fs::write(
        dir.join("corpus.jsonl"),
        "{\"_id\":\"d1\",\"title\":\"Title\",\"text\":\"alpha\"}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("queries.jsonl"),
        "{\"_id\":\"q1\",\"text\":\"alpha\"}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("qrels").join("test.tsv"),
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
    )
    .unwrap();

    let dataset = loader::load_dataset(&dir).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();

    assert_eq!(dataset.docs[0].id, "d1");
    assert_eq!(dataset.docs[0].text, "Title alpha");
    assert_eq!(dataset.queries[0].qid, "q1");
    assert_eq!(dataset.qrels[0].relevance, 1);
}
