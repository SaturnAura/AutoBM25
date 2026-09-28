//! autobm25-rs: Rust rewrite of the AutoBM25 retrieval core.
//!
//! Same pipeline as the Python package - load -> one-pass index+features ->
//! parameter dictionary lookup (OOD guard) with heuristic fallback -> BM25 search
//! -> evaluation - plus the parallel grid search from the research workflow.

#![allow(dead_code)]

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use autobm25::corpus::Corpus;
use autobm25::engine::{Engine, Scorer};
use autobm25::features::{extract, Features};
use autobm25::grid::grid_search;
use autobm25::json::Json;
use autobm25::loader::{load_dataset, load_dataset_subsampled, Dataset, Qrel};
use autobm25::metrics::{evaluate, run_queries, Metrics, RelMap};
use autobm25::params::{
    load_config, predict_with_dictionary, Config, ParamDictionary, Prediction,
};
use autobm25::text::{load_stopwords, TokenBuffer};
// Module paths used with a `json::` / `grid::` / `loader::` prefix further down.
use autobm25::{grid, json, loader};

const USAGE: &str = "\
autobm25-rs - Rust rewrite of the AutoBM25 retrieval core

USAGE:
  autobm25-rs --dataset DIR [--mode batch|interactive|features|eval|grid] [options]

OPTIONS:
  --dataset, -d DIR     dataset directory (docs/queries/qrels.jsonl or BEIR layout)
  --mode, -m NAME       batch (default) | interactive | features | eval | grid
  --interactive, -i     shorthand for --mode interactive
  --topk, -k N          results per query (batch default 100, interactive default 10)
  --max-docs N          subsample huge corpora (keeps every judged-relevant doc)
  --threads, -j N       worker threads (default: all cores)
  --limit-queries N     grid/eval: only use the first N queries
  --grid                eval mode: also run the grid search
  --no-dict             skip the parameter dictionary, use the heuristics only
  --data-dir DIR        directory holding param_dictionary.json / stopwords.txt /
                        config.yaml (default: ./autobm25/data)
  --config FILE         explicit config.yaml path
  --out-dir DIR         where results.jsonl / evaluation.json are written
  --json OUT            features / grid mode: write JSON to this file
  --quiet, -q           suppress progress lines
  --help, -h            show this help";

struct Args {
    dataset: String,
    mode: String,
    topk: Option<usize>,
    max_docs: Option<usize>,
    no_dict: bool,
    threads: usize,
    out_dir: Option<String>,
    data_dir: Option<String>,
    config: Option<String>,
    limit_queries: Option<usize>,
    with_grid: bool,
    json_out: Option<String>,
    quiet: bool,
}

fn take_value(argv: &[String], i: usize, flag: &str) -> Result<String, String> {
    match argv.get(i) {
        Some(v) => Ok(v.clone()),
        None => Err(format!("{} needs a value", flag)),
    }
}

fn parse_usize(text: &str, flag: &str) -> Result<usize, String> {
    text.parse::<usize>().map_err(|_| format!("{}: '{}' is not a number", flag, text))
}

fn parse_args(argv: Vec<String>) -> Result<Args, String> {
    let mut args = Args {
        dataset: String::new(),
        mode: "batch".to_string(),
        topk: None,
        max_docs: None,
        no_dict: false,
        threads: 0,
        out_dir: None,
        data_dir: None,
        config: None,
        limit_queries: None,
        with_grid: false,
        json_out: None,
        quiet: false,
    };
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        match flag.as_str() {
            "--dataset" | "-d" => {
                i += 1;
                args.dataset = take_value(&argv, i, "--dataset")?;
            }
            "--mode" | "-m" => {
                i += 1;
                args.mode = take_value(&argv, i, "--mode")?;
            }
            "--topk" | "-k" => {
                i += 1;
                args.topk = Some(parse_usize(&take_value(&argv, i, "--topk")?, "--topk")?);
            }
            "--max-docs" => {
                i += 1;
                args.max_docs = Some(parse_usize(&take_value(&argv, i, "--max-docs")?, "--max-docs")?);
            }
            "--threads" | "-j" => {
                i += 1;
                args.threads = parse_usize(&take_value(&argv, i, "--threads")?, "--threads")?;
            }
            "--limit-queries" => {
                i += 1;
                args.limit_queries =
                    Some(parse_usize(&take_value(&argv, i, "--limit-queries")?, "--limit-queries")?);
            }
            "--out-dir" => {
                i += 1;
                args.out_dir = Some(take_value(&argv, i, "--out-dir")?);
            }
            "--data-dir" => {
                i += 1;
                args.data_dir = Some(take_value(&argv, i, "--data-dir")?);
            }
            "--config" => {
                i += 1;
                args.config = Some(take_value(&argv, i, "--config")?);
            }
            "--json" => {
                i += 1;
                args.json_out = Some(take_value(&argv, i, "--json")?);
            }
            "--grid" => args.with_grid = true,
            "--no-dict" => args.no_dict = true,
            "--interactive" | "-i" => args.mode = "interactive".to_string(),
            "--quiet" | "-q" => args.quiet = true,
            "--help" | "-h" => {
                println!("{}", USAGE);
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{}'\n\n{}", other, USAGE)),
        }
        i += 1;
    }
    if args.dataset.is_empty() {
        return Err(format!("--dataset is required\n\n{}", USAGE));
    }
    if args.threads == 0 {
        args.threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    }
    Ok(args)
}

fn resolve_data_dir(explicit: Option<&str>) -> PathBuf {
    if let Some(dir) = explicit {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("AUTOBM25_DATA_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    for candidate in ["autobm25/data", "data", "../autobm25/data"] {
        let p = PathBuf::from(candidate);
        if p.join("param_dictionary.json").exists() {
            return p;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for candidate in ["autobm25/data", "../autobm25/data", "../../autobm25/data"] {
                let p = dir.join(candidate);
                if p.join("param_dictionary.json").exists() {
                    return p;
                }
            }
        }
    }
    PathBuf::from("autobm25/data")
}

fn format_secs(d: Duration) -> String {
    format!("{:.3}s", d.as_secs_f64())
}

fn dataset_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string())
}

fn load_dataset_dir(dir: &Path, max_docs: Option<usize>) -> Result<Dataset, String> {
    let has_corpus = loader::exists(dir, "corpus.jsonl");
    if let (true, Some(md)) = (has_corpus, max_docs) {
        load_dataset_subsampled(dir, md, 42).map_err(|e| format!("{}: {}", dir.display(), e))
    } else {
        load_dataset(dir).map_err(|e| format!("{}: {}", dir.display(), e))
    }
}

fn main() {
    let args = match parse_args(std::env::args().skip(1).collect()) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{}", msg);
            std::process::exit(2);
        }
    };
    if let Err(err) = run(&args) {
        eprintln!("[autobm25-rs] error: {}", err);
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    let data_dir = resolve_data_dir(args.data_dir.as_deref());
    let dict_path = data_dir.join("param_dictionary.json");
    let stop_path = data_dir.join("stopwords.txt");
    let config_path = match &args.config {
        Some(p) => PathBuf::from(p),
        None => {
            let p = data_dir.join("config.yaml");
            if p.exists() {
                p
            } else {
                PathBuf::from("config.yaml")
            }
        }
    };
    let config = load_config(&config_path);

    let dictionary = if args.no_dict {
        None
    } else if dict_path.exists() {
        Some(ParamDictionary::load(&dict_path).map_err(|e| format!("dictionary: {}", e))?)
    } else {
        None
    };
    let stopwords = if stop_path.exists() {
        load_stopwords(&stop_path).unwrap_or_default()
    } else {
        HashSet::new()
    };

    let dir = PathBuf::from(&args.dataset);
    match args.mode.as_str() {
        "batch" => cmd_batch(args, &config, dictionary.as_ref(), &stopwords, &dir),
        "interactive" => cmd_interactive(args, &config, dictionary.as_ref(), &stopwords, &dir),
        "features" => cmd_features(args, &config, dictionary.as_ref(), &stopwords, &dir),
        "eval" => cmd_eval(args, &config, dictionary.as_ref(), &stopwords, &dir),
        "grid" => cmd_grid(args, &config, &stopwords, &dir),
        other => Err(format!("unknown mode '{}'", other)),
    }
}

struct Prepared {
    ds: Dataset,
    corpus: Corpus,
    features: Features,
    t_load: Duration,
    t_index: Duration,
    t_features: Duration,
}

fn prepare(
    dir: &Path,
    max_docs: Option<usize>,
    stopwords: &HashSet<String>,
) -> Result<Prepared, String> {
    let t = Instant::now();
    let ds = load_dataset_dir(dir, max_docs)?;
    let t_load = t.elapsed();
    if ds.docs.is_empty() {
        return Err(format!(
            "{} has no documents (need docs.jsonl or corpus.jsonl)",
            dir.display()
        ));
    }
    let t = Instant::now();
    let corpus = Corpus::build(&ds.docs, stopwords);
    let t_index = t.elapsed();
    let t = Instant::now();
    let feats = extract(&corpus, &ds.queries);
    let t_features = t.elapsed();
    Ok(Prepared { ds, corpus, features: feats, t_load, t_index, t_features })
}

fn params_and_prediction(
    feats: &Features,
    config: &Config,
    dict: Option<&ParamDictionary>,
    quiet: bool,
) -> Prediction {
    let prediction = predict_with_dictionary(feats, config, dict);
    if !quiet {
        match &prediction {
            Prediction::Dictionary(hit) => println!(
                "[autobm25-rs] adaptive params: {} (dictionary hit: {}, distance={})",
                hit.params.describe(),
                hit.matches.join(","),
                hit.distance
            ),
            Prediction::Heuristic(p) => {
                println!("[autobm25-rs] adaptive params: {} (heuristic fallback)", p.describe())
            }
        }
    }
    prediction
}

/// Join qrels onto document indices through the freshly built corpus.
fn build_rel_map(corpus: &Corpus, qrels: &[Qrel]) -> RelMap {
    let doc_index = corpus.doc_index_map();
    RelMap::from_qrels(qrels, &doc_index)
}

fn cmd_batch(
    args: &Args,
    config: &Config,
    dict: Option<&ParamDictionary>,
    stopwords: &HashSet<String>,
    dir: &Path,
) -> Result<(), String> {
    let t_start = Instant::now();
    let prepared = prepare(dir, args.max_docs, stopwords)?;
    let ds = prepared.ds;
    let corpus = prepared.corpus;
    let feats = prepared.features;
    let name = dataset_name(dir);
    if !args.quiet {
        println!(
            "[autobm25-rs] dataset={} docs={} queries={} qrels={}",
            name,
            ds.docs.len(),
            ds.queries.len(),
            ds.qrels.len()
        );
    }
    if ds.queries.is_empty() {
        return Err(format!(
            "{} has no queries.jsonl; use --mode interactive to type queries",
            dir.display()
        ));
    }
    let prediction = params_and_prediction(&feats, config, dict, args.quiet);
    let params = prediction.params().clone();

    let top_k = args.topk.unwrap_or(100);
    let engine = Engine::new(&corpus, params.clone());
    let t = Instant::now();
    let rankings = run_queries(&engine, &ds.queries, top_k, args.threads);
    let search_time = t.elapsed();

    let out_dir = match &args.out_dir {
        Some(d) => PathBuf::from(d),
        None => dir.to_path_buf(),
    };
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {}", out_dir.display(), e))?;
    let results_path = out_dir.join("results.jsonl");
    let mut buf = String::new();
    for (i, q) in ds.queries.iter().enumerate() {
        let items: Vec<Json> = rankings[i]
            .iter()
            .enumerate()
            .map(|(rank, (doc, score))| {
                json::obj(vec![
                    ("rank", Json::Int(rank as i64 + 1)),
                    ("doc_id", Json::str(corpus.doc_ids[*doc as usize].clone())),
                    ("score", Json::num(*score)),
                ])
            })
            .collect();
        buf.push_str(
            &json::obj(vec![
                ("qid", Json::str(q.qid.clone())),
                ("results", Json::Arr(items)),
            ])
            .dump(),
        );
        buf.push('\n');
    }
    std::fs::write(&results_path, buf).map_err(|e| format!("{}: {}", results_path.display(), e))?;
    if !args.quiet {
        println!(
            "[autobm25-rs] retrieved {} queries -> {}",
            ds.queries.len(),
            results_path.display()
        );
    }

    if !ds.qrels.is_empty() {
        let rel = build_rel_map(&corpus, &ds.qrels);
        let mut engine = Engine::new(&corpus, config.default_params.clone());
        let default_metrics = evaluate(&engine, &ds.queries, &rel, 100, args.threads);
        engine.set_params(params.clone());
        let adaptive_metrics = evaluate(&engine, &ds.queries, &rel, 100, args.threads);

        let eval_path = out_dir.join("evaluation.json");
        let eval = json::obj(vec![
            ("dataset", Json::str(name)),
            ("adaptive_params", params.to_json()),
            ("default_params", config.default_params.to_json()),
            ("default", default_metrics.to_json()),
            ("adaptive", adaptive_metrics.to_json()),
        ]);
        std::fs::write(&eval_path, eval.dump_pretty())
            .map_err(|e| format!("{}: {}", eval_path.display(), e))?;
        if !args.quiet {
            print_metrics("default", &default_metrics);
            print_metrics("adaptive", &adaptive_metrics);
            println!("[autobm25-rs] evaluation -> {}", eval_path.display());
        }
    } else if !args.quiet {
        println!("[autobm25-rs] no qrels found; evaluation skipped");
    }

    if !args.quiet {
        println!(
            "[autobm25-rs] timings: load={} index={} features={} search={} total={} threads={}",
            format_secs(prepared.t_load),
            format_secs(prepared.t_index),
            format_secs(prepared.t_features),
            format_secs(search_time),
            format_secs(t_start.elapsed()),
            args.threads
        );
        println!(
            "[autobm25-rs] index: {} terms, {} postings, avgdl={:.2}",
            corpus.postings.len(),
            corpus.postings.iter().map(|v| v.len()).sum::<usize>(),
            corpus.avgdl
        );
    }
    Ok(())
}

fn print_metrics(label: &str, m: &Metrics) {
    println!(
        "[autobm25-rs]   {:<8} mrr@10={:.4} ndcg@10={:.4} recall@100={:.4} ({} queries)",
        label, m.mrr10, m.ndcg10, m.recall100, m.num_queries
    );
}

fn cmd_interactive(
    args: &Args,
    config: &Config,
    dict: Option<&ParamDictionary>,
    stopwords: &HashSet<String>,
    dir: &Path,
) -> Result<(), String> {
    let prepared = prepare(dir, args.max_docs, stopwords)?;
    let ds = prepared.ds;
    let corpus = prepared.corpus;
    let feats = prepared.features;
    let prediction = params_and_prediction(&feats, config, dict, args.quiet);
    let engine = Engine::new(&corpus, prediction.params().clone());
    let top_k = args.topk.unwrap_or(10);
    let mut tb = TokenBuffer::new();
    let mut sc = Scorer::new(corpus.n_docs());
    println!(
        "[autobm25-rs] interactive mode (docs={}) - type a query, or exit/quit to stop",
        ds.docs.len()
    );
    let stdin = std::io::stdin();
    loop {
        print!("query> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let query = line.trim();
        if query.is_empty()
            || query.eq_ignore_ascii_case("exit")
            || query.eq_ignore_ascii_case("quit")
            || query.eq_ignore_ascii_case("q")
        {
            break;
        }
        tb.refill(query);
        let hits = engine.search(&tb, top_k, &mut sc);
        if hits.is_empty() {
            println!("  (no match)");
            continue;
        }
        println!("[autobm25-rs] query \"{}\" -> Top {} results:", query, hits.len());
        println!("  {:>4}  {:>12}  doc_id", "rank", "score");
        for (i, (doc, score)) in hits.iter().enumerate() {
            println!(
                "  {:>4}  {:>12.4}  {}",
                i + 1,
                score,
                corpus.doc_ids[*doc as usize]
            );
        }
    }
    Ok(())
}

fn cmd_features(
    args: &Args,
    _config: &Config,
    _dict: Option<&ParamDictionary>,
    stopwords: &HashSet<String>,
    dir: &Path,
) -> Result<(), String> {
    let t = Instant::now();
    let prepared = prepare(dir, args.max_docs, stopwords)?;
    let ds = prepared.ds;
    let feats = prepared.features;
    let text = feats.to_json().dump_pretty();
    match &args.json_out {
        Some(path) => {
            std::fs::write(path, text).map_err(|e| format!("{}: {}", path, e))?;
            println!("[autobm25-rs] features -> {}", path);
        }
        None => println!("{}", text),
    }
    if !args.quiet {
        println!(
            "[autobm25-rs] docs={} queries={} elapsed={}",
            ds.docs.len(),
            ds.queries.len(),
            format_secs(t.elapsed())
        );
    }
    Ok(())
}

fn cmd_eval(
    args: &Args,
    config: &Config,
    dict: Option<&ParamDictionary>,
    stopwords: &HashSet<String>,
    dir: &Path,
) -> Result<(), String> {
    let t = Instant::now();
    let prepared = prepare(dir, args.max_docs, stopwords)?;
    let ds = prepared.ds;
    let corpus = prepared.corpus;
    let feats = prepared.features;
    let name = dataset_name(dir);
    let prediction = params_and_prediction(&feats, config, dict, args.quiet);
    let params = prediction.params().clone();
    let rel = build_rel_map(&corpus, &ds.qrels);
    if rel.is_empty() {
        return Err(format!("{} has no relevance judgements", dir.display()));
    }
    let mut engine = Engine::new(&corpus, config.default_params.clone());
    let default_metrics = evaluate(&engine, &ds.queries, &rel, 100, args.threads);
    engine.set_params(params.clone());
    let adaptive_metrics = evaluate(&engine, &ds.queries, &rel, 100, args.threads);

    let mut fields: Vec<(String, Json)> = Vec::new();
    fields.push(("dataset".to_string(), Json::str(name.clone())));
    fields.push(("default_params".to_string(), config.default_params.to_json()));
    fields.push(("default".to_string(), default_metrics.to_json()));
    fields.push(("adaptive_params".to_string(), params.to_json()));
    fields.push(("adaptive".to_string(), adaptive_metrics.to_json()));
    for (label, base, pred) in [
        ("mrr@10", default_metrics.mrr10, adaptive_metrics.mrr10),
        ("ndcg@10", default_metrics.ndcg10, adaptive_metrics.ndcg10),
        ("recall@100", default_metrics.recall100, adaptive_metrics.recall100),
    ] {
        let improve = if base > 0.0 { (pred - base) / base * 100.0 } else { 0.0 };
        fields.push((format!("improve_vs_default_{}", label), Json::num(improve)));
    }

    if args.with_grid {
        let space = config.grid.clone();
        let outcome = grid_search(
            &corpus,
            &ds.queries,
            &rel,
            &space,
            &space.metric,
            space.top_k,
            args.limit_queries,
            args.threads,
        );
        let best = outcome.results.get(outcome.best_index);
        if let Some(best) = best {
            fields.push((
                "grid_search".to_string(),
                json::obj(vec![
                    ("mrr@10", Json::num(best.mrr)),
                    ("ndcg@10", Json::num(best.ndcg)),
                    ("recall@100", Json::num(best.recall)),
                ]),
            ));
            for (label, base, grid) in [
                ("mrr@10", adaptive_metrics.mrr10, best.mrr),
                ("ndcg@10", adaptive_metrics.ndcg10, best.ndcg),
                ("recall@100", adaptive_metrics.recall100, best.recall),
            ] {
                let reach = if grid > 0.0 { base / grid * 100.0 } else { 0.0 };
                fields.push((format!("reach_grid_{}", label), Json::num(reach)));
            }
            if !args.quiet {
                println!(
                    "[autobm25-rs]   grid best: {} -> {}",
                    best.combo.to_json().dump(),
                    best.to_json().dump()
                );
            }
        }
    }

    let report = Json::Obj(fields);
    let out_dir = match &args.out_dir {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from("results"),
    };
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {}", out_dir.display(), e))?;
    let path = match &args.json_out {
        Some(p) => PathBuf::from(p),
        None => out_dir.join(format!("{}_eval.json", name)),
    };
    std::fs::write(&path, report.dump_pretty())
        .map_err(|e| format!("{}: {}", path.display(), e))?;
    if !args.quiet {
        print_metrics("default", &default_metrics);
        print_metrics("adaptive", &adaptive_metrics);
        println!(
            "[autobm25-rs] report -> {} (elapsed={})",
            path.display(),
            format_secs(t.elapsed())
        );
    }
    Ok(())
}

fn cmd_grid(
    args: &Args,
    config: &Config,
    stopwords: &HashSet<String>,
    dir: &Path,
) -> Result<(), String> {
    let t = Instant::now();
    let prepared = prepare(dir, args.max_docs, stopwords)?;
    let ds = prepared.ds;
    let corpus = prepared.corpus;
    let name = dataset_name(dir);
    let rel = build_rel_map(&corpus, &ds.qrels);
    if rel.is_empty() {
        return Err(format!("{} has no relevance judgements", dir.display()));
    }
    let space = config.grid.clone();
    if !args.quiet {
        println!(
            "[autobm25-rs] grid search: {} combos x {} queries, metric={}, threads={}",
            grid::enumerate_combos(&space).len(),
            args.limit_queries.unwrap_or(ds.queries.len()).min(ds.queries.len()),
            space.metric,
            args.threads
        );
    }
    let outcome = grid_search(
        &corpus,
        &ds.queries,
        &rel,
        &space,
        &space.metric,
        space.top_k,
        args.limit_queries,
        args.threads,
    );
    let out_dir = match &args.out_dir {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from("results"),
    };
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {}", out_dir.display(), e))?;
    let path = match &args.json_out {
        Some(p) => PathBuf::from(p),
        None => out_dir.join(format!("{}_grid.json", name)),
    };
    std::fs::write(&path, outcome.json.dump_pretty())
        .map_err(|e| format!("{}: {}", path.display(), e))?;
    if !args.quiet {
        if let Some(best) = outcome.results.get(outcome.best_index) {
            println!(
                "[autobm25-rs] best: {} -> mrr@10={:.4} ndcg@10={:.4} recall@100={:.4}",
                best.combo.to_json().dump(),
                best.mrr,
                best.ndcg,
                best.recall
            );
        }
        println!(
            "[autobm25-rs] {} combos over {} queries -> {} (elapsed={})",
            outcome.results.len(),
            outcome.num_queries_used,
            path.display(),
            format_secs(t.elapsed())
        );
    }
    Ok(())
}
