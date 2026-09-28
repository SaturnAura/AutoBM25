"""autobm25 command-line entry point.

Usage:
  autobm25 --demo                                # built-in demo corpus, no files needed
  autobm25 --dataset dataset/XXX                 # batch retrieval: run all queries in queries.jsonl,
                                                 #   write results to results.jsonl, evaluate if qrels exist
  autobm25 --dataset dataset/XXX --interactive   # interactive retrieval
  autobm25 --demo --interactive                  # type queries against the demo corpus
"""

import argparse
import json
import os

from .bm25_engine import BM25Engine
from .feature_extractor import extract_features
from .loader import load_dataset, load_dataset_subsampled
from .metrics import evaluate_engine
from .rule_predictor import load_config, predict, predict_with_dictionary


def _load(args):
    if args.max_docs and os.path.exists(os.path.join(args.dataset, "corpus.jsonl")):
        return load_dataset_subsampled(args.dataset, args.max_docs)
    return load_dataset(args.dataset)


def _choose_params(features, args):
    return predict(features) if args.no_dict else predict_with_dictionary(features)


def _log_params(name, docs, params):
    print(
        f"[AutoBM25] dataset={name} docs={len(docs)} | adaptive params: "
        f"k1={params['k1']} b={params['b']} k3={params['k3']} "
        f"delta={params['delta']} idf={params['idf_type']} ({params['model_variant']})"
    )


def _index_and_predict(docs, queries, args):
    """Build the engine and pick the hyperparameters.

    On the Rust backend the index build already produced the corpus statistics, so
    the feature vector comes for free instead of costing a second tokenizing pass.
    """
    engine = BM25Engine().build_index(docs)
    if engine.uses_rust:
        features = engine.features(queries)
    else:
        features = extract_features(docs, queries)
    params = _choose_params(features, args)
    print(f"[AutoBM25] backend: {engine.explain()}")
    return engine, params


def _set_engine_params(engine, params):
    engine.set_params(
        k1=params.get("k1"),
        b=params.get("b"),
        k3=params.get("k3"),
        delta=params.get("delta"),
        idf_type=params.get("idf_type"),
    )


def cmd_batch(args):
    name = os.path.basename(os.path.normpath(args.dataset))
    docs, queries, qrels = _load(args)
    if not docs:
        raise SystemExit(f"[AutoBM25] {args.dataset} has no documents (need docs.jsonl or corpus.jsonl)")
    if not queries:
        raise SystemExit(
            f"[AutoBM25] {args.dataset} has no queries.jsonl; use --interactive to type queries."
        )
    engine, params = _index_and_predict(docs, queries, args)
    _log_params(name, docs, params)
    _set_engine_params(engine, params)
    top_k = args.topk or 100

    results_path = os.path.join(args.dataset, "results.jsonl")
    # One batched search for the whole query set: the Rust backend runs it in
    # parallel with the GIL released, the Python backend loops.
    ranked = engine.search_many([q["query"] for q in queries], top_k=top_k)
    with open(results_path, "w", encoding="utf-8") as f:
        for q, hits in zip(queries, ranked):
            f.write(
                json.dumps(
                    {"qid": q["qid"],
                     "results": [{"rank": i, "doc_id": d, "score": s}
                                 for i, (d, s) in enumerate(hits, 1)]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[AutoBM25] retrieved {len(queries)} queries -> {results_path}")

    if qrels:
        default_params = load_config()["default_params"]
        _set_engine_params(engine, default_params)
        default_metrics = evaluate_engine(engine, queries, qrels, top_k=100)
        _set_engine_params(engine, params)
        adaptive_metrics = evaluate_engine(engine, queries, qrels, top_k=100)
        eval_path = os.path.join(args.dataset, "evaluation.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump({"dataset": name, "adaptive_params": params,
                       "default_params": default_params,
                       "default": default_metrics, "adaptive": adaptive_metrics},
                      f, ensure_ascii=False, indent=2)
        print(f"[AutoBM25] evaluation ({len(queries)} queries) -> {eval_path}")
        print(f"[AutoBM25]   default : {default_metrics}")
        print(f"[AutoBM25]   adaptive: {adaptive_metrics}")
    else:
        print("[AutoBM25] no qrels found; evaluation skipped (add qrels.jsonl or qrels/*.tsv to enable).")


def cmd_interactive(args):
    name = os.path.basename(os.path.normpath(args.dataset))
    docs, queries, _ = _load(args)
    if not docs:
        raise SystemExit(f"[AutoBM25] {args.dataset} has no documents (need docs.jsonl or corpus.jsonl)")
    _run_interactive(name, docs, queries, args)


def cmd_demo(args):
    """The built-in corpus: the library API end to end, without any dataset files."""
    from .demo import DEMO_DOCS, demo_queries, run_demo

    if not args.interactive:
        run_demo(top_k=args.topk or 3, use_dictionary=not args.no_dict)
        print()
        print("[AutoBM25] try your own corpus: examples/quickstart.py, or "
              "autobm25 --demo --interactive to type queries against this one")
        return
    docs = [{"id": d["id"], "text": d["text"]} for d in DEMO_DOCS]
    queries = [{"query": q} for q in demo_queries()]
    _run_interactive("demo (built-in)", docs, queries, args)


def _run_interactive(name, docs, queries, args):
    engine, params = _index_and_predict(docs, queries, args)
    _log_params(name, docs, params)
    _set_engine_params(engine, params)
    top_k = args.topk or 10
    print("[AutoBM25] interactive mode: type a query, press Enter; type exit/quit to quit")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            break
        hits = engine.search(q, top_k=top_k)
        if not hits:
            print("  (no match)")
            continue
        print(f"[AutoBM25] query \"{q}\" -> Top {len(hits)} results (higher score = more relevant):")
        print(f"  {'rank':>4}  {'score':>12}  doc_id")
        for i, (doc_id, score) in enumerate(hits, 1):
            print(f"  {i:>4}  {score:>12.4f}  {doc_id}")


def main():
    parser = argparse.ArgumentParser(
        description="AutoBM25: zero-label BM25 hyperparameter adaptation (batch / interactive)"
    )
    parser.add_argument("--dataset", help="dataset directory, e.g. dataset/fiqa")
    parser.add_argument("--demo", action="store_true",
                        help="run the built-in demo corpus (no dataset directory needed)")
    parser.add_argument("--interactive", action="store_true",
                        help="interactive retrieval; default is batch retrieval over queries.jsonl")
    parser.add_argument("--topk", type=int, default=None,
                        help="number of results (batch default 100, interactive default 10)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="subsample huge corpora (keeps all relevant annotations)")
    parser.add_argument("--no-dict", action="store_true",
                        help="skip the parameter dictionary, use heuristics only")
    args = parser.parse_args()
    if args.demo:
        cmd_demo(args)
        return
    if not args.dataset:
        parser.error("--dataset is required (or run autobm25 --demo for the built-in example)")
    if args.interactive:
        cmd_interactive(args)
    else:
        cmd_batch(args)


if __name__ == "__main__":
    main()
