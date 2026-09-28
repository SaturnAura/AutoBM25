"""Index msmarco with AutoBM25 and search it - the way an ordinary user would.

This is a *sample script*, not part of the package. Everything it does with the
library is the three lines marked "user code" below; the rest is msmarco plumbing
(file layout, timing, printing) that you would swap for your own.

    python run_msmarco.py                        # 200k-document subsample (default, quick)
    python run_msmarco.py --queries 8 --top-k 5  # print 8 queries, 5 hits each
    python run_msmarco.py --evaluate 200         # + BM25 defaults vs AutoBM25 on 200 queries
    python run_msmarco.py --max-docs 0           # the whole 8.8M-document corpus

`--max-docs 0` streams the entire `corpus.jsonl` (3.4 GB on disk, roughly 15-20 GB
of RAM once tokenized+indexed); the default subsamples instead, keeping every
annotated relevant document so the evaluation stays meaningful.
"""

import argparse
import os
import time

from autobm25 import AutoBM25, backend_info
from autobm25.loader import load_dataset, load_dataset_subsampled
from autobm25.metrics import evaluate_engine
from autobm25.rule_predictor import load_config

# Where the corpus is looked for when --dataset is not given, relative to this file
# so the script works from any working directory. msmarco ships in the BEIR layout:
# corpus.jsonl + queries.jsonl + qrels/*.tsv.
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_CANDIDATES = (os.path.join(HERE, "dataset_en", "msmarco"),
                      os.path.join(HERE, "dataset", "msmarco"))

# Building a doc_id -> text map costs one dict entry per document; above this many
# documents the previews are skipped by default so a full-corpus run stays in RAM.
SNIPPET_LIMIT = 2_000_000


def resolve_dataset(explicit):
    if explicit:
        if not os.path.isdir(explicit):
            raise SystemExit(f"--dataset {explicit} is not a directory")
        return explicit
    for candidate in DATASET_CANDIDATES:
        if os.path.exists(os.path.join(candidate, "corpus.jsonl")):
            return candidate
    raise SystemExit(
        "msmarco not found. Pass --dataset /path/to/msmarco, or put the corpus at "
        + " or ".join(DATASET_CANDIDATES)
    )


def load_corpus(path, max_docs, seed):
    """Read the corpus the way a user would: one list of documents, one of queries.

    For an untouched BEIR/standard directory `AutoBM25.from_dataset(path,
    max_docs=...)` does exactly this in one call; it is spelled out here because the
    sample queries and the evaluation below want the same lists.
    """
    if max_docs and os.path.exists(os.path.join(path, "corpus.jsonl")):
        return load_dataset_subsampled(path, max_docs, seed=seed)
    return load_dataset(path)


def previews(docs, enabled):
    """doc_id -> first line of the document, for printing hits."""
    if not enabled:
        return None
    return {doc["id"]: " ".join(doc["text"].split())[:110] for doc in docs}


def print_hits(retriever, query, top_k, preview):
    hits = retriever.search(query, top_k=top_k)
    print(f'  query: "{query}"')
    if not hits:
        print("    (no match)")
        return
    for rank, (doc_id, score) in enumerate(hits, 1):
        tail = f"  {preview[doc_id]}..." if preview else ""
        print(f"    {rank:>2}. {score:8.4f}  doc {doc_id}{tail}")


def compare_params(engine, queries, qrels, params, top_k=100):
    """Evaluate one parameter set on an already-built index."""
    engine.set_params(k1=params["k1"], b=params["b"], k3=params["k3"],
                      delta=params["delta"], idf_type=params["idf_type"])
    return evaluate_engine(engine, queries, qrels, top_k=top_k)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="msmarco directory (BEIR layout)")
    ap.add_argument("--max-docs", type=int, default=200_000,
                    help="subsample the corpus to this many documents; 0 = use all of it")
    ap.add_argument("--seed", type=int, default=42, help="subsampling seed")
    ap.add_argument("--queries", type=int, default=5, help="how many sample queries to run")
    ap.add_argument("--top-k", type=int, default=10, help="hits printed per query")
    ap.add_argument("--query-pool", type=int, default=1000,
                    help="queries whose statistics join the feature vector; -1 = all")
    ap.add_argument("--evaluate", type=int, default=0,
                    help="also score BM25 defaults vs AutoBM25 on this many queries")
    ap.add_argument("--no-snippets", action="store_true", help="never build the preview map")
    args = ap.parse_args()

    # A little progress reporting, so a 3 GB load does not look like a hang.
    total_steps = 2 + int(bool(args.queries)) + int(bool(args.evaluate))
    done_steps = 0

    def step(title):
        nonlocal done_steps
        done_steps += 1
        print(f"[{done_steps}/{total_steps}] {title}")

    path = resolve_dataset(args.dataset)
    mode = f"subsampled to {args.max_docs:,} documents" if args.max_docs else "full corpus"
    step(f"loading {path} ({mode})")
    t0 = time.perf_counter()
    docs, queries, qrels = load_corpus(path, args.max_docs, args.seed)
    load_seconds = time.perf_counter() - t0
    if not docs:
        raise SystemExit(f"{path} has no documents (expected corpus.jsonl or docs.jsonl)")
    print(f"      {len(docs):,} documents / {len(queries):,} queries / {len(qrels):,} qrels"
          f"  in {load_seconds:.1f} s")

    step(f"indexing {len(docs):,} documents")
    # Queries are optional; pass the ones you intend to search and their statistics
    # (length, term overlap, idf) join the same feature vector.
    query_pool = queries if args.query_pool < 0 else queries[:args.query_pool]
    t0 = time.perf_counter()
    # --------------------------------------------------------------- user code
    # This is the whole integration: hand over the corpus (plus your queries, if you
    # have them - they only add query-side statistics to the feature vector) and
    # start searching. No labels, no tuning, no parameter to pick.
    retriever = AutoBM25()
    retriever.index(docs, queries=query_pool)
    # ---------------------------------------------------------------------------
    index_seconds = time.perf_counter() - t0

    info = backend_info()
    print(f"      done in {index_seconds:.1f} s")
    print(f"      backend : {info['backend']} ({info.get('module') or info.get('reason')})")
    print(f"      params  : {retriever.params}")

    sample = queries[:args.queries] if args.queries else []
    use_previews = not args.no_snippets and len(docs) <= SNIPPET_LIMIT
    if sample:
        preview = previews(docs, use_previews)
        if not use_previews and not args.no_snippets:
            print(f"      (previews skipped: {len(docs):,} documents is above the "
                  f"{SNIPPET_LIMIT:,} limit, pass --no-snippets to silence this)")
        step(f"searching {len(sample)} queries from queries.jsonl")
        t0 = time.perf_counter()
        for query in sample:
            print_hits(retriever, query["query"], args.top_k, preview)
        print(f"      {len(sample)} queries in {time.perf_counter() - t0:.2f} s")

    if args.evaluate:
        eval_queries = queries[:args.evaluate]
        defaults = load_config()["default_params"]
        step(f"evaluating on {len(eval_queries)} queries "
             f"(BM25 defaults vs AutoBM25, MRR@10 / NDCG@10 / Recall@100)")
        t0 = time.perf_counter()
        before = compare_params(retriever.engine, eval_queries, qrels, defaults)
        after = compare_params(retriever.engine, eval_queries, qrels, retriever.params)
        print(f"      defaults : mrr={before['mrr@10']:.4f} ndcg={before['ndcg@10']:.4f} "
              f"recall={before['recall@100']:.4f}")
        print(f"      AutoBM25 : mrr={after['mrr@10']:.4f} ndcg={after['ndcg@10']:.4f} "
              f"recall={after['recall@100']:.4f}")
        if before["ndcg@10"]:
            gain = (after["ndcg@10"] / before["ndcg@10"] - 1) * 100
            print(f"      NDCG@10  : {gain:+.1f}%")
        print(f"      ({time.perf_counter() - t0:.1f} s)")

    print()
    print("Swap in your own corpus: build the same list of documents from wherever they")
    print('live - [{"id": ..., "text": ...}] or plain strings - and call')
    print('    retriever = AutoBM25(); retriever.index(docs); retriever.search("...")')
    print("Nothing else about this script is AutoBM25-specific.")


if __name__ == "__main__":
    main()
