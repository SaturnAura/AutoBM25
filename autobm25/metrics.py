"""检索评估指标：MRR@10 / NDCG@10 / Recall@100。"""

import numpy as np

from .bm25_engine import BM25Engine


def _relevant_map(qrels):
    rel = {}
    for r in qrels:
        if r["relevance"] > 0:
            rel.setdefault(r["qid"], {})[r["doc_id"]] = r["relevance"]
    return rel


def evaluate_engine(engine, queries, qrels, top_k=100):
    """对已 build 的引擎打分，返回平均指标。"""
    rel = _relevant_map(qrels)
    mrr, ndcg, recall = [], [], []
    for q in queries:
        rels = rel.get(q["qid"])
        if not rels:
            continue
        ranked_ids = [doc_id for doc_id, _ in engine.search(q["query"], top_k=top_k)]
        rr = 0.0
        for i, doc_id in enumerate(ranked_ids[:10]):
            if doc_id in rels:
                rr = 1.0 / (i + 1)
                break
        mrr.append(rr)
        dcg = 0.0
        for i, doc_id in enumerate(ranked_ids[:10]):
            gain = rels.get(doc_id, 0)
            if gain > 0:
                dcg += gain / np.log2(i + 2)
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        hit = sum(1 for doc_id in ranked_ids[:100] if doc_id in rels)
        recall.append(hit / len(rels))
    return {
        "mrr@10": float(np.mean(mrr)) if mrr else 0.0,
        "ndcg@10": float(np.mean(ndcg)) if ndcg else 0.0,
        "recall@100": float(np.mean(recall)) if recall else 0.0,
        "num_queries_evaluated": len(mrr),
    }


def evaluate(docs, queries, qrels, params, top_k=100):
    """评估单个参数组合。params: {k1, b, k3, delta, idf_type}"""
    engine = BM25Engine().build_index(docs)
    engine.set_params(
        k1=params.get("k1"),
        b=params.get("b"),
        k3=params.get("k3"),
        delta=params.get("delta"),
        idf_type=params.get("idf_type"),
    )
    return evaluate_engine(engine, queries, qrels, top_k=top_k)
