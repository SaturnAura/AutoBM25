"""k3 试验：判断查询词频饱和项是否带来实际收益。

两种模式：
1. 默认：在已有 grid best 参数基础上扫描 k3（0/0.5/1.2/2/4/8），快速判断；
2. --joint：对指定小数据集跑「完整参数空间 × k3」的联合 grid search（3168 组合），
   验证 k3 与 k1/b 的交互是否改变结论。
"""

import argparse
import json
import os

import numpy as np

from bm25_engine import BM25Engine
from data_loader import load_dataset
from evaluator import _QueryPlan, _query_metrics, _relevant_map, grid_search
from rule_predictor import load_config

K3_VALUES = [0.0, 0.5, 1.2, 2.0, 4.0, 8.0]


def evaluate_with_k3(plans, qs, rel, doc_ids, params, k3, top_k=100):
    k1, b, delta, idf_type = params["k1"], params["b"], params["delta"], params["idf_type"]
    mrr, ndcg, recall = [], [], []
    for plan, q in zip(plans, qs):
        rels = rel.get(q["qid"])
        if not rels:
            continue
        scores = plan.score(k1, b, k3, delta, idf_type)
        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            mrr.append(0.0)
            ndcg.append(0.0)
            recall.append(0.0)
            continue
        k = min(top_k, nonzero.size)
        top = nonzero[np.argpartition(scores[nonzero], -k)[-k:]]
        order = top[np.lexsort((top, -scores[top]))]
        ranked_ids = [doc_ids[int(i)] for i in order]
        rr, ndcg_v, rec = _query_metrics(ranked_ids, rels, top_k)
        mrr.append(rr)
        ndcg.append(ndcg_v)
        recall.append(rec)
    return {
        "mrr@10": float(np.mean(mrr)) if mrr else 0.0,
        "ndcg@10": float(np.mean(ndcg)) if ndcg else 0.0,
        "recall@100": float(np.mean(recall)) if recall else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/benchmark_results.json")
    ap.add_argument("--out", default="results/experiment_k3.json")
    ap.add_argument("--datasets", nargs="+", default=None, help="默认取所有有 grid best 的数据集")
    ap.add_argument("--joint", nargs="+", default=[], help="对这些数据集跑联合 grid search（含 k3）")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        res = json.load(f)
    names = args.datasets or [n for n in res if "grid_best_params" in res[n]]
    out = {}

    # 模式一：k3 扫描（其他参数固定在 grid best）
    for name in names:
        r = res[name]
        docs, queries, qrels = load_dataset(r["path"])
        qs = queries[: r["grid_num_queries"]]
        engine = BM25Engine().build_index(docs)
        plans = [_QueryPlan(engine, q["query"]) for q in qs]
        rel = _relevant_map(qrels)
        base = dict(r["grid_best_params"])
        row = {"base_params": base, "num_queries": len(qs), "k3_sweep": {}}
        for k3 in K3_VALUES:
            m = evaluate_with_k3(plans, qs, rel, engine.doc_ids, base, k3)
            row["k3_sweep"][str(k3)] = {k: round(v, 5) for k, v in m.items()}
            print(f"[{name}] k3={k3}: ndcg={m['ndcg@10']:.5f} mrr={m['mrr@10']:.5f} "
                  f"recall={m['recall@100']:.5f}", flush=True)
        out[name] = row

    # 模式二：小数据集联合 grid（完整参数空间 × k3 ∈ {0, 1.2, 4}）
    config = load_config()
    for name in args.joint:
        docs, queries, qrels = load_dataset(os.path.join("dataset", name))
        qs = queries[:100]
        space = dict(config["grid_search"])
        space["k3"] = [0.0, 1.2, 4.0]
        gs = grid_search(docs, qs, qrels, space, metric="ndcg@10", top_k=100,
                         limit_queries=None, progress=False)
        out.setdefault(name, {})["joint_grid"] = {
            "num_queries": len(qs),
            "best_params": gs["best_params"],
            "best_metrics": gs["best_score"],
        }
        print(f"[{name}] 联合 grid best: {gs['best_params']} -> {gs['best_score']}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== k3 扫描汇总（ndcg@10，其他参数=grid best）===")
    print(f"{'dataset':20s} {'k3=0':>8s} {'best':>8s} {'best_k3':>8s} {'Δndcg':>8s}")
    for name in names:
        s = out[name]["k3_sweep"]
        k0 = s["0.0"]["ndcg@10"]
        best_k3 = max(K3_VALUES, key=lambda k: s[str(k)]["ndcg@10"])
        best = s[str(best_k3)]["ndcg@10"]
        print(f"{name:20s} {k0:8.4f} {best:8.4f} {best_k3:8.1f} {best - k0:+8.4f}")


if __name__ == "__main__":
    main()
