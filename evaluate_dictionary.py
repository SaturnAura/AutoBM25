"""参数词典留一法（LOO）评估。

对每个有 grid 结果的数据集：用其余数据集构建词典，预测该数据集的参数
（查表命中则用词典参数，未命中回退启发式），并在与 grid 相同的查询子集上评估，
与 default / heuristic / grid 上界对比——检验"大字典 + 回退"的泛化能力。

用法：python evaluate_dictionary.py [--thresholds 1.5 2.0 2.5 3.0]
"""

import argparse
import json
import os

from data_loader import load_dataset
from evaluator import evaluate
from param_dictionary import DEFAULT_DICT_PATH, ParamDictionary
from rule_predictor import load_config, predict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dict", default=DEFAULT_DICT_PATH)
    ap.add_argument("--benchmark", default=os.path.join("results", "benchmark_results.json"))
    ap.add_argument("--mixes-benchmark",
                    default=os.path.join("results", "benchmark_mixes.json"))
    ap.add_argument("--out", default=os.path.join("results", "experiment_dictionary_loo.json"))
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    args = ap.parse_args()

    with open(args.dict, encoding="utf-8") as f:
        dict_data = json.load(f)
    entries = dict_data["entries"]
    with open(args.benchmark, encoding="utf-8") as f:
        bench = json.load(f)
    if os.path.exists(args.mixes_benchmark):
        with open(args.mixes_benchmark, encoding="utf-8") as f:
            mixes = json.load(f)
        for k, v in mixes.items():
            bench[f"mix::{k}"] = v
    sample_path = os.path.join("results", "benchmark_grid_sample_eval.json")
    sample = {}
    if os.path.exists(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            sample = json.load(f)
    config = load_config()

    # 预加载各数据集的评估材料（查询子集与 grid 上界）
    materials = {}
    for e in entries:
        name = e["name"]
        r = bench.get(name)
        if r is None:
            print(f"[loo] 跳过 {name}（无 benchmark 条目）", flush=True)
            continue
        docs, queries, qrels = load_dataset(r["path"])
        materials[name] = {
            "docs": docs,
            "queries": queries[: r["grid_num_queries"]],
            "qrels": qrels,
            "grid_metrics": r["grid_best_metrics"],
            "num_queries": r["grid_num_queries"],
        }
        print(f"[loo] {name}: 已加载（{materials[name]['num_queries']} 条查询）", flush=True)

    for th in args.thresholds:
        results = {}
        for e in entries:
            name = e["name"]
            others = [x for x in entries if x["name"] != name]
            dictionary = ParamDictionary(others, k_neighbors=1, match_threshold=th)
            hit = dictionary.lookup(e["features"])
            if hit is not None:
                params = {k: hit[k] for k in ("k1", "b", "k3", "delta", "idf_type", "model_variant")}
                mode = "dictionary"
            else:
                params = predict(e["features"], config)
                mode = "heuristic"
            m = materials[name]
            metrics = evaluate(m["docs"], m["queries"], m["qrels"], params)
            if name in sample and "predicted_metrics" in sample[name]:
                heur = sample[name]["predicted_metrics"]
            else:
                heur = evaluate(m["docs"], m["queries"], m["qrels"],
                                predict(e["features"], config))
            results[name] = {
                "mode": mode,
                "dict_distance": hit.get("_dict_distance") if hit else None,
                "dict_match": hit.get("_dict_match") if hit else None,
                "params": params,
                "metrics": metrics,
                "heuristic_metrics": heur,
                "grid_metrics": m["grid_metrics"],
            }
            print(f"[loo] th={th} {name}: mode={mode} ndcg={metrics['ndcg@10']:.4f} "
                  f"(heur={heur['ndcg@10']:.4f}, grid={m['grid_metrics']['ndcg@10']:.4f})", flush=True)

        # 汇总
        hits = sum(1 for r in results.values() if r["mode"] == "dictionary")
        avg_ndcg = sum(r["metrics"]["ndcg@10"] for r in results.values()) / len(results)
        avg_heur = sum(r["heuristic_metrics"]["ndcg@10"] for r in results.values()) / len(results)
        avg_grid = sum(r["grid_metrics"]["ndcg@10"] for r in results.values()) / len(results)
        print(f"[loo] th={th}: 命中 {hits}/{len(results)}，平均 NDCG@10 "
              f"dict+fallback={avg_ndcg:.4f} heuristic={avg_heur:.4f} grid={avg_grid:.4f}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"threshold": th, "results": results,
                       "summary": {"hits": hits, "total": len(results),
                                   "avg_ndcg": avg_ndcg, "avg_heur": avg_heur,
                                   "avg_grid": avg_grid}},
                      f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
