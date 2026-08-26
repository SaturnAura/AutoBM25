"""参数词典命中率分析。

回答两个问题：
1. 大字典需不需要改进：对所有有 grid 结果的数据集做留一法（LOO），统计
   命中/回退分布与命中质量（对比启发式与 grid 上界）；
2. 命中率是否够高、拼接是否有效：把混合语料（mix）纳入词典后，重新探测
   未见数据集与未见 mix，观察命中率是否提升。

用法：
  python analyze_dictionary.py \
      --benchmark results/benchmark_results.json \
      --mixes-benchmark results/benchmark_mixes.json
"""

import argparse
import json
import os

from param_dictionary import DEFAULT_DICT_PATH, ParamDictionary
from rule_predictor import load_config, predict


def load_entries(benchmark_path, mixes_benchmark_path=None):
    """从 benchmark 与（可选）混合语料结果收集词典条目。"""
    with open(benchmark_path, encoding="utf-8") as f:
        bench = json.load(f)
    entries = []
    for name, r in bench.items():
        if "grid_best_params" not in r:
            continue
        entries.append({
            "name": name,
            "features": r["features"],
            "params": {
                **{k: r["grid_best_params"].get(k, 0.0)
                   for k in ("k1", "b", "k3", "delta")},
                "idf_type": r["grid_best_params"]["idf_type"],
            },
            "grid_ndcg": r["grid_best_metrics"]["ndcg@10"],
        })
    if mixes_benchmark_path and os.path.exists(mixes_benchmark_path):
        with open(mixes_benchmark_path, encoding="utf-8") as f:
            mixes = json.load(f)
        for name, r in mixes.items():
            if "grid_best_params" not in r:
                continue
            entries.append({
                "name": f"mix::{name}",
                "features": r["features"],
                "params": {
                    **{k: r["grid_best_params"].get(k, 0.0)
                       for k in ("k1", "b", "k3", "delta")},
                    "idf_type": r["grid_best_params"]["idf_type"],
                },
                "grid_ndcg": r["grid_best_metrics"]["ndcg@10"],
            })
    return entries


def probe_targets(benchmark_path, mixes_benchmark_path=None):
    """收集探测目标：所有数据集（含无 grid 的）+ 所有混合语料。"""
    with open(benchmark_path, encoding="utf-8") as f:
        bench = json.load(f)
    targets = []
    for name, r in bench.items():
        targets.append({"name": name, "features": r["features"],
                        "has_grid": "grid_best_params" in r})
    if mixes_benchmark_path and os.path.exists(mixes_benchmark_path):
        with open(mixes_benchmark_path, encoding="utf-8") as f:
            mixes = json.load(f)
        for name, r in mixes.items():
            targets.append({"name": f"mix::{name}",
                            "features": r["features"],
                            "has_grid": "grid_best_params" in r})
    return targets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default=os.path.join("results", "benchmark_results.json"))
    ap.add_argument("--dict", default=DEFAULT_DICT_PATH)
    ap.add_argument("--mixes-benchmark",
                    default=os.path.join("results", "benchmark_mixes.json"))
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join("results", "dictionary_hitrate.json"))
    args = ap.parse_args()
    config = load_config()

    entries = load_entries(args.benchmark, args.mixes_benchmark)
    targets = probe_targets(args.benchmark, args.mixes_benchmark)
    print(f"[hitrate] 词典条目（仅 benchmark grid）: {[e['name'] for e in entries]}")
    print(f"[hitrate] 探测目标: {[t['name'] for t in targets]}")

    rows = []
    for t in targets:
        others = [e for e in entries if e["name"] != t["name"]]
        dictionary = ParamDictionary(others, k_neighbors=1,
                                     match_threshold=args.threshold)
        probe = dictionary.probe(t["features"])
        hit = dictionary.lookup(t["features"])
        if hit is not None:
            params, mode = hit, "dictionary"
        else:
            params, mode = predict(t["features"], config), "heuristic"
        rows.append({
            "name": t["name"],
            "has_grid": t["has_grid"],
            "mode": mode,
            "oov": probe["oov"],
            "distance": probe["distance"],
            "match": probe["match"],
            "params": params,
        })
        print(f"  {t['name']:18s} mode={mode:10s} oov={probe['oov']} "
              f"dist={probe['distance']} match={probe['match']}")

    # 命中率汇总（按类别）
    def rate(rows_subset):
        return sum(1 for r in rows_subset if r["mode"] == "dictionary") / len(rows_subset)

    grid_targets = [r for r in rows if r["has_grid"]]
    nongrid = [r for r in rows if not r["has_grid"]]
    mixes = [r for r in rows if r["name"].startswith("mix::")]
    summary = {
        "threshold": args.threshold,
        "total_targets": len(rows),
        "hit_rate_all": rate(rows),
        "hit_rate_grid_datasets": rate(grid_targets) if grid_targets else 0.0,
        "hit_rate_new_datasets": rate(nongrid) if nongrid else 0.0,
        "hit_rate_mixes": rate(mixes) if mixes else 0.0,
        "details": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[hitrate] 总命中率 {summary['hit_rate_all']:.1%} "
          f"(grid 数据集 {summary['hit_rate_grid_datasets']:.1%}, "
          f"新数据集 {summary['hit_rate_new_datasets']:.1%}, "
          f"mix {summary['hit_rate_mixes']:.1%})")
    print(f"[hitrate] 结果 -> {args.out}")


if __name__ == "__main__":
    main()
