"""批量实验脚本：对数据集做 特征提取 → 参数预测 → 评估 → 可选 grid search。

结果增量保存到 results/benchmark_results.json，README 中的结果表由此生成。

示例：
  # 全部数据集：特征 + 预测 + 全量评估
  python run_experiments.py --all --eval-queries -1
  # 对指定数据集额外跑 grid search（用前 N 条查询采样上界）
  python run_experiments.py --datasets nfcorpus scifact --grid-datasets nfcorpus scifact --grid-queries 200
"""

import argparse
import json
import os
import time

from data_loader import load_dataset
from evaluator import evaluate, grid_search
from feature_extractor import extract_features
from main import find_dataset_dirs
from rule_predictor import load_config, predict


def run_one(path, config, eval_queries=None, grid_queries=None):
    name = os.path.basename(os.path.normpath(path))
    row = {"name": name, "path": os.path.abspath(path)}

    t0 = time.time()
    docs, queries, qrels = load_dataset(path)
    row["load_seconds"] = round(time.time() - t0, 1)
    row["num_queries"] = len(queries)

    t0 = time.time()
    features = extract_features(docs, queries)
    row["features_seconds"] = round(time.time() - t0, 1)
    row["features"] = features
    row["predicted_params"] = predict(features, config)
    print(f"[{name}] 特征提取完成（{row['features_seconds']}s），"
          f"预测参数 {row['predicted_params']}", flush=True)

    if eval_queries is not None:
        qs = queries if eval_queries < 0 else queries[:eval_queries]
        row["eval_num_queries"] = len(qs)
        t0 = time.time()
        row["default_metrics"] = evaluate(docs, qs, qrels, config["default_params"])
        row["predicted_metrics"] = evaluate(docs, qs, qrels, row["predicted_params"])
        row["eval_seconds"] = round(time.time() - t0, 1)
        for m in ("mrr@10", "ndcg@10", "recall@100"):
            base = row["default_metrics"][m]
            pred = row["predicted_metrics"][m]
            row[f"improve_vs_default_{m}"] = (pred - base) / base * 100 if base > 0 else 0.0
        print(f"[{name}] 评估完成（{row['eval_seconds']}s）", flush=True)

    if grid_queries:
        t0 = time.time()
        gs = grid_search(
            docs, queries, qrels, config["grid_search"],
            metric=config["grid_search"]["metric"],
            top_k=config["grid_search"]["top_k"],
            limit_queries=grid_queries,
            progress=False,
        )
        row["grid_num_queries"] = len(queries) if grid_queries < 0 else min(grid_queries, len(queries))
        row["grid_best_params"] = gs["best_params"]
        row["grid_best_metrics"] = gs["best_score"]
        row["grid_seconds"] = round(time.time() - t0, 1)
        if eval_queries is not None:
            for m in ("mrr@10", "ndcg@10", "recall@100"):
                grid = gs["best_score"][m]
                pred = row["predicted_metrics"][m]
                row[f"reach_grid_{m}"] = pred / grid * 100 if grid > 0 else 0.0
        print(f"[{name}] grid search 完成（{row['grid_seconds']}s），"
              f"最优 {gs['best_params']} -> {gs['best_score']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", help="数据集名列表（dataset/ 下的目录名）")
    ap.add_argument("--all", action="store_true", help="扫描 dataset/ 下全部数据集")
    ap.add_argument("--eval-queries", type=int, default=None,
                    help="评估用查询数；-1 表示全部，缺省不评估")
    ap.add_argument("--grid-datasets", nargs="+", default=[], help="对这些数据集跑 grid search")
    ap.add_argument("--grid-queries", type=int, default=-1,
                    help="grid search 用查询数；-1 表示全部，缺省全部")
    ap.add_argument("--out", default="results/benchmark_results.json")
    args = ap.parse_args()

    if args.all:
        paths = find_dataset_dirs("dataset")
    elif args.datasets:
        paths = [os.path.join("dataset", d) for d in args.datasets]
    else:
        raise SystemExit("需要 --datasets 或 --all")
    if not paths:
        raise SystemExit("没有找到数据集")

    config = load_config()
    results = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for path in paths:
        name = os.path.basename(os.path.normpath(path))
        row = run_one(
            path,
            config,
            eval_queries=args.eval_queries,
            grid_queries=args.grid_queries if name in args.grid_datasets else None,
        )
        results[name] = row
        # 增量保存：中途失败也不丢已跑完的数据集
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[done] {name} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
