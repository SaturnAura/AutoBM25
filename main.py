"""AutoBM25 命令行入口。

用法：
  python main.py --dataset dataset/XXX --predict           # 只预测参数
  python main.py --dataset dataset/XXX --eval              # 预测参数并评估（default vs predicted）
  python main.py --dataset dataset/XXX --eval --grid       # 额外做 grid search 画上界
  python main.py --augment dataset/XXX                     # 生成增强数据集
  python main.py --calibrate                               # 用全部数据集标定系数
"""

import argparse
import os

from data_augmentation import augment_dataset
from data_loader import load_dataset
from evaluator import calibrate, evaluate, grid_search
from feature_extractor import extract_features, save_json as save_features_json
from rule_predictor import load_config, predict, save_json as save_params_json


def find_dataset_dirs(base="dataset", include_augmented=False):
    dirs = []
    if not os.path.isdir(base):
        return dirs
    for entry in sorted(os.listdir(base)):
        p = os.path.join(base, entry)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "docs.jsonl")) or os.path.exists(
            os.path.join(p, "corpus.jsonl")
        ):
            dirs.append(p)
    if include_augmented:
        aug_root = os.path.join(base, "augmented")
        if os.path.isdir(aug_root):
            for entry in sorted(os.listdir(aug_root)):
                sub = os.path.join(aug_root, entry)
                if os.path.isdir(sub):
                    for e2 in sorted(os.listdir(sub)):
                        p = os.path.join(sub, e2)
                        if os.path.exists(os.path.join(p, "docs.jsonl")):
                            dirs.append(p)
    return dirs


def cmd_predict(args):
    docs, queries, qrels = load_dataset(args.dataset)
    name = os.path.basename(os.path.normpath(args.dataset))
    features = extract_features(docs, queries)
    params = predict(features)
    save_features_json(features, os.path.join(args.out_dir, f"{name}_features.json"))
    save_params_json(params, os.path.join(args.out_dir, f"{name}_predicted.json"))
    print(f"[predict] {name}: {params}")
    return features, params


def cmd_eval(args):
    docs, queries, qrels = load_dataset(args.dataset)
    name = os.path.basename(os.path.normpath(args.dataset))
    config = load_config()
    features, params = cmd_predict(args)
    default_params = dict(config["default_params"])
    default_metrics = evaluate(docs, queries, qrels, default_params)
    pred_metrics = evaluate(docs, queries, qrels, params)
    report = {
        "dataset": name,
        "features": features,
        "default_params": default_params,
        "default": default_metrics,
        "predicted_params": params,
        "predicted": pred_metrics,
    }
    for metric in ("mrr@10", "ndcg@10", "recall@100"):
        base, pred = default_metrics[metric], pred_metrics[metric]
        report[f"improve_vs_default_{metric}"] = (pred - base) / base * 100 if base > 0 else 0.0
    if args.grid:
        gs = grid_search(
            docs, queries, qrels, config["grid_search"],
            metric=config["grid_search"]["metric"],
            top_k=config["grid_search"]["top_k"],
            limit_queries=args.limit_queries,
            save_path=os.path.join(args.out_dir, f"{name}_grid.json"),
        )
        report["grid_search"] = gs["best_score"]
        for metric in ("mrr@10", "ndcg@10", "recall@100"):
            grid = gs["best_score"][metric]
            report[f"reach_grid_{metric}"] = (
                pred_metrics[metric] / grid * 100 if grid > 0 else 0.0
            )
    save_params_json(report, os.path.join(args.out_dir, f"{name}_eval.json"))
    print(f"[eval] {name}")
    print(f"  default   : {default_metrics}")
    print(f"  predicted : {pred_metrics}")
    if args.grid:
        print(f"  grid best : {gs['best_params']} -> {gs['best_score']}")
    return report


def cmd_augment(args):
    generated = augment_dataset(args.dataset)
    print(f"[augment] 生成了 {len(generated)} 个增强数据集:")
    for p in generated:
        print(f"  {p}")


def cmd_calibrate(args):
    if args.datasets:
        paths = args.datasets
    else:
        paths = find_dataset_dirs(args.base, include_augmented=args.include_augmented)
    print(f"[calibrate] 使用 {len(paths)} 个数据集: {[os.path.basename(p) for p in paths]}")
    calibrate(
        paths,
        out_dir=args.out_dir,
        test_frac=args.test_frac,
        limit_queries=args.limit_queries,
        progress=not args.no_progress,
    )


def main():
    parser = argparse.ArgumentParser(description="AutoBM25: 基于数据集统计特征的 BM25 自适应超参数选择")
    parser.add_argument("--dataset", type=str, help="数据集目录，如 dataset/fiqa")
    parser.add_argument("--predict", action="store_true", help="只预测参数，不评估")
    parser.add_argument("--eval", action="store_true", help="预测参数并评估（default vs predicted）")
    parser.add_argument("--grid", action="store_true", help="配合 --eval 额外运行 grid search（上界）")
    parser.add_argument("--augment", action="store_true", help="生成增强数据集")
    parser.add_argument("--calibrate", action="store_true", help="用全部数据集标定系数")
    parser.add_argument("--datasets", nargs="+", help="--calibrate 时显式指定数据集目录列表")
    parser.add_argument("--base", default="dataset", help="--calibrate 扫描的根目录，默认 dataset")
    parser.add_argument("--include-augmented", action="store_true",
                        help="--calibrate 时把 dataset/augmented 下的增强数据集也纳入")
    parser.add_argument("--test-frac", type=float, default=0.2, help="标定时留出的测试集比例")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="grid search 时最多使用的查询数（大数据集加速用）")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")
    parser.add_argument("--out-dir", default="results", help="结果 json 输出目录")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.calibrate:
        cmd_calibrate(args)
    elif args.augment:
        if not args.dataset:
            parser.error("--augment 需要 --dataset")
        cmd_augment(args)
    elif args.eval:
        if not args.dataset:
            parser.error("--eval 需要 --dataset")
        cmd_eval(args)
    elif args.predict:
        if not args.dataset:
            parser.error("--predict 需要 --dataset")
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
