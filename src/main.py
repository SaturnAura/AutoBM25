"""AutoBM25 命令行入口。

用法：
  python main.py --dataset dataset/XXX --search "查询词"      # 直接检索，返回 top-k
  python main.py --dataset dataset/XXX --interactive           # 交互式检索（输入查询回车出结果）
  python main.py --dataset dataset/XXX --predict           # 只预测参数
  python main.py --dataset dataset/XXX --eval              # 预测参数并评估（default vs predicted）
  python main.py --dataset dataset/XXX --eval --grid       # 额外做 grid search 画上界
  python main.py --augment dataset/XXX                     # 生成增强数据集
  python main.py --calibrate                               # 用全部数据集标定系数
"""

import argparse
import os
import sys

from bm25_engine import BM25Engine
from data_loader import load_dataset, load_dataset_subsampled
from evaluator import calibrate, evaluate, grid_search
from feature_extractor import extract_features, save_json as save_features_json
from rule_predictor import load_config, predict, predict_with_dictionary, save_json as save_params_json


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
    params = predict(features) if args.no_dict else predict_with_dictionary(features)
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
        space = dict(config["grid_search"])
        if args.grid_k3:
            space["k3"] = [float(v) for v in args.grid_k3]
        gs = grid_search(
            docs, queries, qrels, space,
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
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _research = os.path.join(_root, "research")
    if _research not in sys.path:
        sys.path.insert(0, _research)
    from data_augmentation import augment_dataset
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


def cmd_build_dict(args):
    from param_dictionary import DEFAULT_DICT_PATH, build_dictionary
    build_dictionary(
        benchmark_path=os.path.join(args.out_dir, "benchmark_results.json"),
        out_path=DEFAULT_DICT_PATH,
        k_neighbors=args.dict_k,
        match_threshold=args.dict_threshold,
    )


def _print_search_results(engine, query, top_k):
    results = engine.search(query, top_k=top_k)
    if not results:
        print("  （无匹配结果）")
        return
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i:3d}. [{score:.4f}] {doc_id}")


def cmd_search(args):
    """输入数据集 → 自适应参数 → 建索引 → 直接检索（单条查询或交互式）。"""
    name = os.path.basename(os.path.normpath(args.dataset))
    if args.max_docs and os.path.exists(os.path.join(args.dataset, "corpus.jsonl")):
        docs, queries, qrels = load_dataset_subsampled(args.dataset, args.max_docs)
    else:
        docs, queries, qrels = load_dataset(args.dataset)
    if not docs:
        raise SystemExit(f"[search] {args.dataset} 中没有文档（需要 docs.jsonl 或 corpus.jsonl）")
    features = extract_features(docs, queries)
    params = predict(features) if args.no_dict else predict_with_dictionary(features)
    engine = BM25Engine().build_index(docs)
    engine.set_params(
        k1=params.get("k1"),
        b=params.get("b"),
        k3=params.get("k3"),
        delta=params.get("delta"),
        idf_type=params.get("idf_type"),
    )
    print(f"[search] 数据集 {name}: {len(docs)} 篇文档，自适应参数 {params}")
    if args.search and args.search != "__interactive__":
        _print_search_results(engine, args.search, args.topk)
        return
    print("[search] 交互模式：输入查询后回车检索，输入 exit / quit 退出")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            break
        _print_search_results(engine, q, args.topk)


def main():
    parser = argparse.ArgumentParser(description="AutoBM25: 基于数据集统计特征的 BM25 自适应超参数选择")
    parser.add_argument("--dataset", type=str, help="数据集目录，如 dataset/fiqa")
    parser.add_argument("--search", nargs="?", const="__interactive__", default=None,
                        metavar="QUERY", help="检索模式：直接对查询词返回 top-k；不带查询词则进入交互式检索")
    parser.add_argument("--interactive", action="store_true", help="交互式检索（等价于 --search 不带查询词）")
    parser.add_argument("--topk", type=int, default=10, help="检索返回条数（默认 10）")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="超大数据集子采样（与实验口径一致，相关标注全保留）")
    parser.add_argument("--predict", action="store_true", help="只预测参数，不评估")
    parser.add_argument("--eval", action="store_true", help="预测参数并评估（default vs predicted）")
    parser.add_argument("--grid", action="store_true", help="配合 --eval 额外运行 grid search（上界）")
    parser.add_argument("--augment", action="store_true", help="生成增强数据集")
    parser.add_argument("--calibrate", action="store_true", help="用全部数据集标定系数")
    parser.add_argument("--datasets", nargs="+", help="--calibrate 时显式指定数据集目录列表")
    parser.add_argument("--base", default="dataset", help="--calibrate 扫描的根目录，默认 dataset")
    parser.add_argument("--include-augmented", action="store_true",
                        help="--calibrate 时把 dataset/augmented 下的增强数据集也纳入")
    parser.add_argument("--build-dict", action="store_true",
                        help="从 results/benchmark_results.json 构建参数词典")
    parser.add_argument("--dict-k", type=int, default=1, help="词典最近邻数量")
    parser.add_argument("--dict-threshold", type=float, default=2.0,
                        help="词典命中距离阈值（归一化特征空间）")
    parser.add_argument("--no-dict", action="store_true", help="预测时跳过词典，只用启发式")
    parser.add_argument("--test-frac", type=float, default=0.2, help="标定时留出的测试集比例")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="grid search 时最多使用的查询数（大数据集加速用）")
    parser.add_argument("--grid-k3", nargs="+", default=None,
                        help="覆盖 grid 的 k3 搜索轴，如 --grid-k3 0 1.2 4")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")
    parser.add_argument("--out-dir", default="results", help="结果 json 输出目录")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.search or args.interactive:
        if not args.dataset:
            parser.error("--search / --interactive 需要 --dataset")
        cmd_search(args)
    elif args.build_dict:
        cmd_build_dict(args)
    elif args.calibrate:
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
