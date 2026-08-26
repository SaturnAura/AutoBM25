"""AutoBM25 命令行入口（两种模式）。

用法：
  python main.py --dataset dataset/XXX                # 批量检索：用 queries.jsonl 检索，
                                                      #   结果写回 dataset/XXX/results.jsonl，
                                                      #   有 qrels 时附带评测 evaluation.json
  python main.py --dataset dataset/XXX --interactive  # 交互式检索：输入查询 → 回车出结果

启动时会输出一条日志，显示本数据集自适应选择的 BM25 超参数。
"""

import argparse
import json
import os

from bm25_engine import BM25Engine
from data_loader import load_dataset, load_dataset_subsampled
from evaluator import evaluate_engine
from feature_extractor import extract_features
from rule_predictor import load_config, predict, predict_with_dictionary


def _load(args):
    if args.max_docs and os.path.exists(os.path.join(args.dataset, "corpus.jsonl")):
        return load_dataset_subsampled(args.dataset, args.max_docs)
    return load_dataset(args.dataset)


def _choose_params(features, args):
    return predict(features) if args.no_dict else predict_with_dictionary(features)


def _log_params(name, docs, params):
    print(
        f"[AutoBM25] dataset={name} docs={len(docs)} | 自适应超参: "
        f"k1={params['k1']} b={params['b']} k3={params['k3']} "
        f"δ={params['delta']} idf={params['idf_type']} ({params['model_variant']})"
    )


def _build_engine(docs, params):
    engine = BM25Engine().build_index(docs)
    engine.set_params(
        k1=params.get("k1"),
        b=params.get("b"),
        k3=params.get("k3"),
        delta=params.get("delta"),
        idf_type=params.get("idf_type"),
    )
    return engine


def _set_engine_params(engine, params):
    engine.set_params(
        k1=params.get("k1"),
        b=params.get("b"),
        k3=params.get("k3"),
        delta=params.get("delta"),
        idf_type=params.get("idf_type"),
    )


def cmd_batch(args):
    """批量检索：跑完 dataset/XXX/queries.jsonl 的全部查询，
    结果写回同目录 results.jsonl；有 qrels 时输出 evaluation.json。"""
    name = os.path.basename(os.path.normpath(args.dataset))
    docs, queries, qrels = _load(args)
    if not docs:
        raise SystemExit(f"[AutoBM25] {args.dataset} 中没有文档（需要 docs.jsonl 或 corpus.jsonl）")
    if not queries:
        raise SystemExit(
            f"[AutoBM25] {args.dataset} 没有 queries.jsonl，无法批量检索；"
            "请用 --interactive 交互式输入查询。"
        )

    features = extract_features(docs, queries)
    params = _choose_params(features, args)
    _log_params(name, docs, params)
    engine = _build_engine(docs, params)
    top_k = args.topk or 100

    results_path = os.path.join(args.dataset, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for q in queries:
            hits = engine.search(q["query"], top_k=top_k)
            f.write(
                json.dumps(
                    {
                        "qid": q["qid"],
                        "results": [
                            {"rank": i, "doc_id": d, "score": s}
                            for i, (d, s) in enumerate(hits, 1)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[AutoBM25] 检索完成：{len(queries)} 条查询 -> {results_path}")

    if qrels:
        default_params = load_config()["default_params"]
        _set_engine_params(engine, default_params)
        default_metrics = evaluate_engine(engine, queries, qrels, top_k=100)
        _set_engine_params(engine, params)
        adaptive_metrics = evaluate_engine(engine, queries, qrels, top_k=100)
        eval_path = os.path.join(args.dataset, "evaluation.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": name,
                    "adaptive_params": params,
                    "default_params": default_params,
                    "default": default_metrics,
                    "adaptive": adaptive_metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[AutoBM25] 评测完成（{len(queries)} 条查询）-> {eval_path}")
        print(f"[AutoBM25]   default  : {default_metrics}")
        print(f"[AutoBM25]   adaptive : {adaptive_metrics}")
    else:
        print(
            "[AutoBM25] 未找到 qrels（qrels.jsonl 或 qrels/*.tsv），跳过评测；"
            "提供标注后可自动输出评测结果。"
        )


def cmd_interactive(args):
    """交互式检索：输入查询 → 回车出结果，exit 退出。"""
    name = os.path.basename(os.path.normpath(args.dataset))
    docs, queries, qrels = _load(args)
    if not docs:
        raise SystemExit(f"[AutoBM25] {args.dataset} 中没有文档（需要 docs.jsonl 或 corpus.jsonl）")

    features = extract_features(docs, queries)
    params = _choose_params(features, args)
    _log_params(name, docs, params)
    engine = _build_engine(docs, params)
    top_k = args.topk or 10

    print("[AutoBM25] 交互模式：输入查询后回车检索，输入 exit / quit 退出")
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
            print("  （无匹配结果）")
            continue
        print(f"[AutoBM25] 查询「{q}」→ Top {len(hits)} 结果（分数越高越相关）：")
        print(f"  {'排名':>4}  {'得分':>12}  文档ID")
        for i, (doc_id, score) in enumerate(hits, 1):
            print(f"  {i:>4}  {score:>12.4f}  {doc_id}")


def main():
    parser = argparse.ArgumentParser(
        description="AutoBM25: 零标注 BM25 超参数自适应检索（批量 / 交互式两种模式）"
    )
    parser.add_argument("--dataset", required=True, help="数据集目录，如 dataset/fiqa")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式检索；缺省为批量检索（用 queries.jsonl 跑完并写回结果）")
    parser.add_argument("--topk", type=int, default=None,
                        help="返回条数（批量默认 100，交互默认 10）")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="超大数据集子采样（与实验口径一致，相关标注全保留）")
    parser.add_argument("--no-dict", action="store_true",
                        help="跳过参数词典，只用启发式规则")
    args = parser.parse_args()

    if args.interactive:
        cmd_interactive(args)
    else:
        cmd_batch(args)


if __name__ == "__main__":
    main()
