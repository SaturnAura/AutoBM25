"""混合语料（Mega Corpus）泛化测试。

把多个已有数据集的文档拼接成一个大的混合文档集（doc_id 加数据集前缀防止冲突），
模拟"任意新数据集"：提取混合语料的统计特征 → 参数词典查表（O(1)）→
未命中则回退启发式规则 → 在查询子集上评估。

用法：
  python mega_corpus.py [--datasets a b c] [--eval-queries 500]
"""

import argparse
import json
import os

from data_loader import load_dataset, save_dataset
from bm25_engine import BM25Engine
from evaluator import evaluate_engine
from feature_extractor import extract_features, save_json as save_features_json
from param_dictionary import DEFAULT_DICT_PATH, ParamDictionary
from rule_predictor import load_config, predict

DEFAULT_DATASETS = [
    "arguana", "fiqa", "nfcorpus", "quora", "scidocs", "scifact",
    "trec-covid", "trec-covid-v2", "vihealthqa", "webis-touche2020",
]


def build_mega_corpus(datasets, out_dir, id_sep="::"):
    docs, queries, qrels = [], [], []
    doc_id_map = {}
    for ds in datasets:
        d, q, r = load_dataset(os.path.join("dataset", ds))
        for x in d:
            new_id = f"{ds}{id_sep}{x['id']}"
            doc_id_map[(ds, x["id"])] = new_id
            docs.append({"id": new_id, "text": x["text"]})
        for x in q:
            queries.append({"qid": f"{ds}{id_sep}{x['qid']}", "query": x["query"]})
        for x in r:
            if (ds, x["doc_id"]) in doc_id_map:
                qrels.append({
                    "qid": f"{ds}{id_sep}{x['qid']}",
                    "doc_id": doc_id_map[(ds, x["doc_id"])],
                    "relevance": x["relevance"],
                })
    save_dataset(out_dir, docs, queries, qrels)
    print(f"[mega] {len(docs)} docs / {len(queries)} queries / {len(qrels)} qrels -> {out_dir}")
    return docs, queries, qrels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--eval-queries", type=int, default=500,
                    help="评估用查询数（混合语料的全部查询很大，默认采样 500）")
    ap.add_argument("--out-dir", default=os.path.join("results", "mega_corpus"))
    ap.add_argument("--dict-path", default=DEFAULT_DICT_PATH)
    ap.add_argument("--skip-build", action="store_true",
                    help="复用已生成的混合语料（跳过重新拼接）")
    args = ap.parse_args()

    if args.skip_build and os.path.exists(os.path.join(args.out_dir, "docs.jsonl")):
        docs, queries, qrels = load_dataset(args.out_dir)
        print(f"[mega] 复用已生成的混合语料: {len(docs)} docs / {len(queries)} queries")
    else:
        docs, queries, qrels = build_mega_corpus(args.datasets, args.out_dir)
    features = extract_features(docs, queries)
    save_features_json(features, os.path.join(args.out_dir, "features.json"))
    print(f"[mega] 混合语料特征: doc_count={features['doc_count']} "
          f"avgdl={features['avgdl']:.1f} cv_len={features['cv_len']:.3f} "
          f"avg_ttr={features['avg_ttr']:.3f} avg_query_len={features['avg_query_len']:.1f}")

    config = load_config()
    heuristic = predict(features, config)

    # 参数词典查表（O(1) 最近邻）；未命中回退启发式
    dictionary = None
    if os.path.exists(args.dict_path):
        dictionary = ParamDictionary.load(args.dict_path)
    probe = dictionary.probe(features) if dictionary else None
    hit = dictionary.lookup(features) if dictionary else None
    chosen = hit if hit is not None else heuristic

    qs = queries[: args.eval_queries]
    engine = BM25Engine().build_index(docs)

    def eval_params(params):
        engine.set_params(
            k1=params.get("k1"),
            b=params.get("b"),
            k3=params.get("k3"),
            delta=params.get("delta"),
            idf_type=params.get("idf_type"),
        )
        return evaluate_engine(engine, qs, qrels, top_k=100)

    default_m = eval_params(config["default_params"])
    heur_m = eval_params(heuristic)
    chosen_m = eval_params(chosen)
    report = {
        "num_docs": len(docs),
        "num_queries_total": len(queries),
        "eval_num_queries": len(qs),
        "mode": "dictionary" if hit is not None else "heuristic-fallback",
        "dict_probe": probe,
        "features": features,
        "heuristic_params": heuristic,
        "heuristic_metrics": heur_m,
        "chosen_params": chosen,
        "default_metrics": default_m,
        "chosen_metrics": chosen_m,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[mega] 预测模式: {report['mode']} | 词典诊断: {probe}")
    print(f"[mega] heuristic 参数: {heuristic} -> {heur_m}")
    print(f"[mega] chosen 参数: {chosen} -> {chosen_m}")
    print(f"[mega] default : {default_m}")
    print(f"[mega] 报告已保存 -> {os.path.join(args.out_dir, 'report.json')}")


if __name__ == "__main__":
    main()
