"""跨域混合语料构建：多次"拼接"已有语料，生成混合数据集。

目的：扩充参数词典在"混合/跨域"区域的覆盖。每个 mix 由 2–4 个已有语料的
子集拼接而成（doc_id 加前缀防冲突，分层采样保证相关文档保留），
保存为标准格式，之后可对其做 grid search 生成词典条目。

用法：python mix_corpora.py [--out-dir results/mixed_corpora] [--num-mixes 8]
"""

import argparse
import json
import os
import random

from data_loader import load_dataset_subsampled, save_dataset

DATASETS = [
    "arguana", "fiqa", "nfcorpus", "nq", "quora", "scidocs", "scifact",
    "trec-covid", "vihealthqa", "webis-touche2020",
    "dbpedia-entity", "climate-fever", "fever", "hotpotqa", "msmarco",
]

# 固定 20 个混合配方（组合不同领域/查询类型；quora 已按结论移除）
RECIPES = [
    ["fiqa", "nfcorpus", "scifact"],
    ["trec-covid", "scidocs", "scifact"],
    ["webis-touche2020", "arguana"],
    ["nq", "msmarco"],
    ["fever", "climate-fever"],
    ["dbpedia-entity", "hotpotqa"],
    ["fiqa", "webis-touche2020", "nq"],
    ["nfcorpus", "vihealthqa", "scifact"],
    # 扩充批次（mix08–mix19）：纳入超大规模语料
    ["msmarco", "nfcorpus"],
    ["msmarco", "fiqa"],
    ["fever", "scifact"],
    ["climate-fever", "scifact"],
    ["hotpotqa", "nq"],
    ["hotpotqa", "scidocs"],
    ["dbpedia-entity", "webis-touche2020"],
    ["trec-covid", "webis-touche2020"],
    ["fiqa", "vihealthqa", "scifact"],
    ["msmarco", "scifact"],
    ["fever", "hotpotqa", "climate-fever"],
    ["nfcorpus", "scidocs", "scifact", "trec-covid"],
]


def build_mix(components, out_dir, docs_per_comp=60000, queries_per_comp=60, seed=0):
    rng = random.Random(seed)
    docs, queries, qrels = [], [], []
    for name in components:
        d, q, r = load_dataset_subsampled(os.path.join("dataset", name), docs_per_comp)
        if len(d) > docs_per_comp:
            keep = sorted(rng.sample(range(len(d)), docs_per_comp))
            d = [d[i] for i in keep]
        valid_ids = {x["id"] for x in d}
        q = q[:queries_per_comp]
        valid_qids = {x["qid"] for x in q}
        for x in d:
            docs.append({"id": f"{name}::{x['id']}", "text": x["text"]})
        for x in q:
            queries.append({"qid": f"{name}::{x['qid']}", "query": x["query"]})
        for x in r:
            if x["doc_id"] in valid_ids and x["qid"] in valid_qids:
                qrels.append({"qid": f"{name}::{x['qid']}",
                              "doc_id": f"{name}::{x['doc_id']}",
                              "relevance": x["relevance"]})
    save_dataset(out_dir, docs, queries, qrels)
    return docs, queries, qrels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join("results", "mixed_corpora"))
    ap.add_argument("--recipes", nargs="+", default=None,
                    help="只构建指定编号的 mix，如 --recipes 0 1 2")
    ap.add_argument("--docs-per-comp", type=int, default=60000)
    ap.add_argument("--queries-per-comp", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    built = []
    for i, recipe in enumerate(RECIPES):
        if args.recipes and str(i) not in args.recipes:
            continue
        out = os.path.join(args.out_dir, f"mix{i:02d}")
        docs, queries, qrels = build_mix(
            recipe, out,
            docs_per_comp=args.docs_per_comp,
            queries_per_comp=args.queries_per_comp,
            seed=i,
        )
        meta = {"components": recipe, "num_docs": len(docs),
                "num_queries": len(queries), "num_qrels": len(qrels)}
        with open(os.path.join(out, "_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[mix] mix{i:02d} ({'+'.join(recipe)}): {len(docs)} docs, "
              f"{len(queries)} queries, {len(qrels)} qrels", flush=True)
        built.append(out)
    print(f"[mix] 完成 {len(built)} 个混合语料 -> {args.out_dir}")


if __name__ == "__main__":
    main()
