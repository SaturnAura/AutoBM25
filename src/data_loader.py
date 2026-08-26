"""数据集加载与保存。

兼容两种格式：
1. 项目标准格式：docs.jsonl / queries.jsonl / qrels.jsonl
2. BEIR 格式：corpus.jsonl（_id/title/text）+ queries.jsonl + qrels/*.tsv
"""

import json
import os
import random


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset(path):
    """返回 (docs, queries, qrels)。

    docs:    [{"id": str, "text": str}]
    queries: [{"qid": str, "query": str}]
    qrels:   [{"qid": str, "doc_id": str, "relevance": int}]
    """
    docs, queries, qrels = [], [], []

    docs_path = os.path.join(path, "docs.jsonl")
    if os.path.exists(docs_path):
        for row in load_jsonl(docs_path):
            docs.append({"id": str(row["id"]), "text": row.get("text", "")})
    else:
        corpus_path = os.path.join(path, "corpus.jsonl")
        for row in load_jsonl(corpus_path):
            text = row.get("text", "")
            if row.get("title"):
                text = row["title"] + " " + text
            docs.append({"id": str(row["_id"]), "text": text})

    queries_path = os.path.join(path, "queries.jsonl")
    if os.path.exists(queries_path):
        for row in load_jsonl(queries_path):
            qid = row.get("qid", row.get("_id"))
            queries.append(
                {"qid": str(qid), "query": row.get("query", row.get("text", ""))}
            )

    qrels_path = os.path.join(path, "qrels.jsonl")
    if os.path.exists(qrels_path):
        for row in load_jsonl(qrels_path):
            qrels.append(
                {
                    "qid": str(row["qid"]),
                    "doc_id": str(row["doc_id"]),
                    "relevance": int(row["relevance"]),
                }
            )
    else:
        # BEIR qrels/*.tsv，每行: qid \t corpus_id \t score
        qrels_dir = os.path.join(path, "qrels")
        if os.path.isdir(qrels_dir):
            for fn in sorted(os.listdir(qrels_dir)):
                if not fn.endswith(".tsv"):
                    continue
                with open(os.path.join(qrels_dir, fn), encoding="utf-8") as f:
                    header = f.readline().strip().lower().split("\t")
                    # 兼容 qid/corpus_id/score 与 query-id/corpus-id/score 两种表头
                    def col(name):
                        variants = {
                            "qid": ("qid", "query-id", "query_id", "queryid"),
                            "doc_id": ("corpus_id", "corpus-id", "doc_id", "doc-id", "docid"),
                            "score": ("score", "relevance", "rel"),
                        }
                        for i, h in enumerate(header):
                            if h in variants[name]:
                                return i
                        return None

                    i_q, i_d, i_s = col("qid"), col("doc_id"), col("score")
                    for line in f:
                        parts = line.strip().split("\t")
                        if i_q is None or i_d is None or i_s is None or len(parts) <= max(i_q, i_d, i_s):
                            continue
                        qid, doc_id, rel = parts[i_q], parts[i_d], int(parts[i_s])
                        if rel > 0:  # 只保留相关标注
                            qrels.append(
                                {"qid": qid, "doc_id": doc_id, "relevance": rel}
                            )
    return docs, queries, qrels


def save_dataset(path, docs, queries, qrels):
    """按项目标准格式保存（docs/queries/qrels 三个 jsonl）。"""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "docs.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(os.path.join(path, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(path, "qrels.jsonl"), "w", encoding="utf-8") as f:
        for r in qrels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_dataset_dirs(base="dataset", include_augmented=False):
    """扫描 base 下的数据集目录（含 docs.jsonl 或 corpus.jsonl 即视为数据集）。"""
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


def load_dataset_subsampled(path, max_docs, seed=42):
    """流式加载并分层子采样文档（相关性标注优先保留）。

    用于超大数据集（如 msmarco 880 万文档）：不把全部文档读入内存，
    先读 qrels 拿到相关文档集合，流式扫描语料时**保留全部相关文档**，
    其余文档用 reservoir sampling 补足到 max_docs 篇——避免随机采样
    把稀疏标注的相关文档全部丢弃（如 climate-fever）。
    """
    rng = random.Random(seed)

    # 1) 先读 qrels，收集相关文档 id 与全部标注
    qrels_all = []
    relevant_ids = set()
    qrels_path = os.path.join(path, "qrels.jsonl")
    if os.path.exists(qrels_path):
        with open(qrels_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rel = int(row["relevance"])
                qrels_all.append({"qid": str(row["qid"]), "doc_id": str(row["doc_id"]),
                                 "relevance": rel})
                if rel > 0:
                    relevant_ids.add(str(row["doc_id"]))
    else:
        qrels_dir = os.path.join(path, "qrels")
        if os.path.isdir(qrels_dir):
            for fn in sorted(os.listdir(qrels_dir)):
                if not fn.endswith(".tsv"):
                    continue
                with open(os.path.join(qrels_dir, fn), encoding="utf-8") as f:
                    header = f.readline().strip().lower().split("\t")

                    def col(name):
                        variants = {
                            "qid": ("qid", "query-id", "query_id", "queryid"),
                            "doc_id": ("corpus_id", "corpus-id", "doc_id", "doc-id", "docid"),
                            "score": ("score", "relevance", "rel"),
                        }
                        for i, h in enumerate(header):
                            if h in variants[name]:
                                return i
                        return None

                    i_q, i_d, i_s = col("qid"), col("doc_id"), col("score")
                    for line in f:
                        parts = line.strip().split("\t")
                        if i_q is None or i_d is None or i_s is None or \
                                len(parts) <= max(i_q, i_d, i_s):
                            continue
                        rel = int(parts[i_s])
                        qrels_all.append({"qid": parts[i_q], "doc_id": parts[i_d],
                                          "relevance": rel})
                        if rel > 0:
                            relevant_ids.add(parts[i_d])

    # 2) 流式扫描语料：相关文档全保留，其余 reservoir 采样补足
    docs = []
    nonrel = []
    corpus_path = os.path.join(path, "docs.jsonl")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(path, "corpus.jsonl")
    budget = max_docs - min(len(relevant_ids), max_docs)
    rel_count = 0
    nonrel_seen = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("id", row.get("_id")))
            text = row.get("text", "")
            if row.get("title"):
                text = row["title"] + " " + text
            if doc_id in relevant_ids:
                if rel_count < max_docs:
                    rel_count += 1
                    docs.append({"id": doc_id, "text": text})
            elif budget > 0:
                nonrel_seen += 1
                if len(nonrel) < budget:
                    nonrel.append({"id": doc_id, "text": text})
                else:
                    j = rng.randrange(nonrel_seen)
                    if j < budget:
                        nonrel[j] = {"id": doc_id, "text": text}
    docs.extend(nonrel)
    docs = docs[:max_docs]

    # 3) 查询全量加载
    queries = []
    queries_path = os.path.join(path, "queries.jsonl")
    if os.path.exists(queries_path):
        with open(queries_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                queries.append({
                    "qid": str(row.get("qid", row.get("_id"))),
                    "query": row.get("query", row.get("text", "")),
                })

    valid = {d["id"] for d in docs}
    qrels = [r for r in qrels_all if r["doc_id"] in valid and r["relevance"] > 0]
    return docs, queries, qrels
