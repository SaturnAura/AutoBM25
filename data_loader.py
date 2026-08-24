"""数据集加载与保存。

兼容两种格式：
1. 项目标准格式：docs.jsonl / queries.jsonl / qrels.jsonl
2. BEIR 格式：corpus.jsonl（_id/title/text）+ queries.jsonl + qrels/*.tsv
"""

import json
import os


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
