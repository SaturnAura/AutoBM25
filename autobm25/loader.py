"""Dataset loading (retrieval side): standard format + BEIR format +
streaming subsampling for huge corpora."""

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
    """Return (docs, queries, qrels).

    docs:    [{"id", "text"}]; queries: [{"qid", "query"}];
    qrels:   [{"qid", "doc_id", "relevance"}]
    Supports the standard format (docs/queries/qrels.jsonl) and BEIR format
    (corpus.jsonl + qrels/*.tsv); queries and qrels are optional for pure retrieval.
    """
    docs, queries, qrels = [], [], []
    docs_path = os.path.join(path, "docs.jsonl")
    if os.path.exists(docs_path):
        for row in load_jsonl(docs_path):
            docs.append({"id": str(row["id"]), "text": row.get("text", "")})
    else:
        corpus_path = os.path.join(path, "corpus.jsonl")
        if os.path.exists(corpus_path):
            for row in load_jsonl(corpus_path):
                text = row.get("text", "")
                if row.get("title"):
                    text = row["title"] + " " + text
                docs.append({"id": str(row["_id"]), "text": text})

    queries_path = os.path.join(path, "queries.jsonl")
    if os.path.exists(queries_path):
        for row in load_jsonl(queries_path):
            qid = row.get("qid", row.get("_id"))
            queries.append({"qid": str(qid), "query": row.get("query", row.get("text", ""))})

    qrels_path = os.path.join(path, "qrels.jsonl")
    if os.path.exists(qrels_path):
        for row in load_jsonl(qrels_path):
            qrels.append({"qid": str(row["qid"]), "doc_id": str(row["doc_id"]),
                          "relevance": int(row["relevance"])})
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
                        if int(parts[i_s]) > 0:
                            qrels.append({"qid": parts[i_q], "doc_id": parts[i_d],
                                          "relevance": int(parts[i_s])})
    return docs, queries, qrels


def load_dataset_subsampled(path, max_docs, seed=42):
    """Streaming load + stratified subsampling for huge corpora
    (all relevant annotations are kept)."""
    rng = random.Random(seed)
    qrels_all = []
    relevant_ids = set()
    qrels_path = os.path.join(path, "qrels.jsonl")
    if os.path.exists(qrels_path):
        for row in load_jsonl(qrels_path):
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

    docs, nonrel = [], []
    corpus_path = os.path.join(path, "docs.jsonl")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(path, "corpus.jsonl")
    budget = max_docs - min(len(relevant_ids), max_docs)
    rel_count, nonrel_seen = 0, 0
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

    queries = []
    queries_path = os.path.join(path, "queries.jsonl")
    if os.path.exists(queries_path):
        for row in load_jsonl(queries_path):
            queries.append({"qid": str(row.get("qid", row.get("_id"))),
                            "query": row.get("query", row.get("text", ""))})
    valid = {d["id"] for d in docs}
    qrels = [r for r in qrels_all if r["doc_id"] in valid and r["relevance"] > 0]
    return docs, queries, qrels
