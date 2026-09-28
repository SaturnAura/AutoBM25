from autobm25 import AutoBM25

corpus = [
    {"id": "doc1", "text": "This is the first document."},
    {"id": "doc2", "text": "This document is the second document."},
    {"id": "doc3", "text": "And this is the third one."},
    {"id": "doc4", "text": "Is this the first document?"},
]
QUERIES = [
    "first document",
    "second document",
    "third one",
]
retriever = AutoBM25()
retriever.index(corpus, queries=QUERIES)   # corpus 可为字符串列表或 [{"id","text"}]
print(retriever.params)                    # 该语料自动选出的超参数
retriever.search("your query", top_k=3)    # -> [(doc_id, score), ...]
retriever.search_many(QUERIES, top_k=1)    # 批量，一条 query 一个排序列表