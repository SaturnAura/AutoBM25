"""自实现 BM25 / BM25+ 检索引擎（倒排索引）。

打分公式：
  BM25:  score += idf(q) * [f(q,D)*(k1+1) / (f(q,D) + k1*(1-b+b*|D|/avgdl))]
                 * [f(q,Q)*(k3+1) / (f(q,Q) + k3)]           （查询词频饱和项，k3=0 时退化为 1）
  BM25+: score += idf(q) * [上面那项 + delta]

两种 IDF：
  rsj:      ln((N - df + 0.5) / (df + 0.5))
  smoothed: ln((N + 1) / (df + 1))
"""

import math

import numpy as np


def tokenize(text):
    """空格分词 + 小写化（按项目要求，不去停用词）。"""
    return text.lower().split()


class BM25Engine:
    def __init__(self, k1=1.2, b=0.75, k3=0.0, delta=0.0, idf_type="rsj"):
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self.delta = delta
        self.idf_type = idf_type
        self.N = 0
        self.avgdl = 0.0
        self.doc_ids = []     # doc_idx -> doc_id
        self.doc_len = None   # np.ndarray[int32]，doc_idx -> 词数
        self.postings = {}    # term -> (doc_idx 数组, tf 数组)
        self.doc_freq = {}    # term -> df
        self.idf_cache = {}

    def build_index(self, docs):
        """docs: [{"id": str, "text": str}]"""
        self.doc_ids = [d["id"] for d in docs]
        self.N = len(docs)
        self.doc_len = np.zeros(self.N, dtype=np.int32)
        total_len = 0
        tmp_postings = {}  # term -> [(doc_idx, tf), ...]
        for i, d in enumerate(docs):
            toks = tokenize(d["text"])
            self.doc_len[i] = len(toks)
            total_len += self.doc_len[i]
            seen = {}
            for t in toks:
                seen[t] = seen.get(t, 0) + 1
            for t, tf in seen.items():
                tmp_postings.setdefault(t, []).append((i, tf))
        self.avgdl = total_len / self.N if self.N else 0.0
        self._dl_float = self.doc_len.astype(np.float64)
        for t, pairs in tmp_postings.items():
            arr = np.asarray(pairs, dtype=np.int64)
            self.postings[t] = (arr[:, 0], arr[:, 1])
        self.doc_freq = {t: len(docs_arr) for t, (docs_arr, _) in self.postings.items()}
        self.idf_cache = {}
        return self

    def set_params(self, k1=None, b=None, k3=None, delta=None, idf_type=None):
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b
        if k3 is not None:
            self.k3 = k3
        if delta is not None:
            self.delta = delta
        if idf_type is not None and idf_type != self.idf_type:
            self.idf_type = idf_type
            self.idf_cache = {}

    def idf(self, term):
        if term not in self.idf_cache:
            df = self.doc_freq.get(term, 0)
            N = self.N
            if self.idf_type == "rsj":
                val = math.log((N - df + 0.5) / (df + 0.5))
            else:  # smoothed
                val = math.log((N + 1) / (df + 1))
            self.idf_cache[term] = val
        return self.idf_cache[term]

    def score_doc(self, term, doc_idx, idf_val):
        """单文档打分（保留给调试/单点使用；批量检索走 search 的向量化路径）。"""
        docs_arr, tf_arr = self.postings[term]
        pos = int(np.searchsorted(docs_arr, doc_idx)) if docs_arr[0] <= doc_idx else -1
        if pos >= len(docs_arr) or docs_arr[pos] != doc_idx:
            return 0.0
        tf = int(tf_arr[pos])
        dl = int(self.doc_len[doc_idx])
        k1, b, avgdl = self.k1, self.b, self.avgdl
        denom = tf + k1 * (1 - b + b * dl / avgdl) if avgdl > 0 else tf + k1
        tf_part = tf * (k1 + 1) / denom
        if self.delta > 0:
            return idf_val * (tf_part + self.delta)
        return idf_val * tf_part

    def search(self, query, top_k=10):
        """返回 [(doc_id, score), ...]，按分数降序。"""
        if self.N == 0:
            return []
        k1, b, k3, delta, avgdl = self.k1, self.b, self.k3, self.delta, self.avgdl
        dl = self._dl_float
        scores = np.zeros(self.N, dtype=np.float64)
        qtf = {}
        for t in tokenize(query):
            qtf[t] = qtf.get(t, 0) + 1
        for t, qcount in qtf.items():
            post = self.postings.get(t)
            if post is None:
                continue
            doc_idx, tf = post
            idf_val = self.idf(t)
            # BM25 词频饱和项（向量化）：idf * f*(k1+1) / (f + k1*(1-b+b*dl/avgdl))
            denom = tf + k1 * (1 - b + b * dl[doc_idx] / avgdl) if avgdl > 0 else tf + k1
            part = idf_val * tf * (k1 + 1) / denom
            if k3 > 0 and qcount > 1:  # 查询词频饱和项：f(q,Q)*(k3+1)/(f(q,Q)+k3)
                part = part * (qcount * (k3 + 1) / (qcount + k3))
            if delta > 0:  # BM25+ 补偿项
                part = part + idf_val * delta
            np.add.at(scores, doc_idx, part)
        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            return []
        k = min(top_k, nonzero.size)
        top = nonzero[np.argpartition(scores[nonzero], -k)[-k:]]
        # 按分数降序、doc_idx 升序排序（分数相同时保持确定性）
        order = top[np.lexsort((top, -scores[top]))]
        return [(self.doc_ids[int(i)], float(scores[i])) for i in order]
