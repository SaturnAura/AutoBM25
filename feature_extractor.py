"""统计特征提取。

输入文档集，输出特征 dict：
- 文档长度特征（决定 b）：avgdl, std_len, cv_len, len_skew
- 文档内词频特征（决定 k1）：avg_ttr, avg_max_tf, length_tf_corr
- 全局词汇分布特征（决定 IDF 类型）：vocab_size, hapax_ratio, heaps_beta, stopword_density
"""

import json
import math
import os

import numpy as np
from scipy import stats

from bm25_engine import tokenize

STOPWORDS_FILE = os.path.join(os.path.dirname(__file__), "stopwords.txt")


def load_stopwords():
    with open(STOPWORDS_FILE, encoding="utf-8") as f:
        return {w.strip() for w in f if w.strip()}


def extract_features(docs, stopwords=None, heaps_points=50):
    """docs: [{"id": str, "text": str}] -> 特征 dict"""
    if stopwords is None:
        stopwords = load_stopwords()

    lens = []        # 每篇文档词数
    ttrs = []        # 每篇文档 unique/total
    max_tfs = []     # 每篇文档去停用词后的最大词频
    mean_tfs = []    # 每篇文档的平均词频 = 总词数/唯一词数（长度-词频相关的被解释变量）
    doc_counts = []  # 每篇文档的 term -> tf
    total_tokens = 0
    stopword_tokens = 0

    for d in docs:
        toks = tokenize(d["text"])
        lens.append(len(toks))
        total_tokens += len(toks)
        if not toks:
            ttrs.append(0.0)
            max_tfs.append(0.0)
            mean_tfs.append(0.0)
            doc_counts.append({})
            continue
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
            if t in stopwords:
                stopword_tokens += 1
        uniq = len(counts)
        ttrs.append(uniq / len(toks))
        non_stop = {t: f for t, f in counts.items() if t not in stopwords}
        max_tfs.append(max(non_stop.values()) if non_stop else 0.0)
        mean_tfs.append(len(toks) / uniq)
        doc_counts.append(counts)

    avgdl = float(np.mean(lens)) if lens else 0.0
    std_len = float(np.std(lens, ddof=1)) if len(lens) > 1 else 0.0
    cv_len = std_len / avgdl if avgdl > 0 else 0.0
    len_skew = float(stats.skew(lens, bias=False)) if len(lens) > 2 else 0.0
    if math.isnan(len_skew):
        len_skew = 0.0

    avg_ttr = float(np.mean(ttrs)) if ttrs else 0.0
    avg_max_tf = float(np.mean(max_tfs)) if max_tfs else 0.0
    # length_tf_corr：文档长度 与 文档内平均词频 的皮尔逊相关系数（长文档冗余多 → 正相关）
    if (
        len(lens) > 2
        and np.std(lens) > 0
        and np.std(mean_tfs) > 0
    ):
        corr, _ = stats.pearsonr(lens, mean_tfs)
        length_tf_corr = float(corr) if not math.isnan(corr) else 0.0
    else:
        length_tf_corr = 0.0

    # 全局词汇表与 hapax（全文只出现一次的词）
    vocab_freq = {}
    for counts in doc_counts:
        for t, f in counts.items():
            vocab_freq[t] = vocab_freq.get(t, 0) + f
    vocab_size = len(vocab_freq)
    hapax_ratio = (
        sum(1 for f in vocab_freq.values() if f == 1) / vocab_size
        if vocab_size
        else 0.0
    )

    # Heaps' Law: log(V) = log(K) + beta * log(n)，线性拟合斜率即 beta。
    # 逐文档累计 (累计词数 n, 累计词汇量 V)，再对数间隔采样至多 heaps_points 个点。
    points = []
    vocab_seen = set()
    n = 0
    for counts in doc_counts:
        for t, f in counts.items():
            n += f
        vocab_seen.update(counts.keys())
        points.append((n, len(vocab_seen)))
    if len(points) > heaps_points and len(points) > 1:
        idxs = np.unique(
            np.round(np.logspace(0, math.log10(len(points)), heaps_points)).astype(int)
            - 1
        )
        idxs = np.clip(idxs, 0, len(points) - 1)
        points = [points[int(i)] for i in idxs]
    n_vals = [math.log(p[0]) for p in points if p[0] > 0]
    v_vals = [math.log(p[1]) for p in points if p[0] > 0]
    if len(n_vals) >= 2 and len(set(n_vals)) >= 2:
        slope, _, _, _, _ = stats.linregress(n_vals, v_vals)
        heaps_beta = float(slope)
    else:
        heaps_beta = 0.0

    stopword_density = stopword_tokens / total_tokens if total_tokens else 0.0

    features = {
        "num_docs": len(docs),
        "total_tokens": int(total_tokens),
        "avgdl": float(avgdl),
        "std_len": float(std_len),
        "cv_len": float(cv_len),
        "len_skew": float(len_skew),
        "avg_ttr": float(avg_ttr),
        "avg_max_tf": float(avg_max_tf),
        "length_tf_corr": float(length_tf_corr),
        "vocab_size": int(vocab_size),
        "hapax_ratio": float(hapax_ratio),
        "heaps_beta": float(heaps_beta),
        "stopword_density": float(stopword_density),
    }
    return features


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
