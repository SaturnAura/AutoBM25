"""Statistical feature extraction.

Takes a document collection (optionally with queries) and outputs a feature dict.
Features are tiered by priority:
  P0 (required): doc_count, avgdl, std_len, cv_len, avg_ttr, avg_max_tf, length_tf_corr
  P1 (recommended): len_skew, len_p90, len_p10, len_ratio_p90_p10, hapax_ratio,
                    vocab_size, heaps_beta, stopword_density
  P2 (optional): zipf_alpha, avg_query_len, query_idf_mean, query_idf_std, query_oov_ratio

Feature-to-hyperparameter rationale (heuristic basis):
- Document-length features -> b: high cv_len (large length variance) -> smaller b;
  high length_tf_corr (long docs are redundant) -> larger b
- In-document term-frequency features -> k1: high TTR (concise docs) -> smaller k1;
  high avg_max_tf (high TF ceiling) -> larger k1
- Global vocabulary distribution -> IDF type: high heaps_beta / hapax_ratio
  (sparse, multi-domain vocabulary) -> Smoothed-IDF
"""

import json
import math
import os

import numpy as np
from scipy import stats

from .bm25_engine import tokenize

STOPWORDS_FILE = os.path.join(os.path.dirname(__file__), "data", "stopwords.txt")


def load_stopwords():
    with open(STOPWORDS_FILE, encoding="utf-8") as f:
        return {w.strip() for w in f if w.strip()}


def extract_features(docs, queries=None, stopwords=None, heaps_points=50):
    """docs: [{"id": str, "text": str}], queries: optional [{"qid", "query"}] -> feature dict"""
    if stopwords is None:
        stopwords = load_stopwords()

    lens = []        # token count per document
    ttrs = []        # unique/total per document
    max_tfs = []     # max term frequency per document (stopwords removed)
    mean_tfs = []    # mean term frequency per document = total/unique (target of length-TF correlation)
    doc_counts = []  # term -> tf per document
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

    # ---- Dimension 1: document length & distribution features (determine b) ----
    avgdl = float(np.mean(lens)) if lens else 0.0
    std_len = float(np.std(lens, ddof=1)) if len(lens) > 1 else 0.0
    cv_len = std_len / avgdl if avgdl > 0 else 0.0
    len_skew = float(stats.skew(lens, bias=False)) if len(lens) > 2 else 0.0
    if math.isnan(len_skew):
        len_skew = 0.0
    len_p90 = float(np.percentile(lens, 90)) if lens else 0.0
    len_p10 = float(np.percentile(lens, 10)) if lens else 0.0
    # Tail heaviness: p90/p10; floor p10 at 1 to avoid division by zero
    len_ratio_p90_p10 = len_p90 / max(len_p10, 1.0)

    # ---- Dimension 2: in-document term-frequency & redundancy features (determine k1) ----
    avg_ttr = float(np.mean(ttrs)) if ttrs else 0.0
    avg_max_tf = float(np.mean(max_tfs)) if max_tfs else 0.0
    # length_tf_corr: Pearson correlation between doc length and mean in-doc TF
    # (long docs being more redundant -> positive correlation)
    if (
        len(lens) > 2
        and np.std(lens) > 0
        and np.std(mean_tfs) > 0
    ):
        corr, _ = stats.pearsonr(lens, mean_tfs)
        length_tf_corr = float(corr) if not math.isnan(corr) else 0.0
    else:
        length_tf_corr = 0.0

    # ---- Dimension 3: global vocabulary distribution features (determine IDF type) ----
    vocab_freq = {}
    doc_freq = {}  # term -> number of documents containing it (df, for query-side IDF)
    for counts in doc_counts:
        for t, f in counts.items():
            vocab_freq[t] = vocab_freq.get(t, 0) + f
            doc_freq[t] = doc_freq.get(t, 0) + 1
    vocab_size = len(vocab_freq)
    hapax_ratio = (
        sum(1 for f in vocab_freq.values() if f == 1) / vocab_size
        if vocab_size
        else 0.0
    )

    # Zipf's law: log(freq) = log(C) - alpha * log(rank); |slope| is alpha
    zipf_alpha = 0.0
    if len(vocab_freq) >= 2:
        freqs = np.array(sorted(vocab_freq.values(), reverse=True), dtype=np.float64)
        ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
        slope, _, _, _, _ = stats.linregress(np.log(ranks), np.log(freqs))
        zipf_alpha = float(abs(slope)) if not math.isnan(slope) else 0.0

    # Heaps' Law: log(V) = log(K) + beta * log(n); the fitted slope is beta.
    # Accumulate (cumulative tokens n, cumulative vocabulary V) per document,
    # then sample at most heaps_points points on a log-spaced grid.
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
        "doc_count": len(docs),
        "num_docs": len(docs),  # legacy alias
        "total_tokens": int(total_tokens),
        # Dimension 1
        "avgdl": float(avgdl),
        "std_len": float(std_len),
        "cv_len": float(cv_len),
        "len_skew": float(len_skew),
        "len_p90": float(len_p90),
        "len_p10": float(len_p10),
        "len_ratio_p90_p10": float(len_ratio_p90_p10),
        # Dimension 2
        "avg_ttr": float(avg_ttr),
        "avg_max_tf": float(avg_max_tf),
        "length_tf_corr": float(length_tf_corr),
        "hapax_ratio": float(hapax_ratio),
        # Dimension 3
        "vocab_size": int(vocab_size),
        "heaps_beta": float(heaps_beta),
        "stopword_density": float(stopword_density),
        "zipf_alpha": float(zipf_alpha),
    }

    # ---- Dimension 4: query-side features (optional, computed when queries exist) ----
    if queries:
        q_lens, q_idfs, q_max_tfs = [], [], []
        q_oov = 0
        total_q_tokens = 0
        q_with_repeat = 0
        q_count = 0
        for q in queries:
            toks = tokenize(q["query"])
            if not toks:
                continue
            q_lens.append(len(toks))
            q_count += 1
            qtf = {}
            for t in toks:
                qtf[t] = qtf.get(t, 0) + 1
            q_max_tfs.append(max(qtf.values()))
            if max(qtf.values()) > 1:
                q_with_repeat += 1
            total_q_tokens += len(toks)
            for t in toks:
                if t in doc_freq:
                    df = doc_freq[t]
                    q_idfs.append(math.log((len(docs) - df + 0.5) / (df + 0.5)))
                else:
                    q_oov += 1
        features["avg_query_len"] = float(np.mean(q_lens)) if q_lens else 0.0
        features["avg_query_max_tf"] = float(np.mean(q_max_tfs)) if q_max_tfs else 0.0
        features["query_repeat_ratio"] = q_with_repeat / q_count if q_count else 0.0
        features["query_oov_ratio"] = q_oov / total_q_tokens if total_q_tokens else 0.0
        features["query_idf_mean"] = float(np.mean(q_idfs)) if q_idfs else 0.0
        features["query_idf_std"] = (
            float(np.std(q_idfs, ddof=1)) if len(q_idfs) > 1 else 0.0
        )
    return features


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
