"""参数词典：特征 → BM25 参数的查找表，未命中时回退到启发式规则。

设计动机（非参数最近邻 / 案例推理）：
- 用 grid search 在已有数据集上造（特征向量, 最优参数）条目；
- 预测时把新语料的特征归一化后找最近邻（可 top-k 平均）；
- 距离 ≤ match_threshold 视为"命中"，直接使用词典参数；
- 未命中则回退到 rule_predictor 的启发式公式（冷启动兜底）。

词典以 json 保存（含归一化统计量），无额外依赖。
"""

import json
import os

import numpy as np

from data_loader import load_dataset
from feature_extractor import extract_features

DEFAULT_DICT_PATH = os.path.join("results", "param_dictionary.json")

# 用于匹配的特征（P0/P1/P2 全部数值特征；缺失维度在计算距离时跳过）
FEATURE_KEYS = [
    "avgdl", "std_len", "cv_len", "len_skew",
    "len_p90", "len_p10", "len_ratio_p90_p10",
    "avg_ttr", "avg_max_tf", "length_tf_corr",
    "hapax_ratio", "vocab_size", "heaps_beta", "stopword_density",
    "zipf_alpha",
    "avg_query_len", "avg_query_max_tf", "query_repeat_ratio",
    "query_idf_mean", "query_idf_std", "query_oov_ratio",
]

CONTINUOUS_PARAMS = ("k1", "b", "k3", "delta")


class ParamDictionary:
    def __init__(self, entries, feature_keys=FEATURE_KEYS, k_neighbors=1,
                 match_threshold=2.0, normalize=True, reject_oov=True,
                 oov_margin=0.5, oov_min_features=2):
        """entries: [{"name", "features": dict, "params": dict}]"""
        self.feature_keys = list(feature_keys)
        self.k_neighbors = k_neighbors
        self.match_threshold = match_threshold
        self.normalize = normalize
        self.reject_oov = reject_oov
        self.oov_margin = oov_margin
        self.oov_min_features = oov_min_features
        self.entries = entries
        self.mean = {}
        self.std = {}
        self.fmin = {}
        self.fmax = {}
        if normalize and entries:
            for key in self.feature_keys:
                vals = [e["features"].get(key) for e in entries
                        if e["features"].get(key) is not None]
                if vals:
                    self.mean[key] = float(np.mean(vals))
                    self.std[key] = float(np.std(vals)) or 1.0
                    self.fmin[key] = float(np.min(vals))
                    self.fmax[key] = float(np.max(vals))

    @classmethod
    def load(cls, path=DEFAULT_DICT_PATH):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            data["entries"],
            feature_keys=data.get("feature_keys", FEATURE_KEYS),
            k_neighbors=data.get("k_neighbors", 1),
            match_threshold=data.get("match_threshold", 2.0),
            normalize=data.get("normalize", True),
            reject_oov=data.get("reject_oov", True),
            oov_margin=data.get("oov_margin", 0.5),
            oov_min_features=data.get("oov_min_features", 2),
        )

    def save(self, path=DEFAULT_DICT_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "feature_keys": self.feature_keys,
            "k_neighbors": self.k_neighbors,
            "match_threshold": self.match_threshold,
            "normalize": self.normalize,
            "reject_oov": self.reject_oov,
            "oov_margin": self.oov_margin,
            "oov_min_features": self.oov_min_features,
            "mean": self.mean,
            "std": self.std,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "entries": self.entries,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _vector(self, features):
        """归一化特征向量；缺失维度以 NaN 表示（距离计算时跳过）。"""
        vec = {}
        for key in self.feature_keys:
            v = features.get(key)
            if v is None:
                continue
            if self.normalize:
                mean = self.mean.get(key)
                std = self.std.get(key)
                if mean is not None and std:
                    v = (v - mean) / std
            vec[key] = v
        return vec

    def _distance(self, a, b):
        common = [k for k in a if k in b]
        if not common:
            return float("inf")
        return float(np.sqrt(np.mean([(a[k] - b[k]) ** 2 for k in common])))

    def is_oov(self, features):
        """是否分布外：≥ oov_min_features 个特征超出训练覆盖范围。

        单一特征贴边（如 nfcorpus 的 query_oov_ratio）不算分布外，
        避免误伤正常命中；多个特征同时越界才拒绝命中、回退启发式。
        """
        if not self.reject_oov:
            return False
        n_oov = 0
        for key in self.feature_keys:
            v = features.get(key)
            if v is None or key not in self.fmin:
                continue
            span = (self.fmax[key] - self.fmin[key]) or 1.0
            if v < self.fmin[key] - self.oov_margin * span or \
               v > self.fmax[key] + self.oov_margin * span:
                n_oov += 1
        return n_oov >= self.oov_min_features

    def lookup(self, features):
        """命中返回参数 dict；未命中返回 None。"""
        if self.is_oov(features):
            return None
        vec = self._vector(features)
        scored = []
        for e in self.entries:
            d = self._distance(vec, self._vector(e["features"]))
            scored.append((d, e))
        scored.sort(key=lambda x: x[0])
        if not scored or scored[0][0] > self.match_threshold:
            return None
        neighbors = scored[: self.k_neighbors]
        params = {}
        for p in CONTINUOUS_PARAMS:
            params[p] = float(np.mean([n[1]["params"][p] for n in neighbors]))
        # IDF 类型取最近邻的多数票
        idf_votes = {}
        for d, e in neighbors:
            idf_votes[e["params"]["idf_type"]] = idf_votes.get(e["params"]["idf_type"], 0) + 1
        params["idf_type"] = max(idf_votes, key=idf_votes.get)
        params["model_variant"] = "bm25+" if params["delta"] > 0 else "bm25"
        params["_dict_distance"] = round(scored[0][0], 4)
        params["_dict_match"] = [n[1]["name"] for n in neighbors]
        return params

    def probe(self, features):
        """匹配诊断（不改动命中逻辑）：是否 OOV、最近距离、最近条目。"""
        oov = self.is_oov(features)
        vec = self._vector(features)
        scored = sorted(
            (self._distance(vec, self._vector(e["features"])), e["name"])
            for e in self.entries
        )
        return {
            "oov": oov,
            "distance": round(scored[0][0], 4) if scored else None,
            "match": scored[0][1] if scored else None,
        }


def build_dictionary(benchmark_path=os.path.join("results", "benchmark_results.json"),
                     out_path=DEFAULT_DICT_PATH, k_neighbors=1, match_threshold=2.0,
                     mixes_benchmark_path=os.path.join("results", "benchmark_mixes.json")):
    """从 benchmark 结果（含 features + grid_best_params）构建词典。

    特征在构建时重新提取（保证包含查询侧特征），参数取 grid search 最优。
    """
    with open(benchmark_path, encoding="utf-8") as f:
        res = json.load(f)
    sources = [("", res)]
    if os.path.exists(mixes_benchmark_path):
        with open(mixes_benchmark_path, encoding="utf-8") as f:
            sources.append(("mix", json.load(f)))
    entries = []
    for prefix, source in sources:
        for name, r in source.items():
            if "grid_best_params" not in r:
                continue
            features = r.get("features")
            if features is None:
                docs, queries, _ = load_dataset(r["path"])
                features = extract_features(docs, queries)
            entry_name = f"{prefix}::{name}" if prefix else name
            entries.append({
                "name": entry_name,
                "features": features,
                "params": {
                    **{k: r["grid_best_params"].get(k, 0.0) for k in CONTINUOUS_PARAMS},
                    "idf_type": r["grid_best_params"]["idf_type"],
                },
                "grid_num_queries": r.get("grid_num_queries"),
                "grid_ndcg": r["grid_best_metrics"]["ndcg@10"],
            })
            print(f"[build_dictionary] {entry_name}: {features['doc_count']} docs, "
                  f"params {entries[-1]['params']}")
    dictionary = ParamDictionary(entries, k_neighbors=k_neighbors,
                                 match_threshold=match_threshold)
    dictionary.save(out_path)
    print(f"[build_dictionary] 已保存 {len(entries)} 个条目 -> {out_path}")
    return dictionary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=os.path.join("results", "benchmark_results.json"))
    ap.add_argument("--out", default=DEFAULT_DICT_PATH)
    ap.add_argument("--k-neighbors", type=int, default=1)
    ap.add_argument("--match-threshold", type=float, default=2.0)
    ap.add_argument("--mixes-benchmark",
                    default=os.path.join("results", "benchmark_mixes.json"))
    args = ap.parse_args()
    build_dictionary(args.benchmark, args.out, args.k_neighbors,
                     args.match_threshold, args.mixes_benchmark)
