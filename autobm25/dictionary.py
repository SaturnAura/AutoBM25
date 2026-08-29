"""参数大字典（检索侧）：加载、O(1) 最近邻查表、OOD 防护、probe。

字典的构建（build_dictionary，grid search 造条目）属于复现流程，
在本地 research/ 中完成；本模块只负责"用字典"。
"""

import json
import os

import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DICT_PATH = os.path.join(_PKG_DIR, "data", "param_dictionary.json")

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
        self.feature_keys = list(feature_keys)
        self.k_neighbors = k_neighbors
        self.match_threshold = match_threshold
        self.normalize = normalize
        self.reject_oov = reject_oov
        self.oov_margin = oov_margin
        self.oov_min_features = oov_min_features
        self.entries = entries
        self.mean, self.std, self.fmin, self.fmax = {}, {}, {}, {}
        if entries:
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

    def _vector(self, features):
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
        """是否分布外：≥ oov_min_features 个特征超出训练覆盖范围。"""
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
        """命中返回参数 dict；未命中（距离超阈值或 OOD）返回 None。"""
        if self.is_oov(features):
            return None
        vec = self._vector(features)
        scored = sorted(
            ((self._distance(vec, self._vector(e["features"])), e["name"], e)
             for e in self.entries),
            key=lambda x: (x[0], x[1]),  # 距离并列时按条目名排序，避免比较 dict
        )
        if not scored or scored[0][0] > self.match_threshold:
            return None
        neighbors = scored[: self.k_neighbors]
        params = {}
        for p in CONTINUOUS_PARAMS:
            params[p] = float(np.mean([n[2]["params"][p] for n in neighbors]))
        idf_votes = {}
        for _, _, e in neighbors:
            idf_votes[e["params"]["idf_type"]] = idf_votes.get(e["params"]["idf_type"], 0) + 1
        params["idf_type"] = max(idf_votes, key=idf_votes.get)
        params["model_variant"] = "bm25+" if params["delta"] > 0 else "bm25"
        params["_dict_distance"] = round(scored[0][0], 4)
        params["_dict_match"] = [n[1] for n in neighbors]
        return params

    def probe(self, features):
        """匹配诊断：是否 OOD、最近距离、最近条目。"""
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
