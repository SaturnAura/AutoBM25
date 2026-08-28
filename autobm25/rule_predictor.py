"""基于统计特征的启发式参数预测（含词典查表 + 回退）。

公式（系数在包的 data/config.yaml，可由 research/ 标定后更新）：
  k1    = clip(k1_base - alpha_ttr*avg_ttr + alpha_maxtf*min(avg_max_tf/20, 1), 0.5, 3.0)
  b     = clip(b_base + beta_corr*length_tf_corr - beta_cv*cv_len, 0, 1)
  k3    = clip(k3_base*(avg_query_max_tf-1) + k3_rep*query_repeat_ratio, 0, 8)
  delta = clip(gamma_delta*cv_len, 0, 2.0)
  IDF   = smoothed（若 heaps_beta>0.7 或 hapax_ratio>0.6），否则 rsj
"""

import os

import yaml

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_PKG_DIR, "data", "config.yaml")
DICT_FILE = os.path.join(_PKG_DIR, "data", "param_dictionary.json")


def load_config():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def predict(features, config=None):
    """features: extract_features 的输出 -> 参数 dict"""
    config = config or load_config()
    c = config["coefficients"]
    k1 = clip(
        c["k1_base"]
        - c["alpha_ttr"] * features["avg_ttr"]
        + c["alpha_maxtf"] * min(features["avg_max_tf"] / 20.0, 1.0),
        c["k1_min"],
        c["k1_max"],
    )
    b = clip(
        c["b_base"]
        + c["beta_corr"] * features["length_tf_corr"]
        - c["beta_cv"] * features["cv_len"],
        c["b_min"],
        c["b_max"],
    )
    delta = clip(c["gamma_delta"] * features["cv_len"], c["delta_min"], c["delta_max"])
    k3 = clip(
        c["k3_base"] * max(features.get("avg_query_max_tf", 1.0) - 1.0, 0.0)
        + c["k3_rep"] * features.get("query_repeat_ratio", 0.0),
        c.get("k3_min", 0.0),
        c.get("k3_max", 8.0),
    )
    idf_type = (
        "smoothed"
        if (
            features["heaps_beta"] > c["heaps_beta_threshold"]
            or features["hapax_ratio"] > c["hapax_ratio_threshold"]
        )
        else "rsj"
    )
    model_variant = "bm25+" if delta > 0 else "bm25"
    return {
        "k1": round(float(k1), 4),
        "b": round(float(b), 4),
        "k3": round(float(k3), 4),
        "delta": round(float(delta), 4),
        "idf_type": idf_type,
        "model_variant": model_variant,
    }


def predict_with_dictionary(features, config=None, dictionary=None):
    """先查参数词典（命中返回词典参数），未命中回退到启发式规则。"""
    if dictionary is None:
        if os.path.exists(DICT_FILE):
            from .dictionary import ParamDictionary
            dictionary = ParamDictionary.load(DICT_FILE)
    if dictionary is not None:
        hit = dictionary.lookup(features)
        if hit is not None:
            return {k: hit[k] for k in ("k1", "b", "k3", "delta", "idf_type", "model_variant")}
    return predict(features, config)
