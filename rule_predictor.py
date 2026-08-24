"""基于统计特征的启发式参数预测。

公式（系数在 config.yaml，可用 calibrate 重新标定）：
  k1    = clip(k1_base - alpha_ttr*avg_ttr + alpha_maxtf*min(avg_max_tf/20, 1), 0.5, 3.0)
  b     = clip(b_base + beta_corr*length_tf_corr - beta_cv*cv_len, 0, 1)
  delta = clip(gamma_delta*cv_len, 0, 2.0)
  IDF   = smoothed（若 heaps_beta>0.7 或 hapax_ratio>0.6），否则 rsj
"""

import json
import os

import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config, path=None):
    path = path or CONFIG_FILE
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def predict(features, config=None):
    """features: extract_features 的输出 -> 参数 dict"""
    config = config or load_config()
    c = config["coefficients"]

    # k1：TTR 越高（文档越精炼）→ k1 越小；avg_max_tf 越高（词频冗余大）→ k1 越大
    k1 = clip(
        c["k1_base"]
        - c["alpha_ttr"] * features["avg_ttr"]
        + c["alpha_maxtf"] * min(features["avg_max_tf"] / 20.0, 1.0),
        c["k1_min"],
        c["k1_max"],
    )
    # b：length_tf_corr 高（长文档冗余多）→ b 大；cv_len 高（长度差异大）→ b 小
    b = clip(
        c["b_base"]
        + c["beta_corr"] * features["length_tf_corr"]
        - c["beta_cv"] * features["cv_len"],
        c["b_min"],
        c["b_max"],
    )
    # delta：长度差异越大，越需要 BM25+ 补偿短文档
    delta = clip(c["gamma_delta"] * features["cv_len"], c["delta_min"], c["delta_max"])

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
        "delta": round(float(delta), 4),
        "idf_type": idf_type,
        "model_variant": model_variant,
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
