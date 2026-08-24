"""评估、grid search（上界/ground truth）、系数标定与对比报告。"""

import itertools
import json
import math
import os

import numpy as np
import yaml
from scipy import stats
from scipy.optimize import nnls
from tqdm import tqdm

from bm25_engine import BM25Engine, tokenize
from data_loader import load_dataset
from feature_extractor import extract_features, save_json as save_features_json
from rule_predictor import load_config, predict, save_config


def _relevant_map(qrels):
    """qid -> {doc_id: relevance}，只保留正例。"""
    rel = {}
    for r in qrels:
        if r["relevance"] > 0:
            rel.setdefault(r["qid"], {})[r["doc_id"]] = r["relevance"]
    return rel


class _QueryPlan:
    """grid search 快速路径：预先解析每条查询，把命中的倒排条目拼成
    大数组（docs/tf/dl/idf），每个参数组合只需一次向量化累加，
    避免逐词字典查找和多次 np.add.at。"""

    def __init__(self, engine, query):
        self.N = engine.N
        self.avgdl = engine.avgdl
        docs_list, tf_list, dl_list, idf_rsj, idf_smooth = [], [], [], [], []
        for t in tokenize(query):
            post = engine.postings.get(t)
            if post is None:
                continue
            d, tf = post
            df = engine.doc_freq[t]
            docs_list.append(d)
            tf_list.append(tf)
            dl_list.append(engine._dl_float[d])
            idf_rsj.append(np.full(d.shape, math.log((self.N - df + 0.5) / (df + 0.5))))
            idf_smooth.append(np.full(d.shape, math.log((self.N + 1) / (df + 1))))
        if not docs_list:
            self.docs = self.tf = self.dl = self.idf = None
        else:
            self.docs = np.concatenate(docs_list)
            self.tf = np.concatenate(tf_list)
            self.dl = np.concatenate(dl_list)
            self.idf = {
                "rsj": np.concatenate(idf_rsj),
                "smoothed": np.concatenate(idf_smooth),
            }

    def score(self, k1, b, delta, idf_type):
        """返回长度为 N 的分数向量（未匹配文档为 0）。"""
        if self.docs is None:
            return np.zeros(self.N, dtype=np.float64)
        avgdl = self.avgdl
        denom = self.tf + k1 * (1 - b + b * self.dl / avgdl) if avgdl > 0 else self.tf + k1
        part = self.idf[idf_type] * self.tf * (k1 + 1) / denom
        if delta > 0:  # BM25+ 补偿项
            part = part + delta * self.idf[idf_type]
        # bincount 加权累加（重复 doc 索引自动求和，比 np.add.at 快得多）
        return np.bincount(self.docs, weights=part, minlength=self.N)


def _query_metrics(ranked_ids, rels, top_k):
    """对单条查询的排序结果计算 MRR@10 / NDCG@10 / Recall@100。"""
    rr = 0.0
    for i, doc_id in enumerate(ranked_ids[:10]):
        if doc_id in rels:
            rr = 1.0 / (i + 1)
            break
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:10]):
        gain = rels.get(doc_id, 0)
        if gain > 0:
            dcg += gain / np.log2(i + 2)
    ideal = sorted(rels.values(), reverse=True)[:10]
    idcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(ideal))
    hit = 0
    for doc_id in ranked_ids[:100]:
        if doc_id in rels:
            hit += 1
    return rr, (dcg / idcg if idcg > 0 else 0.0), hit / len(rels)


def evaluate_engine(engine, queries, qrels, top_k=100):
    """对已 build 的引擎打分，返回平均指标。"""
    rel = _relevant_map(qrels)
    mrr, ndcg, recall = [], [], []
    for q in queries:
        rels = rel.get(q["qid"])
        if not rels:
            continue
        ranked_ids = [doc_id for doc_id, _ in engine.search(q["query"], top_k=top_k)]
        # MRR@10
        rr = 0.0
        for i, doc_id in enumerate(ranked_ids[:10]):
            if doc_id in rels:
                rr = 1.0 / (i + 1)
                break
        mrr.append(rr)
        # NDCG@10（支持分级相关性）
        dcg = 0.0
        for i, doc_id in enumerate(ranked_ids[:10]):
            gain = rels.get(doc_id, 0)
            if gain > 0:
                dcg += gain / np.log2(i + 2)
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        # Recall@100
        hit = sum(1 for doc_id in ranked_ids[:100] if doc_id in rels)
        recall.append(hit / len(rels))
    return {
        "mrr@10": float(np.mean(mrr)) if mrr else 0.0,
        "ndcg@10": float(np.mean(ndcg)) if ndcg else 0.0,
        "recall@100": float(np.mean(recall)) if recall else 0.0,
        "num_queries_evaluated": len(mrr),
    }


def evaluate(docs, queries, qrels, params, top_k=100):
    """评估单个参数组合。params: {k1, b, delta, idf_type}"""
    engine = BM25Engine().build_index(docs)
    engine.set_params(
        k1=params.get("k1"),
        b=params.get("b"),
        delta=params.get("delta"),
        idf_type=params.get("idf_type"),
    )
    return evaluate_engine(engine, queries, qrels, top_k=top_k)


def grid_search(docs, queries, qrels, search_space=None, metric="ndcg@10", top_k=100,
                limit_queries=None, progress=True, save_path=None):
    """遍历参数组合，返回最优参数及全部组合得分。

    索引只构建一次，每个组合只改 k1/b/delta/idf_type，避免重复建索引。
    """
    config = load_config()
    search_space = search_space or config["grid_search"]
    # 只把 list/tuple 字段当作搜索轴（排除 metric/top_k 等元信息）
    search_space = {k: v for k, v in search_space.items() if isinstance(v, (list, tuple))}
    qs = queries[:limit_queries] if limit_queries else queries
    combos = [
        dict(zip(search_space.keys(), vals))
        for vals in itertools.product(*search_space.values())
    ]
    engine = BM25Engine().build_index(docs)
    plans = [_QueryPlan(engine, q["query"]) for q in qs]
    rel = _relevant_map(qrels)
    doc_ids = engine.doc_ids
    results = []
    it = tqdm(combos, desc="grid search", mininterval=1.0) if progress else combos
    for params in it:
        k1, b, delta, idf_type = (
            params["k1"],
            params["b"],
            params["delta"],
            params["idf_type"],
        )
        mrr, ndcg, recall = [], [], []
        for plan, q in zip(plans, qs):
            rels = rel.get(q["qid"])
            if not rels:
                continue
            scores = plan.score(k1, b, delta, idf_type)
            nonzero = np.flatnonzero(scores)
            if nonzero.size == 0:
                mrr.append(0.0)
                ndcg.append(0.0)
                recall.append(0.0)
                continue
            k = min(top_k, nonzero.size)
            top = nonzero[np.argpartition(scores[nonzero], -k)[-k:]]
            order = top[np.lexsort((top, -scores[top]))]
            ranked_ids = [doc_ids[int(i)] for i in order]
            rr, ndcg_v, rec = _query_metrics(ranked_ids, rels, top_k)
            mrr.append(rr)
            ndcg.append(ndcg_v)
            recall.append(rec)
        entry = {
            "params": dict(params),
            "mrr@10": float(np.mean(mrr)) if mrr else 0.0,
            "ndcg@10": float(np.mean(ndcg)) if ndcg else 0.0,
            "recall@100": float(np.mean(recall)) if recall else 0.0,
            "num_queries_evaluated": len(mrr),
        }
        results.append(entry)
        if progress:
            it.set_postfix_str(f"best {metric}={max(r[metric] for r in results):.4f}")

    best = max(results, key=lambda r: r[metric])
    out = {
        "metric": metric,
        "num_combos": len(results),
        "num_queries_used": len(qs),
        "best_params": best["params"],
        "best_score": {k: best[k] for k in ("mrr@10", "ndcg@10", "recall@100")},
        "results": results,
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _fit_coefficients(samples):
    """用非负最小二乘拟合规则公式中的系数。

    约束符号方向由公式结构决定：
      k1    = c0 - alpha_ttr*ttr + alpha_maxtf*cap
      b     = c0 + beta_corr*corr - beta_cv*cv
      delta = gamma*cv
    """
    train = [s for s in samples if s.get("_train")]
    coefs = {}
    if len(train) >= 2:
        # k1
        X = np.array(
            [
                [1.0, -s["features"]["avg_ttr"], min(s["features"]["avg_max_tf"] / 20, 1)]
                for s in train
            ]
        )
        y = np.array([s["best"]["k1"] for s in train])
        w, _ = nnls(X, y)
        coefs.update(k1_base=float(w[0]), alpha_ttr=float(w[1]), alpha_maxtf=float(w[2]))
        # b
        X = np.array(
            [
                [1.0, s["features"]["length_tf_corr"], -s["features"]["cv_len"]]
                for s in train
            ]
        )
        y = np.array([s["best"]["b"] for s in train])
        w, _ = nnls(X, y)
        coefs.update(b_base=float(w[0]), beta_corr=float(w[1]), beta_cv=float(w[2]))
        # delta（无截距）
        X = np.array([[s["features"]["cv_len"]] for s in train])
        y = np.array([s["best"]["delta"] for s in train])
        w, _ = nnls(X, y)
        coefs["gamma_delta"] = float(w[0])
    return coefs


def _fit_idf_thresholds(samples):
    """在小网格上搜索 IDF 判断阈值，使训练集准确率最高。"""
    train = [s for s in samples if s.get("_train")]
    if len(train) < 2:
        return 0.7, 0.6
    best = (0.7, 0.6, -1.0)
    for ht in np.arange(0.4, 0.91, 0.1):
        for hr in np.arange(0.3, 0.81, 0.1):
            acc = 0.0
            for s in train:
                pred = (
                    "smoothed"
                    if s["features"]["heaps_beta"] > ht or s["features"]["hapax_ratio"] > hr
                    else "rsj"
                )
                acc += pred == s["best"]["idf_type"]
            acc /= len(train)
            if acc > best[2]:
                best = (round(float(ht), 2), round(float(hr), 2), acc)
    return best[0], best[1]


def calibrate(dataset_paths, out_dir="results", test_frac=0.2, limit_queries=None,
              progress=True, holdout=None):
    """标定流程：
    1. 每个数据集提取特征 + grid search 得到最优参数（ground truth）
    2. 收集 (特征向量, 最优参数) 配对
    3. 线性回归拟合规则系数（非负最小二乘），并搜索 IDF 阈值
    4. 更新 config.yaml
    5. 在未参与标定的数据集上输出对比报告（default / predicted / grid_search）
    """
    config = load_config()
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    for path in dataset_paths:
        name = os.path.basename(os.path.normpath(path))
        docs, queries, qrels = load_dataset(path)
        if not docs or not queries:
            print(f"[calibrate] 跳过空数据集: {path}")
            continue
        features = extract_features(docs, queries)
        gs = grid_search(
            docs, queries, qrels, config["grid_search"],
            metric=config["grid_search"]["metric"],
            top_k=config["grid_search"]["top_k"],
            limit_queries=limit_queries,
            progress=progress,
        )
        samples.append(
            {
                "name": name,
                "path": path,
                "features": features,
                "best": gs["best_params"],
                "best_score": gs["best_score"],
            }
        )
    if not samples:
        raise RuntimeError("没有可用的数据集用于标定")

    gt_path = os.path.join(out_dir, "calibrate_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[calibrate] ground truth 已保存: {gt_path}")

    # 划分标定集 / 测试集（固定随机种子保证可复现）
    rng = np.random.RandomState(0)
    order = rng.permutation(len(samples))
    n_test = max(1, int(round(len(samples) * test_frac)))
    if n_test >= len(samples):
        n_test = 1
    test_idx = set(order[:n_test].tolist())
    for i, s in enumerate(samples):
        s["_train"] = i not in test_idx

    coefs = _fit_coefficients(samples)
    if coefs:
        config["coefficients"].update(coefs)
        ht, hr = _fit_idf_thresholds(samples)
        config["coefficients"]["heaps_beta_threshold"] = ht
        config["coefficients"]["hapax_ratio_threshold"] = hr
        save_config(config)
        print("[calibrate] 已更新 config.yaml 系数:", coefs,
              f"| IDF 阈值: heaps>{ht}, hapax>{hr}")
    else:
        print("[calibrate] 标定样本太少，保持默认系数不变")

    # 对比报告：只在未参与标定的数据集上
    report = {"config_path": "config.yaml", "test_datasets": {}}
    for s in samples:
        if s["_train"]:
            continue
        docs, queries, qrels = load_dataset(s["path"])
        default_params = dict(config["default_params"])
        default_metrics = evaluate(docs, queries, qrels, default_params)
        pred_params = predict(s["features"])  # 读取更新后的 config.yaml
        pred_metrics = evaluate(docs, queries, qrels, pred_params)
        best_metrics = s["best_score"]
        row = {
            "default_params": default_params,
            "default": default_metrics,
            "predicted_params": pred_params,
            "predicted": pred_metrics,
            "grid_search_params": s["best"],
            "grid_search": best_metrics,
        }
        # predicted 相对 default 的提升百分比；predicted 达到 grid search 的百分比
        for metric in ("mrr@10", "ndcg@10", "recall@100"):
            base = default_metrics[metric]
            pred = pred_metrics[metric]
            grid = best_metrics[metric]
            row[f"improve_vs_default_{metric}"] = (
                (pred - base) / base * 100 if base > 0 else 0.0
            )
            row[f"reach_grid_{metric}"] = (pred / grid * 100 if grid > 0 else 0.0)
        report["test_datasets"][s["name"]] = row

    report_path = os.path.join(out_dir, "calibrate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[calibrate] 对比报告已保存: {report_path}")
    _print_report(report)
    return {"samples": len(samples), "test_datasets": len(report["test_datasets"]),
            "ground_truth": gt_path, "report": report_path}


def _print_report(report):
    print("\n=== 对比报告（测试数据集）===")
    for name, row in report["test_datasets"].items():
        d, p, g = row["default"], row["predicted"], row["grid_search"]
        print(f"\n[{name}]")
        print(f"  default    k1={row['default_params']['k1']} b={row['default_params']['b']} "
              f"delta={row['default_params']['delta']} idf={row['default_params']['idf_type']}")
        print(f"  predicted  k1={row['predicted_params']['k1']} b={row['predicted_params']['b']} "
              f"delta={row['predicted_params']['delta']} idf={row['predicted_params']['idf_type']}")
        print(f"  grid       k1={row['grid_search_params']['k1']} b={row['grid_search_params']['b']} "
              f"delta={row['grid_search_params']['delta']} idf={row['grid_search_params']['idf_type']}")
        for metric in ("mrr@10", "ndcg@10", "recall@100"):
            print(f"    {metric:>10}: default={d[metric]:.4f}  predicted={p[metric]:.4f}  "
                  f"grid={g[metric]:.4f}  "
                  f"(提升 {row[f'improve_vs_default_{metric}']:+.1f}%, 达上界 {row[f'reach_grid_{metric}']:.1f}%)")
