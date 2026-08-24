"""数据增强：通过变换文档集扩充（统计特征, 最优参数）样本。

每种变换生成一个标准格式的子数据集，保存到 dataset/augmented/<name>/ 下：
1. random_subsample     按比例随机采样文档（多个 ratio × 多个 seed）
2. length_bucket_split  按文档长度分桶 + 首尾桶跨桶组合（制造 cv_len 大/小样本）
3. truncate_docs        每篇文档截断到 max_len 词（doc_id/标注不变）
4. sliding_window       长文档滑动切片，原相关文档的所有切片都标为相关
"""

import json
import os
import random
import shutil

import numpy as np

from bm25_engine import tokenize
from data_loader import load_dataset, save_dataset
from rule_predictor import load_config


def _keep_qrels(qrels, valid_doc_ids):
    return [r for r in qrels if r["doc_id"] in valid_doc_ids]


def random_subsample(docs, queries, qrels, ratio, seed):
    """按比例随机采样文档，标注中只保留被采样到的 doc_id。"""
    rng = random.Random(seed)
    k = max(1, int(round(len(docs) * ratio)))
    sampled = set(rng.sample(range(len(docs)), k))
    new_docs = [docs[i] for i in sorted(sampled)]
    valid = {d["id"] for d in new_docs}
    return new_docs, queries, _keep_qrels(qrels, valid)


def length_bucket_split(docs, queries, qrels, n_buckets):
    """按文档长度排序后均分 n_buckets 个桶，返回 [(子集名, docs, queries, qrels)]。

    桶内长度接近 → cv_len 小；另生成首尾桶组合 → cv_len 大。
    """
    idx = sorted(range(len(docs)), key=lambda i: len(tokenize(docs[i]["text"])))
    chunks = np.array_split(idx, n_buckets)
    results = []
    for bi, chunk in enumerate(chunks):
        new_docs = [docs[i] for i in chunk]
        valid = {d["id"] for d in new_docs}
        results.append(
            (f"bucket_n{n_buckets}_b{bi}", new_docs, queries, _keep_qrels(qrels, valid))
        )
    if n_buckets >= 2:
        combo = list(chunks[0]) + list(chunks[-1])
        new_docs = [docs[i] for i in combo]
        valid = {d["id"] for d in new_docs}
        results.append(
            (f"cross_n{n_buckets}_first_last", new_docs, queries, _keep_qrels(qrels, valid))
        )
    return results


def truncate_docs(docs, queries, qrels, max_len):
    """每篇文档截断到 max_len 词，doc_id 与标注不变。"""
    new_docs = []
    for d in docs:
        toks = tokenize(d["text"])
        if len(toks) > max_len:
            toks = toks[:max_len]
        new_docs.append({"id": d["id"], "text": " ".join(toks)})
    return new_docs, queries, qrels


def sliding_window(docs, queries, qrels, window_size, stride=None):
    """长文档按固定窗口滑动切片，每片作为新文档。

    新 doc_id = 原id + "#" + 切片序号；原相关文档的所有切片都标为相关。
    """
    stride = stride or window_size
    new_docs = []
    slice_map = {}  # 原 doc_id -> [新 doc_id, ...]
    for d in docs:
        toks = tokenize(d["text"])
        if len(toks) < window_size:
            continue  # 短文档不切片（由 truncate 覆盖）
        for si, start in enumerate(range(0, len(toks) - window_size + 1, stride)):
            new_id = f"{d['id']}#{si}"
            slice_map.setdefault(d["id"], []).append(new_id)
            new_docs.append({"id": new_id, "text": " ".join(toks[start : start + window_size])})
    new_qrels = []
    for r in qrels:
        for sid in slice_map.get(r["doc_id"], []):
            new_qrels.append({"qid": r["qid"], "doc_id": sid, "relevance": r["relevance"]})
    return new_docs, queries, new_qrels


def augment_dataset(dataset_path, out_dir=None, config=None):
    """对 dataset/ 下的某个原始数据集应用全部变换，生成增强数据集。"""
    config = config or load_config()
    aug = config["augmentation"]
    name = os.path.basename(os.path.normpath(dataset_path))
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(dataset_path)), "augmented", name)
    # 重新生成前清空旧产物（仅针对增强输出目录，可随时重跑恢复）
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    docs, queries, qrels = load_dataset(dataset_path)
    generated = []

    # 1. 随机子采样：ratio x 3 个种子
    for ratio in aug["random_subsample"]["ratios"]:
        for seed in aug["random_subsample"]["seeds"]:
            d, q, r = random_subsample(docs, queries, qrels, ratio, seed)
            sub = os.path.join(out_dir, f"subsample_r{ratio}_s{seed}")
            save_dataset(sub, d, q, r)
            generated.append(sub)

    # 2. 长度分桶 + 首尾桶组合
    for nb in aug["length_bucket_split"]["n_buckets_list"]:
        for sub_name, d, q, r in length_bucket_split(docs, queries, qrels, nb):
            sub = os.path.join(out_dir, sub_name)
            save_dataset(sub, d, q, r)
            generated.append(sub)

    # 3. 截断
    for ml in aug["truncate_docs"]["max_lens"]:
        d, q, r = truncate_docs(docs, queries, qrels, ml)
        sub = os.path.join(out_dir, f"truncate_{ml}")
        save_dataset(sub, d, q, r)
        generated.append(sub)

    # 4. 滑动窗口
    for ws in aug["sliding_window"]["window_sizes"]:
        stride = max(1, int(ws * aug["sliding_window"]["stride_fraction"]))
        d, q, r = sliding_window(docs, queries, qrels, ws, stride=stride)
        sub = os.path.join(out_dir, f"window_{ws}")
        save_dataset(sub, d, q, r)
        generated.append(sub)

    with open(os.path.join(out_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"source": os.path.abspath(dataset_path), "datasets": generated}, f, ensure_ascii=False, indent=2)
    return generated
