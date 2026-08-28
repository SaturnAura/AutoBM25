"""AutoBM25: zero-label BM25 hyperparameter adaptation.

输入文档集 → 提取统计特征 → 参数大字典 O(1) 查表（OOD 防护 + 启发式回退）
→ 得到 BM25 超参数 → 建索引检索。
"""

import os

from .bm25_engine import BM25Engine, tokenize
from .dictionary import ParamDictionary
from .feature_extractor import extract_features
from .loader import load_dataset, load_dataset_subsampled
from .rule_predictor import predict, predict_with_dictionary

__version__ = "0.1.0"


class AutoBM25:
    """带参数大字典的 BM25 检索器（零标注、零调参）。

    用法：:

        from autobm25 import AutoBM25

        retriever = AutoBM25(docs)                      # docs: [{"id":..., "text":...}]
        retriever.params                               # 自动选出的超参数
        retriever.search("how does tax refund work")   # [(doc_id, score), ...]

    也可以直接从数据集目录构建：

        retriever = AutoBM25.from_dataset("dataset/fiqa")
    """

    def __init__(self, docs, queries=None, use_dictionary=True):
        if not docs:
            raise ValueError("docs 不能为空（需要至少一篇文档）")
        self.docs = docs
        self.features = extract_features(docs, queries or [])
        self.params = (
            predict_with_dictionary(self.features)
            if use_dictionary
            else predict(self.features)
        )
        self.engine = BM25Engine().build_index(docs)
        self.engine.set_params(
            k1=self.params["k1"],
            b=self.params["b"],
            k3=self.params["k3"],
            delta=self.params["delta"],
            idf_type=self.params["idf_type"],
        )

    @classmethod
    def from_dataset(cls, dataset_path, max_docs=None, use_dictionary=True):
        """从数据集目录构建（自动识别标准格式 / BEIR 格式）。"""
        if max_docs and os.path.exists(os.path.join(dataset_path, "corpus.jsonl")):
            docs, queries, _ = load_dataset_subsampled(dataset_path, max_docs)
        else:
            docs, queries, _ = load_dataset(dataset_path)
        return cls(docs, queries, use_dictionary=use_dictionary)

    def search(self, query, top_k=10):
        """返回 [(doc_id, score), ...]，按相关性降序。"""
        return self.engine.search(query, top_k=top_k)


__all__ = [
    "AutoBM25",
    "BM25Engine",
    "ParamDictionary",
    "extract_features",
    "load_dataset",
    "load_dataset_subsampled",
    "predict",
    "predict_with_dictionary",
    "tokenize",
    "__version__",
]
