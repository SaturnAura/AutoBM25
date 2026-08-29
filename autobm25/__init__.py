"""AutoBM25: zero-label BM25 hyperparameter adaptation.

Feed a corpus in -> extract statistical features -> O(1) lookup in the parameter
dictionary (OOD guard + heuristic fallback) -> get BM25 hyperparameters -> index and search.
"""

import os

from .bm25_engine import BM25Engine, tokenize
from .dictionary import ParamDictionary
from .feature_extractor import extract_features
from .loader import load_dataset, load_dataset_subsampled
from .rule_predictor import predict, predict_with_dictionary

__version__ = "0.1.0"


class AutoBM25:
    """BM25 retriever backed by the parameter dictionary (zero-label, zero-tuning).

    Usage::

        from autobm25 import AutoBM25

        corpus = [
            "BM25 is a probabilistic ranking function used in information retrieval.",
            "Automatic parameter tuning improves retrieval effectiveness.",
        ]
        retriever = AutoBM25()
        retriever.index(corpus)                          # index a plain list of strings
        retriever.params                                 # automatically chosen hyperparameters
        retriever.search("parameter tuning", top_k=5)    # [(doc_id, score), ...]

    It can also be built directly from a dataset directory:

        retriever = AutoBM25.from_dataset("dataset/fiqa")
    """

    def __init__(self, docs=None, queries=None, use_dictionary=True):
        self.docs = None
        self.features = None
        self.params = None
        self.engine = None
        if docs is not None:
            self.index(docs, queries, use_dictionary=use_dictionary)

    def index(self, docs, queries=None, use_dictionary=True):
        """Build the index. docs: [{"id","text"}] or a plain list of strings;
        queries is optional.

        Before indexing, this automatically runs: statistical feature extraction
        -> parameter dictionary lookup (OOD guard + heuristic fallback)
        -> BM25 hyperparameters. No labels are needed.
        """
        if not docs:
            raise ValueError("docs must not be empty (need at least one document)")
        if isinstance(docs[0], str):
            docs = [{"id": str(i), "text": t} for i, t in enumerate(docs)]
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
        return self

    @classmethod
    def from_dataset(cls, dataset_path, max_docs=None, use_dictionary=True):
        """Build from a dataset directory (standard / BEIR formats auto-detected)."""
        if max_docs and os.path.exists(os.path.join(dataset_path, "corpus.jsonl")):
            docs, queries, _ = load_dataset_subsampled(dataset_path, max_docs)
        else:
            docs, queries, _ = load_dataset(dataset_path)
        return cls(docs, queries, use_dictionary=use_dictionary)

    def search(self, query, top_k=10):
        """Return [(doc_id, score), ...] sorted by descending relevance."""
        if self.engine is None:
            raise RuntimeError(
                "no index yet: call index(corpus) first, or build via AutoBM25.from_dataset(path)"
            )
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
