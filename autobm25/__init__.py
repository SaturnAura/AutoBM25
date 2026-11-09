"""AutoBM25: zero-label BM25 hyperparameter adaptation.

Feed a corpus in -> extract statistical features -> O(1) lookup in the parameter
dictionary (OOD guard + heuristic fallback) -> get BM25 hyperparameters -> index and search.
"""

import operator
import os

from . import _accel
from .bm25_engine import BM25Engine, tokenize
from .dictionary import ParamDictionary
from .feature_extractor import extract_features
from .loader import load_dataset, load_dataset_subsampled
from .rule_predictor import predict, predict_with_dictionary

__version__ = "0.1.1"

#: `"rust"` when the PyO3 extension is loaded, `"python"` otherwise.
BACKEND = _accel.BACKEND


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
        # Materialize list-like inputs such as pandas columns. A bare dict is
        # ambiguous, so it must be wrapped by the caller (or supplied as one
        # {"id", "text"} record); a bare string is naturally one document.
        if isinstance(docs, str):
            docs = [docs]
        elif isinstance(docs, dict):
            if "id" in docs and "text" in docs:
                docs = [docs]
            else:
                raise TypeError(
                    "docs must be a sequence of strings or {'id', 'text'} records; "
                    "wrap a bare mapping in a list"
                )
        else:
            docs = list(docs)
        if not docs:
            raise ValueError("docs must not be empty (need at least one document)")

        normalized = []
        for i, doc in enumerate(docs):
            if isinstance(doc, str):
                normalized.append({"id": str(i), "text": doc})
                continue
            try:
                doc_id = doc["id"]
                text = doc["text"]
            except (KeyError, TypeError) as exc:
                raise TypeError(
                    f"document {i} must be a string or a mapping with 'id' and 'text'"
                ) from exc
            normalized.append({"id": str(doc_id), "text": str(text)})
        docs = normalized

        if isinstance(queries, str):
            queries = [queries]
        elif queries is not None:
            queries = list(queries)
        self.docs = docs
        self.engine = BM25Engine().build_index(docs)
        # On the Rust backend the index build already collected every statistic the
        # feature vector needs, so this is a lookup rather than a second pass over
        # the corpus; the Python backend has to re-tokenize everything.
        if self.engine.uses_rust:
            self.features = self.engine.features(queries or [])
        else:
            self.features = extract_features(docs, queries or [])
        self.params = (
            predict_with_dictionary(self.features)
            if use_dictionary
            else predict(self.features)
        )
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
        return self.engine.search(query, top_k=_top_k_value(top_k))

    def search_many(self, queries, top_k=10, threads=None):
        """Batch variant of `search`.

        `queries` is a plain list of query strings; the result is one ranked list
        per query, in input order. On the Rust backend the batch runs in parallel
        with the GIL released.
        """
        if self.engine is None:
            raise RuntimeError(
                "no index yet: call index(corpus) first, or build via AutoBM25.from_dataset(path)"
            )
        return self.engine.search_many(
            list(queries), top_k=_top_k_value(top_k), threads=threads
        )

    @property
    def backend(self):
        """`"rust"` or `"python"` - which implementation this retriever is running."""
        return self.engine.backend() if self.engine is not None else _accel.BACKEND


__all__ = [
    "AutoBM25",
    "BACKEND",
    "BM25Engine",
    "ParamDictionary",
    "backend_info",
    "extract_features",
    "load_dataset",
    "load_dataset_subsampled",
    "predict",
    "predict_with_dictionary",
    "tokenize",
    "__version__",
]


def backend_info():
    """Which retrieval backend is loaded, and why (if it is not the Rust one)."""
    return _accel.backend_info()


def _top_k_value(top_k):
    """Validate a public result limit and return it as a non-negative integer."""
    try:
        value = operator.index(top_k)
    except TypeError as exc:
        raise TypeError("top_k must be an integer") from exc
    if value < 0:
        raise ValueError("top_k must be greater than or equal to zero")
    return value
