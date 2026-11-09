import pytest

import autobm25
from autobm25 import _accel
from autobm25.bm25_engine import BM25Engine
from autobm25.feature_extractor import extract_features


DOCS = [
    {"id": "1", "text": "alpha alpha beta"},
    {"id": "2", "text": "beta gamma"},
    {"id": "3", "text": "delta"},
]
QUERIES = ["alpha beta", "gamma", "missing"]


@pytest.mark.skipif(autobm25.BACKEND != "rust", reason="Rust extension is not built")
def test_python_and_rust_features_and_rankings_match(monkeypatch):
    monkeypatch.setattr(_accel, "ENABLED", False)
    python_engine = BM25Engine().build_index(DOCS)
    python_features = extract_features(DOCS, QUERIES)
    python_rankings = python_engine.search_many(QUERIES, top_k=3)

    monkeypatch.setattr(_accel, "ENABLED", True)
    rust_engine = BM25Engine().build_index(DOCS)
    rust_features = rust_engine.features(QUERIES)
    rust_rankings = rust_engine.search_many(QUERIES, top_k=3)

    assert python_features.keys() == rust_features.keys()
    for key, value in python_features.items():
        assert value == pytest.approx(rust_features[key], abs=1e-12)
    assert python_rankings == rust_rankings
