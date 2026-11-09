import itertools

import pytest

import autobm25
from autobm25 import AutoBM25


CORPUS = [
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Automatic parameter tuning improves retrieval effectiveness.",
]


def test_readme_quickstart_contract():
    retriever = AutoBM25()
    retriever.index(CORPUS)

    assert retriever.params["k1"] >= 0
    assert retriever.params["b"] >= 0
    assert retriever.search("parameter tuning", top_k=5)
    assert autobm25.BACKEND in {"python", "rust"}


def test_list_like_and_generator_inputs():
    column = (text for text in CORPUS)
    retriever = AutoBM25().index(column)

    assert len(retriever.docs) == 2
    assert [doc["id"] for doc in retriever.docs] == ["0", "1"]
    assert retriever.search("ranking function", top_k=1)[0][0] == "0"


def test_single_string_and_single_record():
    by_string = AutoBM25().index("alpha beta")
    by_record = AutoBM25().index({"id": 7, "text": "alpha beta"})

    assert by_string.docs == [{"id": "0", "text": "alpha beta"}]
    assert by_record.docs == [{"id": "7", "text": "alpha beta"}]


def test_pandas_column_is_supported():
    pandas = pytest.importorskip("pandas")
    retriever = AutoBM25().index(pandas.Series(CORPUS))

    assert len(retriever.docs) == 2
    assert retriever.search("ranking function", top_k=1)


def test_document_ids_match_across_backends():
    retriever = AutoBM25().index([{"id": 1, "text": "alpha"}, {"id": 2, "text": "beta"}])

    assert retriever.docs[0]["id"] == "1"
    assert all(isinstance(doc_id, str) for doc_id, _ in retriever.search("alpha", top_k=2))


@pytest.mark.parametrize("top_k", [0, 1, 10, 100])
def test_top_k_boundaries(top_k):
    retriever = AutoBM25().index(CORPUS)
    assert len(retriever.search("document", top_k=top_k)) <= top_k
    assert all(len(hits) <= top_k for hits in retriever.search_many(["document"], top_k=top_k))


def test_bad_top_k_is_rejected_consistently():
    retriever = AutoBM25().index(CORPUS)

    with pytest.raises(ValueError, match="greater than or equal to zero"):
        retriever.search("document", top_k=-1)
    with pytest.raises(TypeError, match="integer"):
        retriever.search("document", top_k=1.5)
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        retriever.search_many(["document"], top_k=-1)


def test_empty_or_malformed_corpus_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        AutoBM25().index([])
    with pytest.raises(TypeError, match="document 0"):
        AutoBM25().index([{"text": "missing id"}])


def test_search_before_index_is_an_error():
    with pytest.raises(RuntimeError, match="no index yet"):
        AutoBM25().search("anything")
