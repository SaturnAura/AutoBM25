import pytest

from autobm25.metrics import evaluate_engine


class StubEngine:
    def search_many(self, queries, top_k=10):
        return [
            [("irrelevant", 2.0), ("relevant", 1.0)],
            [("relevant", 2.0), ("irrelevant", 1.0)],
        ]


def test_metrics_use_the_same_relevance_map_as_the_rankings():
    queries = [
        {"qid": "q1", "query": "one"},
        {"qid": "q2", "query": "two"},
    ]
    qrels = [
        {"qid": "q1", "doc_id": "relevant", "relevance": 1},
        {"qid": "q2", "doc_id": "relevant", "relevance": 1},
    ]

    metrics = evaluate_engine(StubEngine(), queries, qrels)

    assert metrics["mrr@10"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["ndcg@10"] == pytest.approx(((1 / 1.5849625) + 1.0) / 2)
    assert metrics["recall@100"] == 1.0
    assert metrics["num_queries_evaluated"] == 2
