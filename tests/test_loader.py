import json

from autobm25 import load_dataset


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_standard_dataset_layout(tmp_path):
    _write_jsonl(
        tmp_path / "docs.jsonl",
        [{"id": 1, "text": "alpha beta"}, {"id": 2, "text": "gamma"}],
    )
    _write_jsonl(tmp_path / "queries.jsonl", [{"qid": "q1", "query": "alpha"}])
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"qid": "q1", "doc_id": 1, "relevance": 2}],
    )

    docs, queries, qrels = load_dataset(tmp_path)

    assert docs == [{"id": "1", "text": "alpha beta"}, {"id": "2", "text": "gamma"}]
    assert queries == [{"qid": "q1", "query": "alpha"}]
    assert qrels == [{"qid": "q1", "doc_id": "1", "relevance": 2}]


def test_beir_dataset_layout(tmp_path):
    _write_jsonl(
        tmp_path / "corpus.jsonl",
        [{"_id": "d1", "title": "Title", "text": "alpha beta"}],
    )
    _write_jsonl(tmp_path / "queries.jsonl", [{"_id": "q1", "text": "alpha"}])
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    (qrels_dir / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )

    docs, queries, qrels = load_dataset(tmp_path)

    assert docs == [{"id": "d1", "text": "Title alpha beta"}]
    assert queries == [{"qid": "q1", "query": "alpha"}]
    assert qrels == [{"qid": "q1", "doc_id": "d1", "relevance": 1}]
