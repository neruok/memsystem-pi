from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
import numpy as np

import memsystem.workload_evaluation as evaluator
from memsystem.workload_evaluation import (
    TASKS,
    _metrics,
    _prediction_metrics,
    _support_metrics,
    _top_rankings,
    load_dataset,
)
from memsystem.relevance_evaluation import Judgment


def _write_dataset(path):
    parquet.write_table(arrow.table({
        "id": ["doc-a", "doc-b"],
        "title": ["A", "B"],
        "text": ["alpha", "beta"],
    }), path / "corpus.parquet")
    (path / "REVISION").write_text("test-revision\n")
    for index, task in enumerate(TASKS):
        query_id = f"query-{index}"
        parquet.write_table(arrow.table({
            "id": [query_id],
            "text": [f"question {index}"],
        }), path / f"{task}-queries.parquet")
        parquet.write_table(arrow.table({
            "query-id": [query_id],
            "corpus-id": ["doc-b"],
            "score": [1],
        }), path / f"{task}-qrels.parquet")


def test_workload_dataset_loads_all_tasks(tmp_path):
    _write_dataset(tmp_path)

    corpus_ids, judgments, query_ids, tasks = load_dataset(tmp_path)

    assert corpus_ids == ["doc-a", "doc-b"]
    assert len(judgments) == len(TASKS)
    assert query_ids == [f"query-{index}" for index in range(len(TASKS))]
    assert tasks == list(TASKS)
    assert all(judgment.relevant == frozenset({1}) for judgment in judgments)


def test_relevance_dataset_accepts_custom_tasks(tmp_path):
    parquet.write_table(arrow.table({
        "id": ["doc"], "title": ["User"], "text": ["fact"],
    }), tmp_path / "corpus.parquet")
    parquet.write_table(arrow.table({
        "id": ["query"], "text": ["question"],
    }), tmp_path / "custom-queries.parquet")
    parquet.write_table(arrow.table({
        "query-id": ["query"], "corpus-id": ["doc"], "score": [1],
    }), tmp_path / "custom-qrels.parquet")

    corpus_ids, judgments, query_ids, tasks = load_dataset(tmp_path, ("custom",))

    assert corpus_ids == ["doc"]
    assert judgments == [Judgment("question", frozenset({0}))]
    assert query_ids == ["query"]
    assert tasks == ["custom"]


def test_prediction_metrics_require_correct_answer_and_relevant_citation(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text('[{"query_id":"q1","answer":"It is green.","citations":["doc-a"]},'
                    '{"query_id":"q2","answer":"blue","citations":["doc-b"]}]')
    assert _prediction_metrics(
        path,
        ["q1", "q2"],
        ["green", "red"],
        [Judgment("one", frozenset({0})), Judgment("two", frozenset({1}))],
        ["doc-a", "doc-b"],
    ) == {
        "normalized_reference_match": 0.5,
        "exact_citation_match": 1.0,
        "answer_and_citation_match": 0.5,
    }


def test_support_metrics_measure_answer_coverage_and_foreign_personas():
    assert _support_metrics(
        [[0, 1]], 2, ["The answer is green.", "unrelated"], ["green"],
        ["alice", "bob"], ["alice"],
    ) == {
        "reference_answer_token_coverage_at_2": 1.0,
        "mean_foreign_persona_fraction_at_2": 0.5,
    }


def test_workload_metrics_include_ndcg():
    documents = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    rankings = _top_rankings(documents, queries, 2)

    assert rankings == [[0, 1]]
    assert _metrics(rankings, [Judgment("q", frozenset({1}))], 2) == {
        "hit_rate_at_2": 1.0,
        "mean_recall_at_2": 1.0,
        "mean_reciprocal_rank_at_2": 0.5,
        "mean_ndcg_at_2": 0.63093,
    }


def test_workload_evaluation_caches_titled_documents(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    embedded_documents = []

    class Provider:
        def embed(self, texts):
            embedded_documents.extend(texts)
            return [
                np.array([1.0, 0.0]) if text.startswith("A") else np.array([0.0, 1.0])
                for text in texts
            ]

        def query_embed(self, texts):
            return [np.array([0.0, 1.0]) for _ in texts]

    provider = Provider()
    monkeypatch.setattr(
        evaluator, "_load_fastembed",
        lambda model_name, cache_dir: (provider, {
            "backend": "test", "fastembed": "test", "model_artifact_revision": "revision",
        }),
    )
    vector_cache = tmp_path / "vectors"
    report = evaluator.evaluate(
        tmp_path, "test-model", tmp_path, k=1, batch_size=2,
        vector_cache=vector_cache,
    )
    cached_report = evaluator.evaluate(
        tmp_path, "test-model", tmp_path, k=1, batch_size=2,
        vector_cache=vector_cache,
    )
    Path(report["embedding"]["document_cache"]).write_bytes(b"broken")
    repaired_report = evaluator.evaluate(
        tmp_path, "test-model", tmp_path, k=1, batch_size=2,
        vector_cache=vector_cache,
    )

    assert embedded_documents == ["B\nbeta", "A\nalpha"] * 2
    assert report["metrics"]["overall"]["hit_rate_at_1"] == 1.0
    assert set(report["metrics"]["tasks"]) == set(TASKS)
    assert report["queries"][0]["top"] == ["doc-b"]
    assert report["embedding"]["document_cache_hit"] is False
    assert cached_report["embedding"]["document_cache_hit"] is True
    assert repaired_report["embedding"]["document_cache_hit"] is False


def test_workload_rejects_unknown_qrel_query(tmp_path):
    _write_dataset(tmp_path)
    task = TASKS[0]
    parquet.write_table(arrow.table({
        "query-id": ["query-0", "missing-query"],
        "corpus-id": ["doc-b", "doc-a"],
        "score": [1, 1],
    }), tmp_path / f"{task}-qrels.parquet")

    with pytest.raises(ValueError, match="unknown query identifier"):
        load_dataset(tmp_path)


def test_workload_rejects_unknown_qrel_document(tmp_path):
    _write_dataset(tmp_path)
    task = TASKS[0]
    parquet.write_table(arrow.table({
        "query-id": ["query-0"],
        "corpus-id": ["missing"],
        "score": [1],
    }), tmp_path / f"{task}-qrels.parquet")

    with pytest.raises(ValueError, match="unknown corpus identifier"):
        load_dataset(tmp_path)
