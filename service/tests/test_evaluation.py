import numpy as np
import pytest

from memsystem.evaluation import (
    _aggregate_reports,
    _configure_postgres,
    _exact_rerank,
    _metrics,
    _postgres_statement,
    _recall,
    _resolve_candidate_multipliers,
    _resolve_seeds,
    _run_turbovec,
    _status_memory_bytes,
    make_corpus,
)


def test_evaluation_corpus_is_reproducible_normalized_and_filtered():
    first = make_corpus(200, 5, 16, 10, 7)
    second = make_corpus(200, 5, 16, 10, 7)

    assert np.array_equal(first[0], second[0])
    assert np.allclose(np.linalg.norm(first[0], axis=1), 1)
    assert np.allclose(np.linalg.norm(first[1], axis=1), 1)
    assert {name: len(ids) for name, ids in first[2].items()} == {
        "unfiltered": 200,
        "broad_contiguous_50pct": 100,
        "scattered_random_10pct": 20,
    }
    assert _recall([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 3)
    assert _metrics([0.1, 0.2], [0.5, 1.0])["recall_at_k"] == 0.75

    with pytest.raises(ValueError, match="items >= 200"):
        make_corpus(199, 5, 16, 10, 7)


def test_exact_rerank_uses_full_precision_vectors_and_external_ids():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)

    candidates = [3, 2, 1]
    reranked = _exact_rerank(
        vectors, np.array([1.0, 0.0], dtype=np.float32), candidates, 2
    )

    assert reranked == [1, 2]
    assert _recall([1, 2], candidates) == 1.0
    assert _recall([1, 2], reranked) == 1.0
    assert _resolve_candidate_multipliers(None) == ()
    assert _resolve_candidate_multipliers([1, 2, 3, 5, 10]) == (1, 2, 3, 5, 10)
    for invalid in ([3, 11], [2.5], [True]):
        with pytest.raises(ValueError, match="unique integers from 1 through 10"):
            _resolve_candidate_multipliers(invalid)


def test_turbovec_benchmark_emits_bounded_exact_reranking_variants(tmp_path):
    vectors, queries, cases = make_corpus(200, 5, 16, 10, 7)
    ground_truth = {}
    for name, ids in cases.items():
        ground_truth[name] = [
            ids[np.lexsort((ids, -(vectors[ids - 1] @ query)))[:5]].tolist()
            for query in queries
        ]

    reports = _run_turbovec(
        vectors, queries, cases, 5, 4, ground_truth, tmp_path, (1, 2)
    )

    assert set(reports) == {"", "_reranked_1x", "_reranked_2x"}
    assert reports[""]["settings"] == {
        "candidate_multiplier": 3,
        "reranking": "none",
    }
    for suffix in ("_reranked_1x", "_reranked_2x"):
        for metrics in reports[suffix]["cases"].values():
            assert metrics["candidate_recall_at_k"] >= metrics["recall_at_k"]


def test_multi_seed_summary_reports_mean_and_worst_decision_signals():
    def report(seed, recall, latency, rss):
        return {
            "corpus": {"seed": seed},
            "engines": {"turbovec_4bit": {
                "process_peak_rss_delta_bytes": rss,
                "cases": {"unfiltered": {
                    "recall_at_k": recall,
                    "latency_ms_p95": latency,
                }},
            }},
        }

    summary = _aggregate_reports([
        report(7, 0.9, 4.0, 100),
        report(17, 0.8, 6.0, 140),
    ])

    assert _resolve_seeds(7, None) == [7]
    assert _resolve_seeds(7, [7, 17]) == [7, 17]
    with pytest.raises(ValueError, match="unique non-negative"):
        _resolve_seeds(7, [7, 7])
    assert summary == {
        "runs": 2,
        "engines": {"turbovec_4bit": {
            "process_peak_rss_delta_bytes_mean": 120,
            "process_peak_rss_delta_bytes_max": 140,
            "cases": {"unfiltered": {
                "recall_at_k_mean": 0.85,
                "recall_at_k_min": 0.8,
                "latency_ms_p95_mean": 5.0,
                "latency_ms_p95_max": 6.0,
            }},
        }},
    }


def test_multi_seed_summary_includes_candidate_recall_when_present():
    reports = [{
        "engines": {"turbovec_4bit_reranked_3x": {
            "process_peak_rss_delta_bytes": rss,
            "cases": {"unfiltered": {
                "recall_at_k": recall,
                "candidate_recall_at_k": candidate_recall,
                "latency_ms_p95": 2.0,
            }},
        }},
    } for rss, recall, candidate_recall in (
        (100, 0.9, 0.95),
        (120, 0.8, 0.9),
    )]

    case = _aggregate_reports(reports)["engines"][
        "turbovec_4bit_reranked_3x"
    ]["cases"]["unfiltered"]

    assert case["candidate_recall_at_k_mean"] == 0.925
    assert case["candidate_recall_at_k_min"] == 0.9


def test_process_status_memory_parser_reads_peak_rss_and_rejects_bad_signals():
    status = "Name:\tpostgres\nVmRSS:\t42 kB\nVmHWM:\t84 kB\n"

    assert _status_memory_bytes(status, "VmHWM") == 84 * 1024
    with pytest.raises(ValueError, match="VmHWM missing"):
        _status_memory_bytes("Name:\tpostgres\n", "VmHWM")
    with pytest.raises(ValueError, match="unexpected VmHWM unit"):
        _status_memory_bytes("VmHWM:\t84 MB\n", "VmHWM")


def test_hnsw_configuration_pins_memory_and_recall_relevant_settings():
    statements = []

    class Connection:
        def execute(self, statement, parameters=None):
            statements.append((statement, parameters))

    _configure_postgres(Connection(), exact=False)

    sql = "\n".join(statement for statement, _ in statements)
    assert "max_parallel_workers_per_gather = 0" in sql
    assert "max_parallel_maintenance_workers = 0" in sql
    assert "hnsw.ef_search = 40" in sql
    assert "hnsw.iterative_scan = 'strict_order'" in sql


def test_postgres_statement_omits_full_corpus_filter_and_supports_plan_checks():
    assert "WHERE" not in _postgres_statement(False)
    statement = _postgres_statement(True, explain=True)
    assert statement.startswith("EXPLAIN (FORMAT JSON)")
    assert "WHERE id = ANY(%s)" in statement
