from uuid import uuid4

from memsystem import concurrency_evaluation


def test_concurrency_case_reports_recall_latency_and_errors(monkeypatch):
    first, second = uuid4(), uuid4()

    def fake_search(database_url, claims, slot, query, limit):
        if query == "bad":
            return 0.003, [], "RuntimeError: failed"
        return 0.001, [first, second], None

    monkeypatch.setattr(concurrency_evaluation, "_search", fake_search)
    result = concurrency_evaluation._case(
        "database", None, None, ["good", "bad"], [{first, second}, {second}], 2, 2
    )

    assert result["clients"] == result["requests"] == 2
    assert result["errors"] == 1
    assert result["recall_at_k_mean"] == 0.5
    assert result["recall_at_k_min"] == 0.0
    assert result["latency_ms_p95"] == 2.9
