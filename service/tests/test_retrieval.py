from uuid import UUID

import pytest

from memsystem.retrieval import (
    LexicalResult,
    VectorResult,
    _vector_allowlist_limit,
    fuse_ranked_results,
)


def _lexical(chunk: int, score: float) -> LexicalResult:
    identifier = UUID(int=chunk)
    return LexicalResult(
        identifier, identifier, 1, f"Document {chunk}", [], f"text {chunk}",
        0, 6, score,
    )


def _vector(chunk: int, score: float) -> VectorResult:
    identifier = UUID(int=chunk)
    return VectorResult(
        identifier, identifier, 1, f"Document {chunk}", [], f"text {chunk}",
        0, 6, score,
    )


def test_reciprocal_rank_fusion_combines_chunks_and_preserves_scores():
    results = fuse_ranked_results(
        [_lexical(1, 0.9), _lexical(2, 0.8)],
        [_vector(2, 0.7), _vector(3, 0.6)],
    )

    assert [item.chunk_id.int for item in results] == [2, 1, 3]
    assert results[0].lexical_score == 0.8
    assert results[0].vector_score == 0.7
    assert results[0].fused_score == pytest.approx(1 / 62 + 1 / 61)


def test_reciprocal_rank_fusion_is_stable_and_bounded():
    results = fuse_ranked_results(
        [_lexical(2, 1.0)], [_vector(1, 1.0)], limit=1
    )

    assert [item.chunk_id.int for item in results] == [1]


def test_vector_allowlist_limit_is_optional_runtime_configuration(monkeypatch):
    monkeypatch.delenv("MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT", raising=False)
    assert _vector_allowlist_limit() is None

    monkeypatch.setenv("MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT", "200000")
    assert _vector_allowlist_limit() == 200_000

    for value in ("0", "-1", "bad"):
        monkeypatch.setenv("MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT", value)
        with pytest.raises(RuntimeError, match="positive integer"):
            _vector_allowlist_limit()


def test_reciprocal_rank_fusion_degrades_to_lexical_results():
    results = fuse_ranked_results(
        [_lexical(2, 0.9), _lexical(1, 0.8)], None
    )

    assert [item.chunk_id.int for item in results] == [2, 1]
    assert all(item.vector_score is None for item in results)
