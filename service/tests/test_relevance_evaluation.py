import json
from pathlib import Path

import pytest

from memsystem.relevance_evaluation import (
    Judgment,
    _first_relevant_rank,
    _qwen_revision,
    _rank_metrics,
    load_corpus,
)


def test_labeled_corpus_resolves_project_chunks(tmp_path: Path):
    (tmp_path / "doc.md").write_text(
        "# Title\n\n## First\n" + "alpha " * 513 + "\n\n## Second\nbeta\n"
    )
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "documents": ["doc.md"],
        "queries": [{
            "query": "where is beta?",
            "relevant": [{"path": "doc.md", "heading": "Second", "contains": "beta"}],
        }],
    }))

    chunks, judgments = load_corpus(tmp_path, manifest)

    assert chunks[next(iter(judgments[0].relevant))].heading_path[-1] == "Second"
    assert _rank_metrics([[0, 1, 2]], [Judgment("q", frozenset({1}))], 2) == {
        "hit_rate_at_2": 1.0,
        "mean_recall_at_2": 1.0,
        "mean_reciprocal_rank_at_2": 0.5,
    }
    assert _first_relevant_rank([0, 1, 2], frozenset({2}), 2) is None


def test_corpus_manifest_can_extend_one_in_same_directory(tmp_path: Path):
    (tmp_path / "doc.md").write_text("# Title\n" + "text " * 513)
    (tmp_path / "base.json").write_text(json.dumps({
        "version": 1,
        "documents": ["doc.md"],
        "queries": [{"query": "q", "relevant": [{"path": "doc.md", "heading": "Title"}]}],
    }))
    child = tmp_path / "child.json"
    child.write_text(json.dumps({
        "version": 1,
        "base": "base.json",
        "documents": [],
        "queries": [],
    }))

    chunks, judgments = load_corpus(tmp_path, child)

    assert len(chunks) == 2
    assert len(judgments) == 1


def test_qwen_revision_must_be_a_commit():
    with pytest.raises(ValueError, match="full lowercase commit"):
        _qwen_revision("Qwen/custom", "main")
    assert _qwen_revision("Qwen/custom", "a" * 40) == "a" * 40


def test_labeled_corpus_rejects_unmatched_selector(tmp_path: Path):
    (tmp_path / "doc.md").write_text("# Title\n" + "text " * 513)
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "documents": ["doc.md"],
        "queries": [{"query": "q", "relevant": [{"path": "doc.md", "heading": "missing"}]}],
    }))

    with pytest.raises(ValueError, match="matched no chunks"):
        load_corpus(tmp_path, manifest)
