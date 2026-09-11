from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pyarrow.parquet as parquet
import pytest


spec = spec_from_file_location(
    "build_convomem_pilot",
    Path(__file__).parents[1] / "benchmarks" / "build-convomem-pilot.py",
)
builder = module_from_spec(spec)
spec.loader.exec_module(builder)


def test_message_evidence_matches_minor_generated_variation():
    indexed = [
        ("wrong", "Assistant", "I used to host a campus radio show.", "persona"),
        ("right", "User", "I actually used to host a late-night campus radio show.", "persona"),
    ]
    evidence = [{"speaker": "User", "text": "I used to host a late-night campus radio show."}]

    assert builder._relevant_messages(indexed, evidence) == ["right"]


def test_message_evidence_matches_text_split_across_messages():
    indexed = [
        ("query-0-0", "Assistant", "Use Vietnam for affordable travel.", "persona"),
        ("query-0-1", "Assistant", "It also has a growing technology scene.", "persona"),
    ]
    evidence = [{
        "speaker": "Assistant",
        "text": "Use Vietnam for affordable travel. It also has a growing technology scene.",
    }]

    assert builder._relevant_messages(indexed, evidence) == ["query-0-0", "query-0-1"]


def test_message_evidence_rejects_ambiguous_split_matches():
    indexed = [
        ("first-0-0", "Assistant", "Use Vietnam for affordable travel.", "persona"),
        ("first-0-1", "Assistant", "It has a growing technology scene.", "persona"),
        ("second-0-0", "Assistant", "Use Vietnam for affordable travel.", "persona"),
        ("second-0-1", "Assistant", "It has a growing technology scene.", "persona"),
    ]
    evidence = [{
        "speaker": "Assistant",
        "text": "Use Vietnam for affordable travel. It has a growing technology scene.",
    }]

    with pytest.raises(ValueError, match="one unique"):
        builder._relevant_messages(indexed, evidence)


def test_message_evidence_rejects_multiple_exact_matches():
    indexed = [
        ("first", "User", "Actually, I use green for hot leads.", "persona"),
        ("second", "User", "Usually, I use green for hot leads.", "persona"),
    ]
    evidence = [{"speaker": "User", "text": "I use green for hot leads."}]

    with pytest.raises(ValueError, match="multiple"):
        builder._relevant_messages(indexed, evidence)


def test_message_evidence_rejects_ambiguous_fuzzy_match():
    indexed = [
        ("first", "User", "I actually used to host a late-night campus radio show.", "persona"),
        ("second", "User", "I usually used to host a late-night campus radio show.", "persona"),
    ]
    evidence = [{"speaker": "User", "text": "I used to host a late-night campus radio show."}]

    with pytest.raises(ValueError, match="one unique"):
        builder._relevant_messages(indexed, evidence)


def test_build_pools_personas_and_separates_current_from_history(tmp_path, monkeypatch):
    def fake_download(path):
        persona = path.rsplit("/", 1)[-1].removesuffix(".json")
        if path.startswith("filler_conversations/"):
            return {"evidence_items": [{"conversations": [{"messages": [
                {"speaker": "User", "text": f"filler for {persona}"}
            ]}]}]}
        task = next(name for name, source in builder.TASKS.items() if source in path)
        evidences = [
            {"speaker": "User", "text": f"old {task} for {persona}"},
            {"speaker": "User", "text": f"new {task} for {persona}"},
        ] if task == "changing_evidence" else [
            {"speaker": "User", "text": f"{task} for {persona}"}
        ]
        return {"evidence_items": [{
            "question": f"question {task}",
            "answer": f"answer {task}",
            "message_evidences": evidences,
            "conversations": [{"messages": evidences}],
        }]}

    monkeypatch.setattr(builder, "_download", fake_download)
    builder.build(tmp_path, 1, 1, 7, ("first.json", "second.json"))

    corpus = parquet.read_table(tmp_path / "corpus.parquet")
    current = parquet.read_table(tmp_path / "changing_evidence-qrels.parquet")
    history = parquet.read_table(tmp_path / "changing_evidence_history-qrels.parquet")
    queries = parquet.read_table(tmp_path / "user_evidence-queries.parquet")
    assert set(corpus["persona"].to_pylist()) == {"first", "second"}
    assert len(current) == 2
    assert len(history) == 4
    assert queries.column_names == ["id", "text", "answer", "persona"]
