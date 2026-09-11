from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pyarrow.parquet as parquet


spec = spec_from_file_location(
    "build_operational_workload",
    Path(__file__).parents[1] / "benchmarks" / "build-operational-workload.py",
)
builder = module_from_spec(spec)
spec.loader.exec_module(builder)


def test_build_operational_workload_separates_update_labels(tmp_path):
    source = Path(__file__).parents[1] / "benchmarks" / "operational-memory-workload.json"
    builder.build(source, tmp_path)

    corpus = parquet.read_table(tmp_path / "corpus.parquet")
    current = parquet.read_table(tmp_path / "current_update-qrels.parquet")
    history = parquet.read_table(tmp_path / "update_history-qrels.parquet")
    assert len(corpus) == 14
    assert len(current) == 2
    assert len(history) == 4
    assert set(corpus["persona"].to_pylist()) == {"alice", "bob", "carol"}
