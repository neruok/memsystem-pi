#!/usr/bin/env python3
"""Convert the committed atomic-memory workload to evaluator Parquet files."""

import argparse
import json
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet


def build(source: Path, output: Path) -> None:
    data = json.loads(source.read_text())
    if data.get("version") != 1 or not data.get("records") or not data.get("queries"):
        raise ValueError("workload must use version 1 and contain records and queries")
    records = data["records"]
    identifiers = [record["id"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("record identifiers must be unique")
    if any(set(query["relevant"]) - set(identifiers) for query in data["queries"]):
        raise ValueError("query references an unknown record")
    output.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.table({
        "id": identifiers,
        "title": [record["topic"] for record in records],
        "text": [record["text"] for record in records],
        "persona": [record["persona"] for record in records],
    }), output / "corpus.parquet")
    tasks = list(dict.fromkeys(query["task"] for query in data["queries"]))
    for task in tasks:
        queries = [query for query in data["queries"] if query["task"] == task]
        parquet.write_table(arrow.table({
            "id": [query["id"] for query in queries],
            "text": [query["text"] for query in queries],
            "answer": [query["answer"] for query in queries],
            "persona": [query["persona"] for query in queries],
        }), output / f"{task}-queries.parquet")
        parquet.write_table(arrow.table({
            "query-id": [query["id"] for query in queries for _ in query["relevant"]],
            "corpus-id": [identifier for query in queries for identifier in query["relevant"]],
            "score": [1 for query in queries for _ in query["relevant"]],
        }), output / f"{task}-qrels.parquet")
    (output / "NAME").write_text("memsystem atomic operational memory workload\n")
    (output / "REVISION").write_text("1\n")
    print(json.dumps({
        "documents": len(records),
        "queries": len(data["queries"]),
        "tasks": tasks,
        "output": str(output),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("service/benchmarks/operational-memory-workload.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/operational-memory-workload"))
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
