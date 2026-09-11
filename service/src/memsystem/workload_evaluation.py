"""Measure message-memory retrieval on a labeled Parquet workload."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np
from numpy.lib.format import open_memmap
import pyarrow.parquet as parquet

from memsystem.evaluation import _peak_rss_bytes
from memsystem.relevance_evaluation import (
    DEFAULT_MODEL,
    DEFAULT_QWEN_TASK,
    Judgment,
    _first_relevant_rank,
    _qwen_revision,
    _rank_metrics,
)

DEFAULT_DATASET = Path("data/convomem-pilot")
TASKS = (
    "user_evidence",
    "assistant_facts",
    "changing_evidence",
    "changing_evidence_history",
    "preference_evidence",
    "implicit_connection",
)


def load_dataset(
    directory: Path, tasks: tuple[str, ...] = TASKS
) -> tuple[list[str], list[Judgment], list[str], list[str]]:
    ids = parquet.read_table(directory / "corpus.parquet", columns=["id"])["id"].to_pylist()
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError("corpus identifiers must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("corpus identifiers must be unique")
    positions = {identifier: index for index, identifier in enumerate(ids)}
    judgments, query_ids, task_names = [], [], []
    for task in tasks:
        queries = parquet.read_table(directory / f"{task}-queries.parquet").to_pylist()
        qrels = parquet.read_table(directory / f"{task}-qrels.parquet").to_pylist()
        if not queries:
            raise ValueError(f"task has no queries: {task}")
        if any(
            not isinstance(row["id"], str) or not row["id"]
            or not isinstance(row["text"], str) or not row["text"].strip()
            for row in queries
        ):
            raise ValueError(f"task has an invalid query: {task}")
        task_query_ids = {row["id"] for row in queries}
        if len(task_query_ids) != len(queries):
            raise ValueError(f"task has duplicate query identifiers: {task}")
        relevant: dict[str, set[int]] = {}
        for row in qrels:
            if (
                not isinstance(row["query-id"], str) or not row["query-id"]
                or not isinstance(row["corpus-id"], str) or not row["corpus-id"]
                or not isinstance(row["score"], (int, float))
            ):
                raise ValueError(f"task has an invalid qrel: {task}")
            if row["query-id"] not in task_query_ids:
                raise ValueError(f"qrels contain unknown query identifier: {row['query-id']}")
            if row["corpus-id"] not in positions:
                raise ValueError(f"unknown corpus identifier in qrels: {row['corpus-id']}")
            if row["score"] > 0:
                relevant.setdefault(row["query-id"], set()).add(positions[row["corpus-id"]])
        for row in queries:
            matched = relevant.get(row["id"])
            if not matched:
                raise ValueError(f"query has no positive relevance labels: {row['id']}")
            query_ids.append(row["id"])
            task_names.append(task)
            judgments.append(Judgment(row["text"], frozenset(matched)))
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query identifiers must be unique")
    return ids, judgments, query_ids, task_names


def _corpus_batches(path: Path, batch_size: int):
    for batch in parquet.ParquetFile(path).iter_batches(
        batch_size=batch_size, columns=("title", "text")
    ):
        yield [
            f"{title}\n{text}" if title else (text or "")
            for title, text in zip(
                batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
            )
        ]


def _top_rankings(
    documents: np.ndarray, queries: np.ndarray, k: int, batch_size: int = 8
) -> list[list[int]]:
    rankings = []
    for start in range(0, len(queries), batch_size):
        scores = documents @ queries[start : start + batch_size].T
        for column in range(scores.shape[1]):
            values = scores[:, column]
            top = np.argpartition(values, -k)[-k:]
            rankings.append([
                int(index) for index in top[np.lexsort((top, -values[top]))]
            ])
    return rankings


def _metrics(rankings: list[list[int]], judgments: list[Judgment], k: int) -> dict[str, float]:
    metrics = _rank_metrics(rankings, judgments, k)
    ndcg = []
    for ranking, judgment in zip(rankings, judgments, strict=True):
        dcg = sum(
            1 / np.log2(rank + 1)
            for rank, item in enumerate(ranking[:k], 1)
            if item in judgment.relevant
        )
        ideal = sum(
            1 / np.log2(rank + 1)
            for rank in range(1, min(k, len(judgment.relevant)) + 1)
        )
        ndcg.append(dcg / ideal)
    metrics[f"mean_ndcg_at_{k}"] = round(float(np.mean(ndcg)), 6)
    return metrics


def _optional_context(directory: Path, tasks: tuple[str, ...]):
    corpus = parquet.ParquetFile(directory / "corpus.parquet")
    corpus_columns = set(corpus.schema_arrow.names)
    table = parquet.read_table(
        directory / "corpus.parquet",
        columns=[name for name in ("text", "persona") if name in corpus_columns],
    )
    texts = table["text"].to_pylist() if "text" in table.column_names else []
    personas = table["persona"].to_pylist() if "persona" in table.column_names else None
    answers, query_personas = [], []
    has_answers = has_personas = True
    for task in tasks:
        path = directory / f"{task}-queries.parquet"
        columns = set(parquet.ParquetFile(path).schema_arrow.names)
        selected = [name for name in ("answer", "persona") if name in columns]
        query_table = parquet.read_table(path, columns=selected)
        rows = len(parquet.read_table(path, columns=["id"]))
        if "answer" in query_table.column_names:
            answers.extend(query_table["answer"].to_pylist())
        else:
            has_answers = False
            answers.extend([None] * rows)
        if "persona" in query_table.column_names:
            query_personas.extend(query_table["persona"].to_pylist())
        else:
            has_personas = False
            query_personas.extend([None] * rows)
    return texts, personas if has_personas else None, answers if has_answers else None, query_personas if has_personas else None


def _normalize_answer(text: str) -> str:
    return " ".join("".join(character if character.isalnum() else " " for character in text.casefold()).split())


def _prediction_metrics(
    path: Path,
    query_ids: list[str],
    answers: list[str] | None,
    judgments: list[Judgment],
    corpus_ids: list[str],
) -> dict[str, float]:
    if answers is None:
        raise ValueError("predictions require reference answers")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("predictions must be a JSON array")
    predictions = {row.get("query_id"): row for row in rows if isinstance(row, dict)}
    if set(predictions) != set(query_ids) or len(predictions) != len(rows):
        raise ValueError("predictions must contain each query identifier once")
    answer_matches, citation_matches = [], []
    for query_id, reference, judgment in zip(query_ids, answers, judgments, strict=True):
        prediction = predictions[query_id]
        answer = prediction.get("answer")
        citations = prediction.get("citations")
        if (
            not isinstance(answer, str)
            or not isinstance(citations, list)
            or any(not isinstance(citation, str) for citation in citations)
        ):
            raise ValueError("each prediction needs an answer and citation list")
        normalized_reference = _normalize_answer(reference)
        answer_matches.append(
            bool(normalized_reference)
            and normalized_reference in _normalize_answer(answer)
        )
        relevant_ids = {corpus_ids[index] for index in judgment.relevant}
        citation_matches.append(set(citations) == relevant_ids)
    grounded = [
        answer_match and citation_match
        for answer_match, citation_match in zip(answer_matches, citation_matches, strict=True)
    ]
    return {
        "normalized_reference_match": round(float(np.mean(answer_matches)), 6),
        "exact_citation_match": round(float(np.mean(citation_matches)), 6),
        "answer_and_citation_match": round(float(np.mean(grounded)), 6),
    }


def _support_metrics(
    rankings: list[list[int]],
    k: int,
    texts: list[str],
    answers: list[str] | None,
    document_personas: list[str] | None,
    query_personas: list[str] | None,
) -> dict[str, float]:
    metrics = {}
    if answers is not None:
        covered = [
            (set(_normalize_answer(answer).split()) - {"a", "an", "and", "or", "the"})
            <= set(_normalize_answer(" ".join(texts[index] for index in ranking[:k])).split())
            for ranking, answer in zip(rankings, answers, strict=True)
        ]
        metrics[f"reference_answer_token_coverage_at_{k}"] = round(float(np.mean(covered)), 6)
    if document_personas is not None and query_personas is not None:
        foreign = [
            sum(document_personas[index] != persona for index in ranking[:k]) / k
            for ranking, persona in zip(rankings, query_personas, strict=True)
        ]
        metrics[f"mean_foreign_persona_fraction_at_{k}"] = round(float(np.mean(foreign)), 6)
    return metrics


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _labels_sha256(directory: Path, tasks: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        for kind in ("queries", "qrels"):
            path = directory / f"{task}-{kind}.parquet"
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_fastembed(model_name: str, cache_dir: Path):
    from fastembed import TextEmbedding

    provider = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=1)
    model_dir = getattr(provider.model, "_model_dir", None)
    metadata = {
        "backend": "fastembed",
        "fastembed": version("fastembed"),
        "model_artifact_revision": model_dir.name if model_dir else None,
    }
    return provider, metadata


def _load_qwen(model_name: str, cache_dir: Path, device: str, revision: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, padding_side="left", revision=revision
    )
    model = AutoModel.from_pretrained(
        model_name, cache_dir=cache_dir, dtype=dtype, revision=revision
    )
    model.to(device)
    model.eval()
    metadata = {
        "backend": "qwen-transformers",
        "torch": version("torch"),
        "transformers": version("transformers"),
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "model_revision": revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None) or revision,
    }
    if device.startswith("cuda"):
        metadata["gpu"] = torch.cuda.get_device_name(device)
        metadata["cuda_runtime"] = torch.version.cuda
        driver = Path("/proc/driver/nvidia/version")
        metadata["nvidia_driver"] = driver.read_text().splitlines()[0] if driver.exists() else None
    return (tokenizer, model, device), metadata


def _run_qwen_batch(provider, texts, dimensions, max_length):
    import torch
    import torch.nn.functional as functional

    tokenizer, model, device = provider
    with torch.inference_mode():
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        pooled = model(**inputs).last_hidden_state[:, -1]
        if dimensions is not None:
            if not 32 <= dimensions <= pooled.shape[1]:
                raise ValueError("Qwen dimensions must be within the model output")
            pooled = pooled[:, :dimensions]
        return functional.normalize(pooled, p=2, dim=1).float().cpu().numpy()


def _try_qwen_batch(provider, texts, dimensions, max_length):
    import torch

    try:
        return _run_qwen_batch(provider, texts, dimensions, max_length)
    except torch.OutOfMemoryError:
        return None


def _embed_qwen_batch(provider, texts, dimensions, max_length):
    import torch

    vectors = _try_qwen_batch(provider, texts, dimensions, max_length)
    if vectors is not None:
        return vectors
    if len(texts) == 1:
        raise RuntimeError("Qwen document exceeds available CUDA memory")
    torch.cuda.empty_cache()
    middle = len(texts) // 2
    return np.concatenate((
        _embed_qwen_batch(provider, texts[:middle], dimensions, max_length),
        _embed_qwen_batch(provider, texts[middle:], dimensions, max_length),
    ))


def _embed_batch(provider, texts: list[str], backend: str, dimensions: int | None, max_length: int) -> np.ndarray:
    vectors = (
        list(provider.embed(texts))
        if backend == "fastembed"
        else _embed_qwen_batch(provider, texts, dimensions, max_length)
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _embed_queries(provider, judgments, backend, dimensions, max_length, task, batch_size):
    texts = [item.query for item in judgments]
    if backend == "fastembed":
        return np.ascontiguousarray(list(provider.query_embed(texts)), dtype=np.float32)
    vectors = [
        _embed_batch(
            provider,
            [f"Instruct: {task}\nQuery:{text}" for text in texts[start : start + batch_size]],
            backend,
            dimensions,
            max_length,
        )
        for start in range(0, len(texts), batch_size)
    ]
    return np.concatenate(vectors)


def _cache_paths(vector_cache: Path, identity: dict) -> tuple[Path, Path]:
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return vector_cache / f"{key}.npy", vector_cache / f"{key}.json"


def _load_cached_vectors(path: Path, metadata_path: Path, identity: dict, rows: int):
    try:
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("identity") != identity
            or metadata.get("sha256") != _file_sha256(path)
        ):
            return None
        vectors = np.load(path, mmap_mode="r")
        if (
            vectors.ndim != 2
            or vectors.shape[0] != rows
            or vectors.dtype != np.float32
            or (identity["dimensions"] is not None and vectors.shape[1] != identity["dimensions"])
            or not np.all(np.isfinite(vectors))
        ):
            return None
        return vectors
    except (AttributeError, FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None


def _write_document_vectors(
    path, metadata_path, identity, corpus_path, rows, provider, backend,
    dimensions, max_length, batch_size, token_budget,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    vectors = None
    position = 0
    try:
        for texts in _corpus_batches(corpus_path, max(1024, batch_size)):
            if backend == "qwen":
                lengths = provider[0](
                    texts, truncation=True, max_length=max_length,
                    return_length=True,
                )["length"]
            else:
                lengths = [len(text) for text in texts]
            ordered = sorted(range(len(texts)), key=lengths.__getitem__)
            start = 0
            while start < len(ordered):
                size = min(batch_size, len(ordered) - start)
                while size > 1 and lengths[ordered[start + size - 1]] * size > token_budget:
                    size -= 1
                indexes = ordered[start : start + size]
                batch = _embed_batch(
                    provider, [texts[index] for index in indexes],
                    backend, dimensions, max_length,
                )
                if vectors is None:
                    vectors = open_memmap(
                        temporary, mode="w+", dtype=np.float32,
                        shape=(rows, batch.shape[1]),
                    )
                if batch.ndim != 2 or batch.shape[1] != vectors.shape[1]:
                    raise ValueError("document embedding dimensions changed between batches")
                vectors[position + np.asarray(indexes)] = batch
                start += size
            position += len(texts)
        if vectors is None or position != rows:
            raise ValueError("document vector count does not match corpus")
        for start in range(0, rows, batch_size):
            batch = vectors[start : start + batch_size]
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            if np.any(norms == 0) or not np.all(np.isfinite(norms)):
                raise ValueError("document embedding vectors must be finite and nonzero")
            batch /= norms
        vectors.flush()
        temporary_metadata.write_text(json.dumps({
            "identity": identity,
            "sha256": _file_sha256(temporary),
        }, sort_keys=True) + "\n")
        os.replace(temporary, path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return np.load(path, mmap_mode="r")


def evaluate(
    dataset: Path,
    model_name: str,
    cache_dir: Path,
    *,
    backend: str = "fastembed",
    dimensions: int | None = None,
    device: str = "auto",
    batch_size: int = 8,
    max_length: int = 8192,
    task: str = DEFAULT_QWEN_TASK,
    revision: str | None = None,
    k: int = 10,
    vector_cache: Path | None = None,
    token_budget: int = 8192,
    tasks: tuple[str, ...] = TASKS,
    predictions: Path | None = None,
) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    if batch_size < 1 or max_length < 1 or token_budget < 1:
        raise ValueError("batch size, maximum length, and token budget must be positive")
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be nonempty and unique")
    corpus_ids, judgments, query_ids, task_names = load_dataset(dataset, tasks)
    texts, document_personas, answers, query_personas = _optional_context(dataset, tasks)
    if k > len(corpus_ids):
        raise ValueError("k cannot exceed corpus size")
    corpus_path = dataset / "corpus.parquet"
    corpus_sha256 = _file_sha256(corpus_path)
    if backend == "qwen":
        revision = _qwen_revision(model_name, revision)
        provider, backend_metadata = _load_qwen(model_name, cache_dir, device, revision)
    elif backend == "fastembed":
        provider, backend_metadata = _load_fastembed(model_name, cache_dir)
    else:
        raise ValueError(f"unsupported embedding backend: {backend}")
    identity = {
        "version": 1,
        "corpus_sha256": corpus_sha256,
        "rows": len(corpus_ids),
        "backend": backend,
        "model": model_name,
        "model_artifact_revision": backend_metadata.get("model_artifact_revision"),
        "model_revision": backend_metadata.get("resolved_model_revision"),
        "fastembed": backend_metadata.get("fastembed"),
        "torch": backend_metadata.get("torch"),
        "transformers": backend_metadata.get("transformers"),
        "device": backend_metadata.get("device"),
        "dtype": backend_metadata.get("dtype"),
        "dimensions": dimensions,
        "max_length": max_length,
        "title_separator": "newline",
    }
    vector_cache = vector_cache or cache_dir / "longmemeval"
    cache_path, metadata_path = _cache_paths(vector_cache, identity)

    started = time.perf_counter()
    documents = _load_cached_vectors(cache_path, metadata_path, identity, len(corpus_ids))
    cache_hit = documents is not None
    if documents is None:
        documents = _write_document_vectors(
            cache_path, metadata_path, identity, corpus_path, len(corpus_ids), provider,
            backend, dimensions, max_length, batch_size, token_budget,
        )
    queries = _embed_queries(
        provider, judgments, backend, dimensions, max_length, task, batch_size
    )
    if documents.shape[1] != queries.shape[1]:
        raise ValueError("document and query vectors must have matching dimensions")
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if np.any(query_norms == 0) or not np.all(np.isfinite(query_norms)):
        raise ValueError("query embedding vectors must be finite and nonzero")
    queries /= query_norms
    embedding_seconds = time.perf_counter() - started

    rankings = _top_rankings(documents, queries, k, batch_size)
    task_metrics = {}
    for task_name in tasks:
        selected = [index for index, name in enumerate(task_names) if name == task_name]
        selected_rankings = [rankings[index] for index in selected]
        task_metrics[task_name] = {
            **_metrics(
                selected_rankings,
                [judgments[index] for index in selected],
                k,
            ),
            **_support_metrics(
                selected_rankings,
                k,
                texts,
                [answers[index] for index in selected] if answers is not None else None,
                document_personas,
                [query_personas[index] for index in selected] if query_personas is not None else None,
            ),
        }
    overall_metrics = {
        **_metrics(rankings, judgments, k),
        **_support_metrics(
            rankings, k, texts, answers, document_personas, query_personas
        ),
    }
    candidate_cutoffs = {}
    for cutoff in sorted({10, 30, k}):
        if cutoff > k:
            continue
        candidate_cutoffs[str(cutoff)] = {
            "overall": _metrics(rankings, judgments, cutoff),
            "tasks": {
                task_name: _metrics(
                    [ranking for ranking, name in zip(rankings, task_names, strict=True) if name == task_name],
                    [judgment for judgment, name in zip(judgments, task_names, strict=True) if name == task_name],
                    cutoff,
                )
                for task_name in tasks
            },
        }
    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pyarrow": version("pyarrow"),
            **backend_metadata,
            "evaluator_sha256": _file_sha256(Path(__file__)),
        },
        "dataset": {
            "name": (
                (dataset / "NAME").read_text().strip()
                if (dataset / "NAME").is_file()
                else "mteb/LongMemEval"
            ),
            "revision": (dataset / "REVISION").read_text().strip(),
            "corpus_rows": len(corpus_ids),
            "queries": len(judgments),
            "qrels": sum(len(item.relevant) for item in judgments),
            "corpus_sha256": corpus_sha256,
            "labels_sha256": _labels_sha256(dataset, tasks),
            "tasks": list(tasks),
        },
        "embedding": {
            "model": model_name,
            "dimension": int(documents.shape[1]),
            "seconds": round(embedding_seconds, 3),
            "document_cache": str(cache_path),
            "document_cache_hit": cache_hit,
            "batch_size": batch_size,
            "batch_token_budget": token_budget,
            "max_length": max_length,
            "query_instruction": task if backend == "qwen" else None,
        },
        "metrics": {
            "overall": overall_metrics,
            "candidate_cutoffs": candidate_cutoffs,
            "overall_weighting": "one equal weight per query",
            "tasks": task_metrics,
            **({
                "predictions": _prediction_metrics(
                    predictions, query_ids, answers, judgments, corpus_ids
                )
            } if predictions else {}),
        },
        "queries": [
            {
                "id": query_id,
                "task": task_name,
                "query": judgment.query,
                "relevant": [corpus_ids[index] for index in sorted(judgment.relevant)],
                f"first_relevant_rank_at_{k}": _first_relevant_rank(ranking, judgment.relevant, k),
                "top": [corpus_ids[index] for index in ranking],
            }
            for query_id, task_name, judgment, ranking in zip(
                query_ids, task_names, judgments, rankings, strict=True
            )
        ],
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--backend", choices=("fastembed", "qwen"), default="fastembed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/memsystem/fastembed")
    parser.add_argument("--vector-cache", type=Path, default=Path.home() / ".cache/memsystem/workload")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-token-budget", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--task", default=DEFAULT_QWEN_TASK)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--revision")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.dataset, args.model, args.cache_dir,
        backend=args.backend,
        dimensions=args.dimensions,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        token_budget=args.batch_token_budget,
        task=args.task,
        tasks=tuple(args.tasks),
        revision=args.revision,
        k=args.k,
        vector_cache=args.vector_cache,
        predictions=args.predictions,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    if args.summary_output:
        args.summary_output.write_text(json.dumps(
            {key: value for key, value in report.items() if key != "queries"},
            indent=2,
        ) + "\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
