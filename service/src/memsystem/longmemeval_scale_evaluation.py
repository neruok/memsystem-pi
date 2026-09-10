"""Benchmark authorized vector retrieval with the MTEB LongMemEval corpus."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
from uuid import uuid4

import numpy as np
from pgvector.psycopg import register_vector
import psycopg
import pyarrow.parquet as parquet

from memsystem.chunking import CHUNK_PROFILE
from memsystem.context_token import ContextTokenClaims
from memsystem.database import tenant_transaction
from memsystem.embeddings import ACTIVE_PROFILE
from memsystem.evaluation import _machine, make_corpus
from memsystem.retrieval import search_vector
from memsystem.vector_index import IndexSlot, TurboVecIndex, rebuild_index

DEFAULT_DATASET = Path("data/longmemeval")
TASKS = (
    "knowledge_update",
    "multi_session",
    "single_session_assistant",
    "single_session_preference",
    "single_session_user",
    "temporal_reasoning",
)


def load_corpus(path: Path, limit: int) -> list[tuple[str, str]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    documents = []
    for batch in parquet.ParquetFile(path).iter_batches(
        batch_size=8192, columns=("title", "text")
    ):
        titles, texts = batch.column(0).to_pylist(), batch.column(1).to_pylist()
        documents.extend(
            ((title or "LongMemEval")[:500], (text or "").replace("\x00", ""))
            for title, text in zip(titles, texts, strict=True)
            if text
        )
        if len(documents) >= limit:
            return documents[:limit]
    raise ValueError(f"LongMemEval corpus contains only {len(documents)} documents")


def _dataset_counts(directory: Path) -> dict[str, dict[str, int]]:
    return {
        task: {
            "queries": parquet.read_metadata(
                directory / f"{task}-queries.parquet"
            ).num_rows,
            "qrels": parquet.read_metadata(
                directory / f"{task}-qrels.parquet"
            ).num_rows,
        }
        for task in TASKS
    }


def _setup_identity(connection, claims, compartment_id):
    with tenant_transaction(connection, claims.tenant_id):
        connection.execute(
            "INSERT INTO tenants (id, name) VALUES (%s, 'LongMemEval benchmark')",
            (claims.tenant_id,),
        )
        connection.execute(
            "INSERT INTO tenant_memberships (tenant_id, user_id) VALUES (%s, %s)",
            (claims.tenant_id, claims.user_id),
        )
        root_id = uuid4()
        connection.execute(
            """INSERT INTO compartments (tenant_id, id, path, scope_type, name)
               VALUES (%s, %s, 'longmemeval', 'tenant', 'LongMemEval')""",
            (claims.tenant_id, root_id),
        )
        connection.execute(
            """INSERT INTO compartments
               (tenant_id, id, parent_id, path, scope_type, scope_id, name)
               VALUES (%s, %s, %s, 'longmemeval.user', 'user', %s, 'User')""",
            (claims.tenant_id, compartment_id, root_id, claims.user_id),
        )


def _append_documents(connection, claims, compartment_id, documents, vectors, offset):
    chunk_ids = []
    for batch_start in range(0, len(documents), 5_000):
        batch_documents = documents[batch_start : batch_start + 5_000]
        batch_vectors = vectors[batch_start : batch_start + 5_000]
        document_ids = [uuid4() for _ in batch_documents]
        revision_ids = [uuid4() for _ in batch_documents]
        batch_chunk_ids = [uuid4() for _ in batch_documents]
        with tenant_transaction(connection, claims.tenant_id):
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO documents
                       (tenant_id, id, compartment_id, kind, slug, current_revision_id)
                       VALUES (%s, %s, %s, 'page', %s, %s)""",
                    (
                        (
                            claims.tenant_id, document_id, compartment_id,
                            f"longmemeval-{offset + batch_start + index}", revision_id,
                        )
                        for index, (document_id, revision_id) in enumerate(
                            zip(document_ids, revision_ids, strict=True)
                        )
                    ),
                )
                cursor.executemany(
                    """INSERT INTO document_revisions
                       (tenant_id, document_id, id, revision, title, markdown,
                        created_by_type, created_by_id)
                       VALUES (%s, %s, %s, 1, %s, %s, 'system', NULL)""",
                    (
                        (claims.tenant_id, document_id, revision_id, title, text)
                        for document_id, revision_id, (title, text) in zip(
                            document_ids, revision_ids, batch_documents, strict=True
                        )
                    ),
                )
                cursor.executemany(
                    """INSERT INTO chunks
                       (tenant_id, document_id, revision_id, id, heading_path, text,
                        position, source_start, source_end, chunk_profile, search_document,
                        embedding, embedding_model, embedding_version, embedding_dimension,
                        embedding_state)
                       VALUES (%s, %s, %s, %s, '{}', %s, 0, 0, %s, %s,
                               setweight(to_tsvector(%s::regconfig, %s), 'A') ||
                               setweight(to_tsvector(%s::regconfig, %s), 'C'),
                               %s, %s, %s, %s, 'ready')""",
                    (
                        (
                            claims.tenant_id, document_id, revision_id, chunk_id, text,
                            len(text), CHUNK_PROFILE, ACTIVE_PROFILE.text_search_config,
                            title, ACTIVE_PROFILE.text_search_config, text, vector,
                            ACTIVE_PROFILE.embedding_model, ACTIVE_PROFILE.embedding_version,
                            ACTIVE_PROFILE.embedding_dimension,
                        )
                        for document_id, revision_id, chunk_id, (title, text), vector in zip(
                            document_ids, revision_ids, batch_chunk_ids,
                            batch_documents, batch_vectors, strict=True
                        )
                    ),
                )
        chunk_ids.extend(batch_chunk_ids)
    return chunk_ids


def _measure_queries(database_url, claims, slot, vectors, query_vectors, chunk_ids, k):
    latencies, recalls = [], []
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            search_vector(connection, claims, slot, query_vectors[0], limit=k)
        for query in query_vectors:
            expected = {
                chunk_ids[int(index)]
                for index in np.argsort(-(vectors @ query), kind="stable")[:k]
            }
            started = time.perf_counter()
            with tenant_transaction(connection, claims.tenant_id):
                results = search_vector(connection, claims, slot, query, limit=k)
            latencies.append(time.perf_counter() - started)
            recalls.append(len(expected & {item.chunk_id for item in results}) / k)
    return {
        "queries": len(latencies),
        "latency_ms_p50": round(float(np.percentile(latencies, 50)) * 1000, 3),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)) * 1000, 3),
        "throughput_qps": round(len(latencies) / sum(latencies), 3),
        "recall_at_k_mean": round(float(np.mean(recalls)), 6),
        "recall_at_k_min": round(min(recalls), 6),
        "latency_ms_samples": [round(value * 1000, 3) for value in latencies],
    }


def benchmark(
    database_url: str,
    admin_database_url: str,
    dataset: Path,
    *,
    sizes: tuple[int, ...] = (50_000, 100_000, 200_000),
    queries: int = 20,
    k: int = 10,
    seed: int = 7,
) -> dict:
    if not sizes or tuple(sorted(set(sizes))) != sizes or sizes[0] < 200:
        raise ValueError("sizes must be unique increasing integers of at least 200")
    if queries < 1 or not 1 <= k <= 100:
        raise ValueError("queries must be positive and k must be from 1 through 100")
    corpus_path = dataset / "corpus.parquet"
    documents = load_corpus(corpus_path, sizes[-1])
    vectors, query_vectors, _ = make_corpus(
        sizes[-1], queries, ACTIVE_PROFILE.embedding_dimension, 100, seed
    )
    claims = ContextTokenClaims(uuid4(), uuid4(), None, None, None, None, 0, 2**31)
    compartment_id = uuid4()
    chunk_ids = []
    slot = None
    stages = []
    try:
        with (
            psycopg.connect(database_url, autocommit=True) as connection,
            tempfile.TemporaryDirectory(prefix="memsystem-longmemeval-") as directory,
        ):
            postgres_version = connection.execute("SHOW server_version").fetchone()[0]
            pgvector_version = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0]
            _setup_identity(connection, claims, compartment_id)
            slot = IndexSlot(TurboVecIndex(
                claims.tenant_id, Path(directory) / "initial.tvim"
            ))
            previous = 0
            for size in sizes:
                load_started = time.perf_counter()
                chunk_ids.extend(_append_documents(
                    connection, claims, compartment_id,
                    documents[previous:size], vectors[previous:size], previous,
                ))
                load_seconds = time.perf_counter() - load_started
                rebuild_started = time.perf_counter()
                owner = rebuild_index(
                    connection, claims.tenant_id, slot, directory
                )
                rebuild_seconds = time.perf_counter() - rebuild_started
                search = _measure_queries(
                    database_url, claims, slot, vectors[:size], query_vectors,
                    chunk_ids, k,
                )
                stages.append({
                    "vectors": size,
                    "incremental_load_seconds": round(load_seconds, 3),
                    "index_rebuild_seconds": round(rebuild_seconds, 3),
                    "index_bytes": owner.path.stat().st_size,
                    **search,
                })
                previous = size
    finally:
        if slot is not None:
            slot.current(claims.tenant_id).close()
        with psycopg.connect(admin_database_url, autocommit=True) as cleanup:
            cleanup.execute("DELETE FROM documents WHERE tenant_id = %s", (claims.tenant_id,))
            cleanup.execute("DELETE FROM tenants WHERE id = %s", (claims.tenant_id,))

    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **_machine(),
            "postgres": postgres_version,
            "pgvector": pgvector_version,
            "turbovec": version("turbovec"),
            "pyarrow": version("pyarrow"),
        },
        "dataset": {
            "name": "mteb/LongMemEval",
            "revision": (dataset / "REVISION").read_text().strip(),
            "corpus_rows": parquet.read_metadata(corpus_path).num_rows,
            "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "tasks": _dataset_counts(dataset),
        },
        "profile": {
            "name": ACTIVE_PROFILE.name,
            "dimension": ACTIVE_PROFILE.embedding_dimension,
            "turbovec_bits": ACTIVE_PROFILE.turbovec_bits,
            "allowlist_limit": os.getenv("MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT") or None,
            "benchmark_vectors": "deterministic clustered vectors; semantic labels are not used in this scale test",
            "k": k,
            "seed": seed,
        },
        "stages": stages,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "MEMSYSTEM_TEST_DATABASE_URL",
            "postgresql://memsystem_service:memsystem@127.0.0.1:5432/memsystem",
        ),
    )
    parser.add_argument(
        "--admin-database-url",
        default=os.getenv(
            "MEMSYSTEM_TEST_ADMIN_DATABASE_URL",
            "postgresql://postgres:postgres@127.0.0.1:5432/memsystem",
        ),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[50_000, 100_000, 200_000])
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.database_url,
        args.admin_database_url,
        args.dataset,
        sizes=tuple(args.sizes),
        queries=args.queries,
        k=args.k,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
