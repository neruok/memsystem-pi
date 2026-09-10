"""Benchmark authorized reranked retrieval with Wikipedia-sized stored content."""

import argparse
import bz2
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
import xml.etree.ElementTree as ET
from uuid import UUID, uuid4

import numpy as np
from pgvector.psycopg import register_vector
import psycopg

from memsystem.chunking import CHUNK_PROFILE, split_markdown
from memsystem.context_token import ContextTokenClaims
from memsystem.database import tenant_transaction
from memsystem.embeddings import ACTIVE_PROFILE
from memsystem.evaluation import _machine, make_corpus
from memsystem.retrieval import search_vector
from memsystem.vector_index import IndexSlot, TurboVecIndex, rebuild_index


DEFAULT_DUMP = Path("data/simplewiki-latest-pages-articles-multistream.xml.bz2")


def load_wikipedia_chunks(path: Path, limit: int) -> tuple[list[tuple[str, str]], int]:
    """Stream non-redirect article chunks without expanding the dump on disk."""
    if limit < 1:
        raise ValueError("limit must be positive")
    chunks: list[tuple[str, str]] = []
    articles = 0
    with bz2.open(path, "rb") as source:
        for _, page in ET.iterparse(source, events=("end",)):
            if not page.tag.endswith("}page"):
                continue
            title = page.findtext("{*}title") or ""
            text = page.findtext("./{*}revision/{*}text") or ""
            if page.findtext("{*}ns") == "0" and page.find("{*}redirect") is None and text:
                articles += 1
                chunks.extend((title, chunk.text) for chunk in split_markdown(text))
            page.clear()
            if len(chunks) >= limit:
                return chunks[:limit], articles
    raise ValueError(f"dump contains only {len(chunks)} eligible chunks")


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _setup_corpus(connection, claims, compartment_id, chunks, vectors):
    document_ids = [uuid4() for _ in chunks]
    revision_ids = [uuid4() for _ in chunks]
    chunk_ids = [uuid4() for _ in chunks]
    with tenant_transaction(connection, claims.tenant_id):
        connection.execute(
            "INSERT INTO tenants (id, name) VALUES (%s, 'Wikipedia benchmark')",
            (claims.tenant_id,),
        )
        connection.execute(
            "INSERT INTO tenant_memberships (tenant_id, user_id) VALUES (%s, %s)",
            (claims.tenant_id, claims.user_id),
        )
        root_id = uuid4()
        connection.execute(
            """INSERT INTO compartments (tenant_id, id, path, scope_type, name)
               VALUES (%s, %s, 'benchmark', 'tenant', 'Benchmark')""",
            (claims.tenant_id, root_id),
        )
        connection.execute(
            """INSERT INTO compartments
               (tenant_id, id, parent_id, path, scope_type, scope_id, name)
               VALUES (%s, %s, %s, 'benchmark.user', 'user', %s, 'User')""",
            (claims.tenant_id, compartment_id, root_id, claims.user_id),
        )
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO documents
                   (tenant_id, id, compartment_id, kind, slug, current_revision_id)
                   VALUES (%s, %s, %s, 'page', %s, %s)""",
                (
                    (claims.tenant_id, document_id, compartment_id, f"wiki-{index}", revision_id)
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
                    (claims.tenant_id, document_id, revision_id, title[:500], text)
                    for document_id, revision_id, (title, text) in zip(
                        document_ids, revision_ids, chunks, strict=True
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
                        document_ids, revision_ids, chunk_ids, chunks, vectors, strict=True
                    )
                ),
            )
    return chunk_ids


def _latency_metrics(latencies: list[float]) -> dict[str, float]:
    return {
        "latency_ms_p50": round(float(np.percentile(latencies, 50)) * 1000, 3),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)) * 1000, 3),
        "throughput_qps": round(len(latencies) / sum(latencies), 3),
    }


def benchmark(
    database_url: str,
    admin_database_url: str,
    dump_path: Path,
    *,
    items: int = 25_000,
    queries: int = 50,
    k: int = 10,
    seed: int = 7,
) -> dict:
    if not dump_path.is_file():
        raise ValueError(f"Wikipedia dump not found: {dump_path}")
    if not 1 <= k <= items or queries < 1:
        raise ValueError("queries must be positive and k must not exceed items")

    parse_started = time.perf_counter()
    chunks, articles = load_wikipedia_chunks(dump_path, items)
    parse_seconds = time.perf_counter() - parse_started
    vectors, query_vectors, _ = make_corpus(
        items, queries, ACTIVE_PROFILE.embedding_dimension, min(100, items), seed
    )
    claims = ContextTokenClaims(uuid4(), uuid4(), None, None, None, None, 0, 2**31)
    compartment_id = uuid4()
    slot = None
    load_seconds = rebuild_seconds = 0.0
    latencies: list[float] = []
    recalls: list[float] = []

    try:
        with (
            psycopg.connect(database_url, autocommit=True) as connection,
            psycopg.connect(admin_database_url, autocommit=True) as monitor,
            tempfile.TemporaryDirectory(prefix="memsystem-wikipedia-") as directory,
        ):
            backend_pid = connection.execute("SELECT pg_backend_pid()").fetchone()[0]
            load_started = time.perf_counter()
            chunk_ids = _setup_corpus(connection, claims, compartment_id, chunks, vectors)
            load_seconds = time.perf_counter() - load_started

            owner = TurboVecIndex(claims.tenant_id, Path(directory) / "initial.tvim")
            slot = IndexSlot(owner)
            rebuild_started = time.perf_counter()
            rebuilt = rebuild_index(connection, claims.tenant_id, slot, directory)
            rebuild_seconds = time.perf_counter() - rebuild_started

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

            latencies = [round(value, 6) for value in latencies]
            index_bytes = rebuilt.path.stat().st_size
            postgres_peak_rss_bytes = int(
                monitor.execute(
                    "SELECT pg_read_file(%s)", (f"/proc/{backend_pid}/status",)
                ).fetchone()[0].split("VmHWM:", 1)[1].split("kB", 1)[0].strip()
            ) * 1024
            postgres_version = connection.execute("SHOW server_version").fetchone()[0]
            pgvector_version = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0]
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
        },
        "corpus": {
            "kind": "simplewiki_article_chunks_with_deterministic_clustered_vectors",
            "dump": str(dump_path),
            "dump_bytes": dump_path.stat().st_size,
            "dump_sha1": _sha1(dump_path),
            "chunk_profile": CHUNK_PROFILE,
            "articles_read": articles,
            "chunks": len(chunks),
            "source_bytes": sum(len(text.encode()) for _, text in chunks),
            "queries": queries,
            "dimension": ACTIVE_PROFILE.embedding_dimension,
            "seed": seed,
            "k": k,
        },
        "profile": {
            "name": ACTIVE_PROFILE.name,
            "turbovec_bits": ACTIVE_PROFILE.turbovec_bits,
            "candidate_multiplier": 3,
            "reranking": "authorized PostgreSQL exact inner product",
            "benchmark_vectors": "deterministic clustered vectors; semantic quality measured separately",
        },
        "measurement": {
            "latency_scope": "transaction start, tenant context, authorization allowlist, TurboVec search, exact PostgreSQL rerank, and commit after one warmup",
            "parse_seconds": round(parse_seconds, 3),
            "database_load_seconds": round(load_seconds, 3),
            "index_rebuild_seconds": round(rebuild_seconds, 3),
            "index_bytes": index_bytes,
            "benchmark_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "postgres_backend_peak_rss_bytes": postgres_peak_rss_bytes,
            **_latency_metrics(latencies),
            "latency_ms_samples": [round(value * 1000, 3) for value in latencies],
            "recall_at_k_mean": round(float(np.mean(recalls)), 6),
            "recall_at_k_min": round(min(recalls), 6),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
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
    parser.add_argument("--items", type=int, default=25_000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.database_url,
        args.admin_database_url,
        args.dump,
        items=args.items,
        queries=args.queries,
        k=args.k,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
