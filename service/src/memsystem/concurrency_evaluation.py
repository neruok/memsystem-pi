"""Measure concurrent authorized retrieval with Wikipedia-shaped storage."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import numpy as np
import psycopg

from memsystem.context_token import ContextTokenClaims
from memsystem.database import tenant_transaction
from memsystem.embeddings import ACTIVE_PROFILE
from memsystem.evaluation import make_corpus
from memsystem.retrieval import search_vector
from memsystem.vector_index import IndexSlot, TurboVecIndex, rebuild_index
from memsystem.wikipedia_evaluation import DEFAULT_DUMP, _setup_corpus, load_wikipedia_chunks


def _search(database_url, claims, slot, query, limit):
    started = time.perf_counter()
    try:
        with psycopg.connect(database_url, autocommit=True) as connection:
            with tenant_transaction(connection, claims.tenant_id):
                results = search_vector(connection, claims, slot, query, limit=limit)
        return time.perf_counter() - started, [item.chunk_id for item in results], None
    except Exception as error:
        return time.perf_counter() - started, [], f"{type(error).__name__}: {error}"


def _case(database_url, claims, slot, queries, expected, limit, workers):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            lambda pair: _search(database_url, claims, slot, pair[1], limit),
            enumerate(queries),
        ))
    wall_seconds = time.perf_counter() - started
    latencies = [item[0] for item in results]
    errors = [item[2] for item in results if item[2] is not None]
    successful = len(results) - len(errors)
    recalls = [
        0.0 if item[2] is not None else len(set(item[1]) & expected[index]) / limit
        for index, item in enumerate(results)
    ]
    return {
        "clients": workers,
        "requests": len(results),
        "errors": len(errors),
        "error_samples": errors[:3],
        "throughput_qps": round(successful / wall_seconds, 3),
        "latency_ms_p50": round(float(np.percentile(latencies, 50)) * 1000, 3),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)) * 1000, 3),
        "latency_ms_p99": round(float(np.percentile(latencies, 99)) * 1000, 3),
        "recall_at_k_mean": round(float(np.mean(recalls)), 6) if recalls else 0.0,
        "recall_at_k_min": round(min(recalls), 6) if recalls else 0.0,
    }


def benchmark(
    database_url: str,
    admin_database_url: str,
    dump_path: Path,
    *,
    items: int = 25_000,
    requests: int = 64,
    clients: tuple[int, ...] = (1, 4, 8, 16, 32),
    k: int = 10,
    seed: int = 7,
) -> dict:
    if (
        not clients
        or len(set(clients)) != len(clients)
        or requests < max(clients)
        or any(value < 1 for value in clients)
    ):
        raise ValueError("clients must be unique and positive, with at least one request each")
    if not 1 <= k <= min(100, items):
        raise ValueError("k must be from 1 through 100 and must not exceed items")
    chunks, articles = load_wikipedia_chunks(dump_path, items)
    vectors, query_vectors, _ = make_corpus(
        items, requests, ACTIVE_PROFILE.embedding_dimension, min(100, items), seed
    )
    claims = ContextTokenClaims(uuid4(), uuid4(), None, None, None, None, 0, 2**31)
    compartment_id = uuid4()
    slot = None
    try:
        with (
            psycopg.connect(database_url, autocommit=True) as connection,
            tempfile.TemporaryDirectory(prefix="memsystem-concurrency-") as directory,
        ):
            load_started = time.perf_counter()
            chunk_ids = _setup_corpus(connection, claims, compartment_id, chunks, vectors)
            load_seconds = time.perf_counter() - load_started
            slot = IndexSlot(TurboVecIndex(claims.tenant_id, Path(directory) / "initial.tvim"))
            rebuild_started = time.perf_counter()
            rebuild_index(connection, claims.tenant_id, slot, directory)
            initial_rebuild_seconds = time.perf_counter() - rebuild_started

            expected = [
                {
                    chunk_ids[int(index)]
                    for index in np.argsort(-(vectors @ query), kind="stable")[:k]
                }
                for query in query_vectors
            ]
            _search(database_url, claims, slot, query_vectors[0], k)
            cases = [
                _case(database_url, claims, slot, query_vectors, expected, k, workers)
                for workers in clients
            ]

            rebuild_queries = []
            rebuild_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=1) as executor:
                rebuilding = executor.submit(
                    _rebuild, database_url, claims.tenant_id, slot, directory
                )
                index = 0
                while not rebuilding.done():
                    rebuild_queries.append(_search(
                        database_url, claims, slot,
                        query_vectors[index % len(query_vectors)], k,
                    ))
                    index += 1
                rebuilding.result()
            rebuild_seconds = time.perf_counter() - rebuild_started
            rebuild_errors = [item[2] for item in rebuild_queries if item[2] is not None]
            rebuild_latencies = [item[0] for item in rebuild_queries]
            rebuild_recalls = [
                len(set(item[1]) & expected[index % len(expected)]) / k
                for index, item in enumerate(rebuild_queries)
                if item[2] is None
            ]
    finally:
        if slot is not None:
            slot.current(claims.tenant_id).close()
        with psycopg.connect(admin_database_url, autocommit=True) as cleanup:
            cleanup.execute("DELETE FROM documents WHERE tenant_id = %s", (claims.tenant_id,))
            cleanup.execute("DELETE FROM tenants WHERE id = %s", (claims.tenant_id,))

    return {
        "corpus": {
            "dump": str(dump_path),
            "articles_read": articles,
            "chunks": len(chunks),
            "dimension": ACTIVE_PROFILE.embedding_dimension,
            "requests_per_case": requests,
            "k": k,
            "seed": seed,
        },
        "profile": {
            "name": ACTIVE_PROFILE.name,
            "turbovec_bits": ACTIVE_PROFILE.turbovec_bits,
            "candidate_multiplier": 3,
        },
        "setup": {
            "database_load_seconds": round(load_seconds, 3),
            "initial_index_rebuild_seconds": round(initial_rebuild_seconds, 3),
        },
        "cases": cases,
        "search_during_rebuild": {
            "rebuild_seconds": round(rebuild_seconds, 3),
            "requests": len(rebuild_queries),
            "errors": len(rebuild_errors),
            "error_samples": rebuild_errors[:3],
            "latency_ms_p95": (
                round(float(np.percentile(rebuild_latencies, 95)) * 1000, 3)
                if rebuild_latencies else None
            ),
            "recall_at_k_min": round(min(rebuild_recalls), 6) if rebuild_recalls else 0.0,
        },
    }


def _rebuild(database_url, tenant_id, slot, directory):
    with psycopg.connect(database_url, autocommit=True) as connection:
        rebuild_index(connection, tenant_id, slot, directory)


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
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.database_url,
        args.admin_database_url,
        args.dump,
        items=args.items,
        requests=args.requests,
        clients=tuple(args.clients),
        k=args.k,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
