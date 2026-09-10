"""Reproducible vector retrieval benchmark for Phase 5."""

import argparse
from importlib.metadata import version
import json
import multiprocessing
import platform
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from turbovec import IdMapIndex

from memsystem.embeddings import ACTIVE_PROFILE


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def make_corpus(items: int, queries: int, dimension: int, clusters: int, seed: int):
    """Create deterministic clustered vectors plus broad and scattered filters."""
    if items < 200 or queries < 1 or dimension < 8 or dimension % 8 or not 2 <= clusters <= items:
        raise ValueError("items >= 200, queries >= 1, dimension divisible by 8, and 2 <= clusters <= items")
    rng = np.random.default_rng(seed)
    centers = _normalize(rng.normal(size=(clusters, dimension)).astype(np.float32))
    labels = np.arange(items) % clusters
    vectors = _normalize(centers[labels] + rng.normal(0, 0.12, (items, dimension)))
    anchors = rng.integers(0, items, queries)
    query_vectors = _normalize(vectors[anchors] + rng.normal(0, 0.04, (queries, dimension)))
    ids = np.arange(1, items + 1, dtype=np.uint64)
    scattered = np.sort(rng.choice(ids, items // 10, replace=False))
    return vectors, query_vectors, {
        "unfiltered": ids,
        "broad_contiguous_50pct": ids[: items // 2],
        "scattered_random_10pct": scattered,
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _metrics(latencies: list[float], recalls: list[float]) -> dict[str, float]:
    return {
        "recall_at_k": round(float(np.mean(recalls)), 6),
        "latency_ms_p50": round(_percentile(latencies, 50) * 1000, 3),
        "latency_ms_p95": round(_percentile(latencies, 95) * 1000, 3),
        "throughput_qps": round(len(latencies) / sum(latencies), 3),
    }


def _recall(expected: list[int], actual: list[int]) -> float:
    return len(set(expected) & set(actual)) / len(expected) if expected else 1.0


def _exact_rerank(
    vectors: np.ndarray, query: np.ndarray, candidate_ids: list[int], limit: int
) -> list[int]:
    ids = np.asarray(candidate_ids, dtype=np.int64)
    scores = vectors[ids - 1] @ query
    return ids[np.lexsort((ids, -scores))[:limit]].tolist()


def _machine() -> dict[str, object]:
    cpu_model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
    }


def _status_memory_bytes(status: str, field: str) -> int:
    for line in status.splitlines():
        if line.startswith(f"{field}:"):
            value, unit = line.removeprefix(f"{field}:").split()
            if unit != "kB":
                raise ValueError(f"unexpected {field} unit: {unit}")
            return int(value) * 1024
    raise ValueError(f"{field} missing from process status")


def _peak_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        return _status_memory_bytes(status.read_text(), "VmHWM")
    # Linux reports KiB; macOS reports bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


def _postgres_peak_rss_bytes(connection, backend_pid: int) -> int:
    status = connection.execute(
        "SELECT pg_read_file(%s)", (f"/proc/{backend_pid}/status",)
    ).fetchone()[0]
    return _status_memory_bytes(status, "VmHWM")


def _configure_postgres(connection, exact: bool) -> None:
    connection.execute("SET LOCAL max_parallel_workers_per_gather = 0")
    connection.execute("SET LOCAL max_parallel_maintenance_workers = 0")
    connection.execute(
        "SELECT set_config('enable_seqscan', %s, true)",
        ("on" if exact else "off",),
    )
    connection.execute(
        "SELECT set_config('enable_indexscan', %s, true)",
        ("off" if exact else "on",),
    )
    connection.execute("SELECT set_config('enable_bitmapscan', 'off', true)")
    if not exact:
        connection.execute("SET LOCAL hnsw.ef_search = 40")
        connection.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")


def _postgres_statement(filtered: bool, *, explain: bool = False) -> str:
    prefix = "EXPLAIN (FORMAT JSON) " if explain else ""
    where = "WHERE id = ANY(%s) " if filtered else ""
    return prefix + "SELECT id FROM retrieval_benchmark " + where + "ORDER BY embedding <#> %s LIMIT %s"


def _postgres_results(connection, query, limit: int, allowlist: np.ndarray, filtered: bool):
    parameters = (allowlist.tolist(), query, limit) if filtered else (query, limit)
    return [
        row[0]
        for row in connection.execute(
            _postgres_statement(filtered), parameters
        ).fetchall()
    ]


def _uses_hnsw(connection, query, limit: int, allowlist: np.ndarray, filtered: bool) -> bool:
    parameters = (allowlist.tolist(), query, limit) if filtered else (query, limit)
    plan = connection.execute(
        _postgres_statement(filtered, explain=True), parameters
    ).fetchone()[0][0]["Plan"]
    pending = [plan]
    while pending:
        node = pending.pop()
        if node.get("Index Name") == "retrieval_benchmark_hnsw":
            return True
        pending.extend(node.get("Plans", []))
    return False


def _run_postgres(connection, queries, cases, limit: int, exact: bool, ground_truth=None):
    _configure_postgres(connection, exact)
    all_results: dict[str, list[list[int]]] = {}
    case_metrics = {}
    for name, allowlist in cases.items():
        filtered = name != "unfiltered"
        if not exact and not _uses_hnsw(connection, queries[0], limit, allowlist, filtered):
            raise RuntimeError(f"HNSW plan not used for {name}")
        _postgres_results(connection, queries[0], limit, allowlist, filtered)
        latencies, results = [], []
        for query in queries:
            started = time.perf_counter()
            ids = _postgres_results(connection, query, limit, allowlist, filtered)
            latencies.append(time.perf_counter() - started)
            results.append(ids)
        all_results[name] = results
        recalls = [1.0] * len(results) if ground_truth is None else [
            _recall(expected, actual)
            for expected, actual in zip(ground_truth[name], results)
        ]
        case_metrics[name] = _metrics(latencies, recalls)
    return case_metrics, all_results


def _run_turbovec(
    vectors, queries, cases, limit: int, bits: int, ground_truth, directory: Path,
    candidate_multipliers: tuple[int, ...],
):
    path = directory / f"turbovec-{bits}.tvim"
    baseline_bytes = _peak_rss_bytes()
    started = time.perf_counter()
    index = IdMapIndex(dim=vectors.shape[1], bit_width=bits)
    index.add_with_ids(vectors, np.arange(1, len(vectors) + 1, dtype=np.uint64))
    index.prepare()
    index.sync(str(path))
    build_seconds = time.perf_counter() - started
    case_metrics = {}
    reranked_metrics = {multiplier: {} for multiplier in candidate_multipliers}
    multipliers = sorted(set(candidate_multipliers) | {3})
    for name, allowlist in cases.items():
        prepared_allowlist = np.ascontiguousarray(allowlist, dtype=np.uint64)
        for multiplier in multipliers:
            _, warm_ids = index.search(
                np.ascontiguousarray([queries[0]], dtype=np.float32),
                min(limit * multiplier, len(allowlist)),
                allowlist=prepared_allowlist,
            )
            _exact_rerank(
                vectors, queries[0], [int(value) for value in warm_ids[0]], limit
            )
        measurements = {
            multiplier: {"latencies": [], "recalls": [], "candidate_recalls": []}
            for multiplier in candidate_multipliers
        }
        approximate_latencies, approximate_recalls = [], []
        for query_index, (query, expected) in enumerate(zip(queries, ground_truth[name])):
            offset = query_index % len(multipliers)
            for multiplier in multipliers[offset:] + multipliers[:offset]:
                started = time.perf_counter()
                _, ids = index.search(
                    np.ascontiguousarray([query], dtype=np.float32),
                    min(limit * multiplier, len(allowlist)),
                    allowlist=prepared_allowlist,
                )
                search_elapsed = time.perf_counter() - started
                candidates = [int(value) for value in ids[0]]
                if multiplier == 3:
                    approximate_latencies.append(search_elapsed)
                    approximate_recalls.append(_recall(expected, candidates[:limit]))
                if multiplier in measurements:
                    reranked = _exact_rerank(vectors, query, candidates, limit)
                    measurement = measurements[multiplier]
                    measurement["latencies"].append(time.perf_counter() - started)
                    measurement["recalls"].append(_recall(expected, reranked))
                    measurement["candidate_recalls"].append(
                        _recall(expected, candidates)
                    )
        case_metrics[name] = _metrics(approximate_latencies, approximate_recalls)
        for multiplier, measurement in measurements.items():
            metrics = _metrics(measurement["latencies"], measurement["recalls"])
            metrics["candidate_recall_at_k"] = round(
                float(np.mean(measurement["candidate_recalls"])), 6
            )
            reranked_metrics[multiplier][name] = metrics
    size = path.stat().st_size
    del index
    started = time.perf_counter()
    recovered = IdMapIndex.load(str(path))
    recovered.prepare()
    load_seconds = time.perf_counter() - started
    peak_bytes = _peak_rss_bytes()
    peak_delta_bytes = max(0, peak_bytes - baseline_bytes)
    del recovered
    shared = {
        "build_seconds": round(build_seconds, 3),
        "load_seconds": round(load_seconds, 3),
        "index_bytes": size,
        "process_peak_rss_bytes": peak_bytes,
        "process_peak_rss_delta_bytes": peak_delta_bytes,
    }
    reports = {"": {
        **shared,
        "settings": {"candidate_multiplier": 3, "reranking": "none"},
        "cases": case_metrics,
    }}
    for multiplier, metrics in reranked_metrics.items():
        reports[f"_reranked_{multiplier}x"] = {
            **shared,
            "settings": {
                "candidate_multiplier": multiplier,
                "reranking": "exact_full_precision_inner_product",
            },
            "cases": metrics,
        }
    return reports


def _turbovec_worker(
    send, vectors, queries, cases, limit, bits, ground_truth, directory,
    candidate_multipliers,
):
    try:
        send.send(_run_turbovec(
            vectors, queries, cases, limit, bits, ground_truth, Path(directory),
            candidate_multipliers,
        ))
    except Exception as error:
        send.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        send.close()


def _run_turbovec_isolated(
    vectors, queries, cases, limit, bits, ground_truth, directory,
    candidate_multipliers,
):
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_turbovec_worker,
        args=(
            send, vectors, queries, cases, limit, bits, ground_truth, directory,
            candidate_multipliers,
        ),
    )
    process.start()
    send.close()
    if not receive.poll(300):
        process.terminate()
        process.join()
        raise TimeoutError("TurboVec worker exceeded 300 seconds")
    try:
        result = receive.recv()
    except EOFError:
        result = {"error": "TurboVec worker exited without a result"}
    process.join()
    if process.exitcode or "error" in result:
        raise RuntimeError(result.get("error", f"TurboVec worker exited {process.exitcode}"))
    return result


def _aggregate_reports(reports: list[dict]) -> dict:
    """Summarize the mean and worst observed decision signals across seeds."""
    if not reports:
        raise ValueError("at least one benchmark report is required")
    engines = {}
    for name in reports[0]["engines"]:
        runs = [report["engines"][name] for report in reports]
        engines[name] = {
            "process_peak_rss_delta_bytes_mean": round(float(np.mean([
                run["process_peak_rss_delta_bytes"] for run in runs
            ]))),
            "process_peak_rss_delta_bytes_max": max(
                run["process_peak_rss_delta_bytes"] for run in runs
            ),
            "cases": {},
        }
        for case in runs[0]["cases"]:
            metrics = [run["cases"][case] for run in runs]
            summary = {
                "recall_at_k_mean": round(float(np.mean([
                    item["recall_at_k"] for item in metrics
                ])), 6),
                "recall_at_k_min": min(item["recall_at_k"] for item in metrics),
                "latency_ms_p95_mean": round(float(np.mean([
                    item["latency_ms_p95"] for item in metrics
                ])), 3),
                "latency_ms_p95_max": max(
                    item["latency_ms_p95"] for item in metrics
                ),
            }
            if "candidate_recall_at_k" in metrics[0]:
                summary.update({
                    "candidate_recall_at_k_mean": round(float(np.mean([
                        item["candidate_recall_at_k"] for item in metrics
                    ])), 6),
                    "candidate_recall_at_k_min": min(
                        item["candidate_recall_at_k"] for item in metrics
                    ),
                })
            engines[name]["cases"][case] = summary
    return {"runs": len(reports), "engines": engines}


def _resolve_seeds(seed: int, seeds: list[int] | None) -> list[int]:
    resolved = seeds or [seed]
    if any(value < 0 for value in resolved) or len(resolved) != len(set(resolved)):
        raise ValueError("seeds must be unique non-negative integers")
    return resolved


def _resolve_candidate_multipliers(values: list[int] | None) -> tuple[int, ...]:
    resolved = tuple(values or ())
    if any(type(value) is not int or not 1 <= value <= 10 for value in resolved) or len(resolved) != len(set(resolved)):
        raise ValueError("candidate multipliers must be unique integers from 1 through 10")
    return resolved


def benchmark(
    database_url: str, *, items: int, queries: int, dimension: int, clusters: int,
    k: int, seed: int, candidate_multipliers: tuple[int, ...] = (),
):
    vectors, query_vectors, cases = make_corpus(items, queries, dimension, clusters, seed)
    candidate_multipliers = _resolve_candidate_multipliers(list(candidate_multipliers))
    if type(k) is not int or not 1 <= k <= min(map(len, cases.values())):
        raise ValueError("k must be positive and no larger than the smallest allowlist")
    report = {
        "environment": {
            "platform": platform.platform(),
            **_machine(),
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pgvector_python": version("pgvector"),
            "postgres": None,
            "pgvector_postgres": None,
            "turbovec": version("turbovec"),
        },
        "measurement": {
            "latency_scope": "engine call after one warmup query; PostgreSQL includes client serialization and server round trip; reranked variants include candidate conversion and full-precision inner products",
            "throughput": "serial inverse mean latency",
            "memory": "process VmHWM growth from a fresh isolated process baseline through build, load, and search; PostgreSQL parallel workers disabled",
        },
        "corpus": {
            "kind": "deterministic_clustered_vectors",
            "items": items,
            "queries": queries,
            "dimension": dimension,
            "clusters": clusters,
            "seed": seed,
            "k": k,
            "filters": {name: len(ids) for name, ids in cases.items()},
        },
        "engines": {},
    }
    with (
        psycopg.connect(database_url) as connection,
        psycopg.connect(database_url, autocommit=True) as monitor,
        connection.transaction(),
    ):
        register_vector(connection)
        backend_pid = connection.execute("SELECT pg_backend_pid()").fetchone()[0]
        report["environment"]["postgres"] = connection.execute(
            "SHOW server_version"
        ).fetchone()[0]
        report["environment"]["pgvector_postgres"] = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0]
        connection.execute(
            f"CREATE TEMP TABLE retrieval_benchmark "
            f"(id bigint NOT NULL, embedding vector({dimension}) NOT NULL)"
        )
        baseline_bytes = _postgres_peak_rss_bytes(monitor, backend_pid)
        started = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO retrieval_benchmark VALUES (%s, %s)",
                ((index, vector) for index, vector in enumerate(vectors, 1)),
            )
        insert_seconds = time.perf_counter() - started
        connection.execute("ANALYZE retrieval_benchmark")
        exact_metrics, ground_truth = _run_postgres(
            connection, query_vectors, cases, k, exact=True
        )
        table_bytes = connection.execute(
            "SELECT pg_table_size('retrieval_benchmark')"
        ).fetchone()[0]
        exact_peak_bytes = _postgres_peak_rss_bytes(monitor, backend_pid)
        report["engines"]["pgvector_exact"] = {
            "build_seconds": round(insert_seconds, 3),
            "index_bytes": 0,
            "table_bytes": table_bytes,
            "process_peak_rss_bytes": exact_peak_bytes,
            "process_peak_rss_delta_bytes": max(0, exact_peak_bytes - baseline_bytes),
            "cases": exact_metrics,
        }

    with (
        psycopg.connect(database_url) as connection,
        psycopg.connect(database_url, autocommit=True) as monitor,
        connection.transaction(),
    ):
        register_vector(connection)
        backend_pid = connection.execute("SELECT pg_backend_pid()").fetchone()[0]
        connection.execute(
            f"CREATE TEMP TABLE retrieval_benchmark "
            f"(id bigint NOT NULL, embedding vector({dimension}) NOT NULL)"
        )
        baseline_bytes = _postgres_peak_rss_bytes(monitor, backend_pid)
        started = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO retrieval_benchmark VALUES (%s, %s)",
                ((index, vector) for index, vector in enumerate(vectors, 1)),
            )
        connection.execute("ANALYZE retrieval_benchmark")
        _configure_postgres(connection, exact=False)
        connection.execute(
            "CREATE INDEX retrieval_benchmark_hnsw ON retrieval_benchmark "
            "USING hnsw (embedding vector_ip_ops)"
        )
        hnsw_build_seconds = time.perf_counter() - started
        connection.execute("ANALYZE retrieval_benchmark")
        hnsw_metrics, _ = _run_postgres(
            connection, query_vectors, cases, k, exact=False, ground_truth=ground_truth
        )
        hnsw_bytes = connection.execute(
            "SELECT pg_relation_size('retrieval_benchmark_hnsw')"
        ).fetchone()[0]
        table_bytes = connection.execute(
            "SELECT pg_table_size('retrieval_benchmark')"
        ).fetchone()[0]
        backend_memory = connection.execute(
            "SELECT sum(total_bytes)::bigint FROM pg_backend_memory_contexts"
        ).fetchone()[0]
        hnsw_peak_bytes = _postgres_peak_rss_bytes(monitor, backend_pid)
        report["engines"]["pgvector_hnsw"] = {
            "build_seconds": round(hnsw_build_seconds, 3),
            "index_bytes": hnsw_bytes,
            "table_bytes": table_bytes,
            "process_peak_rss_bytes": hnsw_peak_bytes,
            "process_peak_rss_delta_bytes": max(0, hnsw_peak_bytes - baseline_bytes),
            "backend_memory_context_bytes": backend_memory,
            "settings": {
                "plans_verified": list(cases),
                "ef_search": int(connection.execute("SHOW hnsw.ef_search").fetchone()[0]),
                "iterative_scan": "strict_order",
                "index_options": "defaults",
                "parallel_workers": 0,
            },
            "cases": hnsw_metrics,
        }

    with tempfile.TemporaryDirectory(prefix="memsystem-eval-") as directory:
        for bits in (4, 2):
            variants = _run_turbovec_isolated(
                vectors, query_vectors, cases, k, bits, ground_truth, directory,
                candidate_multipliers,
            )
            for suffix, result in variants.items():
                report["engines"][f"turbovec_{bits}bit{suffix}"] = result
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "MEMSYSTEM_TEST_ADMIN_DATABASE_URL",
            "postgresql://postgres:postgres@127.0.0.1:5432/memsystem",
        ),
    )
    parser.add_argument("--items", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--dimension", type=int, default=ACTIVE_PROFILE.embedding_dimension)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="run and aggregate several unique seeds instead of --seed",
    )
    parser.add_argument(
        "--candidate-multipliers", type=int, nargs="+",
        help="exactly rerank k times each multiplier of TurboVec candidates",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.items, args.queries, args.dimension, args.clusters = 256, 8, 64, 16
    seeds = _resolve_seeds(args.seed, args.seeds)
    candidate_multipliers = _resolve_candidate_multipliers(args.candidate_multipliers)
    reports = [
        benchmark(
            args.database_url,
            items=args.items,
            queries=args.queries,
            dimension=args.dimension,
            clusters=args.clusters,
            k=args.k,
            seed=seed,
            candidate_multipliers=candidate_multipliers,
        )
        for seed in seeds
    ]
    report = reports[0] if len(reports) == 1 else {
        "seeds": seeds,
        "aggregate": _aggregate_reports(reports),
        "reports": reports,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
