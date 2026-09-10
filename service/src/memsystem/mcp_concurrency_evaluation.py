"""Measure concurrent authenticated MCP hybrid recall with the Qwen provider."""

import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

import numpy as np
import psycopg
from mcp.client.session import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from memsystem.context_token import ContextTokenClaims, ContextTokenCodec
from memsystem.embeddings import ACTIVE_PROFILE
from memsystem.evaluation import _machine, make_corpus
from memsystem.request_context import ResolvedContext
from memsystem.vector_index import IndexSlot, TurboVecIndex, rebuild_index
from memsystem.wikipedia_evaluation import DEFAULT_DUMP, _setup_corpus, load_wikipedia_chunks


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(process: subprocess.Popen, port: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MCP server exited with status {process.returncode}")
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError("MCP server did not start")


async def _worker(url, headers, queries, barrier):
    measurements = []
    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as streams:
            read, write = streams[:2]
            async with ClientSession(read, write) as session:
                await session.initialize()
                await barrier.wait()
                for query in queries:
                    started = time.perf_counter()
                    try:
                        result = await session.call_tool(
                            "memory_recall", {"query": query, "limit": 10}
                        )
                        payload = result.structured_content
                        valid_results = (
                            isinstance(payload, dict)
                            and isinstance(payload.get("results"), list)
                            and bool(payload["results"])
                            and all(
                                isinstance(item.get("vectorScore"), (int, float))
                                and isinstance(item.get("fusedScore"), (int, float))
                                for item in payload["results"]
                            )
                        )
                        if result.is_error:
                            error = "MCP tool error"
                        elif not (
                            valid_results
                            and payload.get("mode") == "hybrid"
                            and payload.get("contentTrust") == "untrusted"
                            and payload.get("nextCursor") is None
                        ):
                            error = "MCP recall returned incomplete hybrid output"
                        else:
                            error = None
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    measurements.append((time.perf_counter() - started, error))
    return measurements


async def _case(url, headers, queries, clients):
    barrier = asyncio.Barrier(clients + 1)
    groups = [queries[index::clients] for index in range(clients)]
    tasks = [asyncio.create_task(_worker(url, headers, group, barrier)) for group in groups]
    await barrier.wait()
    started = time.perf_counter()
    results = [item for group in await asyncio.gather(*tasks) for item in group]
    wall_seconds = time.perf_counter() - started
    latencies = [item[0] for item in results]
    errors = [item[1] for item in results if item[1] is not None]
    successful = len(results) - len(errors)
    return {
        "clients": clients,
        "requests": len(results),
        "errors": len(errors),
        "error_samples": errors[:3],
        "throughput_qps": round(successful / wall_seconds, 3),
        "latency_ms_p50": round(float(np.percentile(latencies, 50)) * 1000, 3),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)) * 1000, 3),
        "latency_ms_p99": round(float(np.percentile(latencies, 99)) * 1000, 3),
        "latency_ms_samples": [round(value * 1000, 3) for value in latencies],
    }


async def _run_cases(url, headers, requests, clients):
    warmup = await _case(url, headers, ["How often is the billing secret rotated?"], 1)
    cases = []
    queries = [f"What information should memory recall for request {index}?" for index in range(requests)]
    for count in clients:
        cases.append(await _case(url, headers, queries, count))
    return warmup, cases


def benchmark(
    database_url: str,
    admin_database_url: str,
    dump_path: Path,
    *,
    items: int = 25_000,
    requests: int = 64,
    clients: tuple[int, ...] = (1, 4, 8, 16, 32),
    seed: int = 7,
) -> dict:
    if (
        not clients
        or len(set(clients)) != len(clients)
        or requests < max(clients)
        or any(value < 1 for value in clients)
    ):
        raise ValueError("clients must be unique and positive, with at least one request each")
    chunks, articles = load_wikipedia_chunks(dump_path, items)
    vectors, _, _ = make_corpus(
        items, 1, ACTIVE_PROFILE.embedding_dimension, min(100, items), seed
    )
    tenant_id, user_id, compartment_id = uuid4(), uuid4(), uuid4()
    resolved = ResolvedContext(tenant_id=tenant_id, user_id=user_id)
    storage_claims = ContextTokenClaims(
        tenant_id, user_id, None, None, None, None, 0, 2**31
    )
    key = "mcp-concurrency-context-key-00000000000000000000000000000000"
    bearer = "mcp-concurrency-bearer"
    port = _free_port()
    process = None
    slot = None
    try:
        with (
            psycopg.connect(database_url, autocommit=True) as connection,
            tempfile.TemporaryDirectory(prefix="memsystem-mcp-concurrency-") as directory,
        ):
            postgres_version = connection.execute("SHOW server_version").fetchone()[0]
            pgvector_version = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0]
            load_started = time.perf_counter()
            _setup_corpus(
                connection,
                storage_claims,
                compartment_id,
                chunks,
                vectors,
            )
            load_seconds = time.perf_counter() - load_started
            slot = IndexSlot(TurboVecIndex(tenant_id, Path(directory) / "initial.tvim"))
            rebuild_started = time.perf_counter()
            rebuild_index(connection, tenant_id, slot, directory)
            rebuild_seconds = time.perf_counter() - rebuild_started
            slot.current(tenant_id).close()
            slot = None

            token = ContextTokenCodec(key, ttl_seconds=900).issue(resolved)
            device = os.getenv("MEMSYSTEM_EMBEDDING_DEVICE", "cuda")
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            env = {
                **os.environ,
                "MEMSYSTEM_DATABASE_URL": database_url,
                "MEMSYSTEM_API_TOKEN": bearer,
                "MEMSYSTEM_API_TOKEN_USER_ID": str(user_id),
                "MEMSYSTEM_API_TOKEN_EXPIRES_AT": str(int(time.time()) + 3600),
                "MEMSYSTEM_CONTEXT_TOKEN_KEY": key,
                "MEMSYSTEM_VECTOR_RECALL": "true",
                "MEMSYSTEM_EMBEDDING_DEVICE": device,
                "MEMSYSTEM_EMBEDDING_CACHE_DIR": os.getenv(
                    "MEMSYSTEM_EMBEDDING_CACHE_DIR", str(Path.home() / ".cache/memsystem/fastembed")
                ),
            }
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from memsystem.server import mcp; "
                    f"mcp.run('streamable-http', host='127.0.0.1', port={port})",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _wait_for_port(process, port)
            headers = {
                "Authorization": f"Bearer {bearer}",
                "X-Memsystem-Context": token,
            }
            warmup, cases = asyncio.run(_run_cases(
                f"http://127.0.0.1:{port}/mcp", headers, requests, clients
            ))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if slot is not None:
            slot.current(tenant_id).close()
        with psycopg.connect(admin_database_url, autocommit=True) as cleanup:
            cleanup.execute("DELETE FROM documents WHERE tenant_id = %s", (tenant_id,))
            cleanup.execute("DELETE FROM tenants WHERE id = %s", (tenant_id,))

    import torch

    driver = Path("/proc/driver/nvidia/version")
    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **_machine(),
            "mcp": version("mcp"),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "turbovec": version("turbovec"),
            "postgres": postgres_version,
            "pgvector": pgvector_version,
            "embedding_device": device,
            "embedding_dtype": "float16" if device.startswith("cuda") else "float32",
            "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": driver.read_text().splitlines()[0] if driver.exists() else None,
        },
        "corpus": {
            "dump": str(dump_path),
            "dump_sha1": _sha1(dump_path),
            "articles_read": articles,
            "chunks": len(chunks),
            "dimension": ACTIVE_PROFILE.embedding_dimension,
            "requests_per_case": requests,
            "seed": seed,
        },
        "profile": {
            "name": ACTIVE_PROFILE.name,
            "embedding_model": ACTIVE_PROFILE.embedding_model,
            "embedding_version": ACTIVE_PROFILE.embedding_version,
            "turbovec_bits": ACTIVE_PROFILE.turbovec_bits,
        },
        "measurement": {
            "scope": "authenticated Streamable HTTP MCP call, context verification, lexical search, serialized Qwen query embedding, authorized vector search, exact reranking, fusion, and response framing",
            "database_load_seconds": round(load_seconds, 3),
            "index_rebuild_seconds": round(rebuild_seconds, 3),
            "warmup_seconds_including_model_and_index_load": round(
                warmup["latency_ms_samples"][0] / 1000, 3
            ),
        },
        "cases": cases,
        "source_sha256": {
            "evaluator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "server": hashlib.sha256(
                Path(__file__).with_name("server.py").read_bytes()
            ).hexdigest(),
            "provider": hashlib.sha256(
                Path(__file__).with_name("qwen_provider.py").read_bytes()
            ).hexdigest(),
        },
    }


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
