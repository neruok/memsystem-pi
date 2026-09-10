"""Evaluate a local embedding model against labeled project documentation."""

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import numpy as np

from memsystem.chunking import CHUNK_PROFILE, split_markdown

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_REVISIONS = {
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "Qwen/Qwen3-Embedding-4B": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "Qwen/Qwen3-Embedding-8B": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
}


@dataclass(frozen=True)
class CorpusChunk:
    identifier: str
    path: str
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Judgment:
    query: str
    relevant: frozenset[int]


def _load_manifest(
    path: Path,
    seen: frozenset[Path] = frozenset(),
    directory: Path | None = None,
) -> dict:
    path = path.resolve()
    directory = directory or path.parent
    if not path.is_relative_to(directory):
        raise ValueError("corpus manifest base must stay in its directory")
    if path in seen:
        raise ValueError("corpus manifest inheritance cycle")
    data = json.loads(path.read_text())
    if data.get("version") != 1 or not isinstance(data.get("documents"), list):
        raise ValueError("corpus manifest must use version 1 and contain documents")
    if base := data.get("base"):
        if not isinstance(base, str):
            raise ValueError("corpus manifest base must be a string")
        inherited = _load_manifest(path.parent / base, seen | {path}, directory)
        data = {
            **data,
            "documents": inherited["documents"] + data["documents"],
            "queries": inherited.get("queries", []) + data.get("queries", []),
        }
    return data


def load_corpus(root: Path, manifest_path: Path) -> tuple[list[CorpusChunk], list[Judgment]]:
    """Load project files and resolve labeled selectors to chunk indexes."""
    root = root.resolve()
    data = _load_manifest(manifest_path)

    if not data["documents"] or any(not isinstance(item, str) for item in data["documents"]):
        raise ValueError("corpus documents must be nonempty strings")
    if len(data["documents"]) != len(set(data["documents"])):
        raise ValueError("corpus documents must be unique")
    chunks = []
    for relative in data["documents"]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"invalid corpus document: {relative}")
        for position, chunk in enumerate(split_markdown(path.read_text())):
            chunks.append(CorpusChunk(
                f"{relative}:{position}", relative, chunk.heading_path, chunk.text
            ))

    judgments = []
    for item in data.get("queries", []):
        query = item.get("query")
        selectors = item.get("relevant")
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(selectors, list)
            or not selectors
        ):
            raise ValueError("each query needs text and relevant selectors")
        relevant = set()
        for selector in selectors:
            if not isinstance(selector, dict) or set(selector) - {"path", "heading", "contains"}:
                raise ValueError(f"invalid relevance selector for query: {query}")
            path, heading = selector.get("path"), selector.get("heading")
            contains = selector.get("contains")
            if (
                not isinstance(path, str)
                or not isinstance(heading, str)
                or (contains is not None and not isinstance(contains, str))
            ):
                raise ValueError(f"invalid relevance selector for query: {query}")
            matches = {
                index
                for index, chunk in enumerate(chunks)
                if chunk.path == path
                and chunk.heading_path
                and chunk.heading_path[-1] == heading
                and (contains is None or contains in chunk.text)
            }
            if not matches:
                raise ValueError(f"relevance selector matched no chunks for query: {query}")
            relevant.update(matches)
        judgments.append(Judgment(query, frozenset(relevant)))
    if not judgments:
        raise ValueError("corpus manifest must contain queries")
    return chunks, judgments


def _first_relevant_rank(ranking: list[int], relevant: frozenset[int], k: int) -> int | None:
    return next(
        (index for index, item in enumerate(ranking[:k], 1) if item in relevant),
        None,
    )


def _rank_metrics(rankings: list[list[int]], judgments: list[Judgment], k: int) -> dict[str, float]:
    recalls, reciprocal_ranks, hits = [], [], []
    for ranking, judgment in zip(rankings, judgments):
        top = ranking[:k]
        recalls.append(len(set(top) & judgment.relevant) / len(judgment.relevant))
        rank = _first_relevant_rank(ranking, judgment.relevant, k)
        reciprocal_ranks.append(1 / rank if rank else 0)
        hits.append(bool(set(top) & judgment.relevant))
    return {
        f"hit_rate_at_{k}": round(float(np.mean(hits)), 6),
        f"mean_recall_at_{k}": round(float(np.mean(recalls)), 6),
        f"mean_reciprocal_rank_at_{k}": round(float(np.mean(reciprocal_ranks)), 6),
    }


def _exact_rankings(documents: np.ndarray, queries: np.ndarray) -> list[list[int]]:
    return [
        [int(index) for index in np.argsort(-(documents @ query))]
        for query in queries
    ]


def _turbovec_rankings(documents: np.ndarray, queries: np.ndarray, bits: int) -> list[list[int]]:
    from turbovec import IdMapIndex

    index = IdMapIndex(dim=documents.shape[1], bit_width=bits)
    ids = np.arange(1, len(documents) + 1, dtype=np.uint64)
    index.add_with_ids(documents, ids)
    index.prepare()
    rankings = []
    for query in queries:
        _, found = index.search(
            np.ascontiguousarray([query], dtype=np.float32),
            len(documents),
            allowlist=ids,
        )
        rankings.append([int(item) - 1 for item in found[0]])
    return rankings


def _qwen_revision(model_name: str, revision: str | None) -> str:
    revision = revision or QWEN_REVISIONS.get(model_name)
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Qwen models require a full lowercase commit revision")
    return revision


def _embed_fastembed(chunks, judgments, model_name: str, cache_dir: Path):
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=1)
    documents = list(model.embed([chunk.text for chunk in chunks]))
    queries = list(model.query_embed([item.query for item in judgments]))
    return documents, queries, {"backend": "fastembed", "fastembed": version("fastembed")}


def _embed_qwen(
    chunks,
    judgments,
    model_name: str,
    cache_dir: Path,
    dimensions: int | None,
    device: str,
    batch_size: int,
    max_length: int,
    task: str,
    revision: str,
):
    import torch
    import torch.nn.functional as functional
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
    texts = [chunk.text for chunk in chunks] + [
        f"Instruct: {task}\nQuery:{item.query}" for item in judgments
    ]
    vectors = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**inputs).last_hidden_state
            pooled = hidden[:, -1]
            if dimensions is not None:
                if not 32 <= dimensions <= pooled.shape[1]:
                    raise ValueError("Qwen dimensions must be within the model output")
                pooled = pooled[:, :dimensions]
            pooled = functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.float().cpu().numpy())
    split = len(chunks)
    metadata = {
        "backend": "qwen-transformers",
        "torch": version("torch"),
        "transformers": version("transformers"),
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "model_revision": revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", revision),
        "task_instruction": task,
        "max_length": max_length,
        "batch_size": batch_size,
    }
    if device.startswith("cuda"):
        metadata["gpu"] = torch.cuda.get_device_name(device)
        metadata["cuda_runtime"] = torch.version.cuda
        driver = Path("/proc/driver/nvidia/version")
        metadata["nvidia_driver"] = driver.read_text().splitlines()[0] if driver.exists() else None
        metadata["peak_pytorch_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return vectors[:split], vectors[split:], metadata


def evaluate(
    root: Path,
    manifest_path: Path,
    model_name: str,
    cache_dir: Path,
    k: int = 5,
    *,
    backend: str = "fastembed",
    dimensions: int | None = None,
    device: str = "auto",
    batch_size: int = 8,
    max_length: int = 8192,
    exact_only: bool = False,
    task: str = DEFAULT_QWEN_TASK,
    revision: str | None = None,
    quality_only: bool = False,
) -> dict:
    if type(k) is not int or k < 1:
        raise ValueError("k must be positive")
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch size and maximum length must be positive")
    chunks, judgments = load_corpus(root, manifest_path)
    source_payload = json.dumps(
        {
            "chunks": [
                [chunk.identifier, chunk.heading_path, chunk.text] for chunk in chunks
            ],
            "judgments": [
                [item.query, sorted(item.relevant)] for item in judgments
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()

    started = time.perf_counter()
    if backend == "fastembed":
        raw_documents, raw_queries, backend_metadata = _embed_fastembed(
            chunks, judgments, model_name, cache_dir
        )
    elif backend == "qwen":
        revision = _qwen_revision(model_name, revision)
        raw_documents, raw_queries, backend_metadata = _embed_qwen(
            chunks, judgments, model_name, cache_dir, dimensions,
            device, batch_size, max_length, task, revision,
        )
    else:
        raise ValueError(f"unsupported embedding backend: {backend}")
    documents = np.ascontiguousarray(raw_documents, dtype=np.float32)
    queries = np.ascontiguousarray(raw_queries, dtype=np.float32)
    embedding_seconds = time.perf_counter() - started
    documents /= np.linalg.norm(documents, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    if not exact_only and documents.shape[1] % 8:
        raise ValueError("TurboVec dimensions must be divisible by 8")

    rankings = {"exact": _exact_rankings(documents, queries)}
    if not exact_only:
        rankings.update({
            f"turbovec_{bits}bit": _turbovec_rankings(documents, queries, bits)
            for bits in (4, 2)
        })
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        **backend_metadata,
    }
    if not exact_only:
        environment["turbovec"] = version("turbovec")
    embedding = {
        "model": model_name,
        "dimension": int(documents.shape[1]),
        "vectors_sha256": hashlib.sha256(
            documents.tobytes() + queries.tobytes()
        ).hexdigest(),
    }
    if not quality_only:
        embedding["seconds"] = round(embedding_seconds, 3)
    return {
        "environment": {
            **environment,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "corpus": {
            "manifest": str(manifest_path),
            "chunk_profile": CHUNK_PROFILE,
            "documents": len(chunks),
            "queries": len(judgments),
            "source_sha256": hashlib.sha256(source_payload).hexdigest(),
        },
        "embedding": embedding,
        "metrics": {
            name: _rank_metrics(result, judgments, k)
            for name, result in rankings.items()
        },
        "queries": [
            {
                "query": judgment.query,
                "relevant": [chunks[index].identifier for index in sorted(judgment.relevant)],
                "rankings": {
                    name: {
                        f"first_relevant_rank_at_{k}": _first_relevant_rank(
                            ranking[position], judgment.relevant, k
                        ),
                        "top": [
                            chunks[index].identifier
                            for index in ranking[position][:k]
                        ],
                    }
                    for name, ranking in rankings.items()
                },
            }
            for position, judgment in enumerate(judgments)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("service/benchmarks/project-docs-corpus.json"),
    )
    parser.add_argument("--backend", choices=("fastembed", "qwen"), default="fastembed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/memsystem/fastembed")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--task", default=DEFAULT_QWEN_TASK)
    parser.add_argument("--revision")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.root, args.manifest, args.model, args.cache_dir, args.k,
        backend=args.backend,
        dimensions=args.dimensions,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        exact_only=args.exact_only,
        task=args.task,
        revision=args.revision,
        quality_only=args.quality_only,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
