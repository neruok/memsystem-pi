"""Active embedding profile and vector validation."""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from memsystem.chunking import CHUNK_PROFILE


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    chunk_profile: str
    embedding_provider: str
    embedding_model: str
    embedding_version: str
    embedding_dimension: int
    unit_normalized: bool
    text_search_config: str
    turbovec_version: str
    turbovec_bits: int
    turbovec_calibrated: bool
    turbovec_file_format: str


ACTIVE_PROFILE = RetrievalProfile(
    name="qwen3-4b-1536-v1",
    chunk_profile=CHUNK_PROFILE,
    embedding_provider="transformers",
    embedding_model="Qwen/Qwen3-Embedding-4B",
    embedding_version="5cf2132abc99cad020ac570b19d031efec650f2b",
    embedding_dimension=1536,
    unit_normalized=True,
    text_search_config="simple",
    turbovec_version="1.0.0",
    turbovec_bits=4,
    turbovec_calibrated=False,
    turbovec_file_format="1.0",
)


class EmbeddingProvider(Protocol):
    provider: str
    model: str
    version: str
    dimension: int

    def embed(self, text: str) -> Sequence[float]: ...


def normalize_embedding(values: Sequence[float]) -> NDArray[np.float32]:
    """Return one finite, non-zero unit vector for the active profile."""
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (ACTIVE_PROFILE.embedding_dimension,) or not np.isfinite(vector).all():
        raise ValueError(
            f"embedding must contain {ACTIVE_PROFILE.embedding_dimension} finite values"
        )
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("embedding must be non-zero")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)
