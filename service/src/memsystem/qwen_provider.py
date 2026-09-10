"""Pinned local Qwen embedding provider."""

from pathlib import Path
import os
from threading import Lock

from memsystem.embeddings import ACTIVE_PROFILE

QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


class QwenEmbeddingProvider:
    provider = ACTIVE_PROFILE.embedding_provider
    model = ACTIVE_PROFILE.embedding_model
    version = ACTIVE_PROFILE.embedding_version
    dimension = ACTIVE_PROFILE.embedding_dimension

    def __init__(self, *, device: str | None = None, cache_dir: str | Path | None = None):
        self.device = device or os.getenv("MEMSYSTEM_EMBEDDING_DEVICE", "auto")
        self.cache_dir = Path(
            cache_dir
            or os.getenv("MEMSYSTEM_EMBEDDING_CACHE_DIR", Path.home() / ".cache/memsystem/transformers")
        ).expanduser()
        self._lock = Lock()
        self._model = self._tokenizer = None

    def embed(self, text: str):
        return self._embed(text)

    def embed_query(self, query: str):
        return self._embed(f"Instruct: {QUERY_TASK}\nQuery:{query}")

    def _embed(self, text: str):
        import torch
        import torch.nn.functional as functional

        with self._lock:
            self._load()
            inputs = self._tokenizer(
                [text], padding=True, truncation=True, max_length=8192,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                pooled = self._model(**inputs).last_hidden_state[:, -1, : self.dimension]
                return functional.normalize(pooled, p=2, dim=1)[0].float().cpu().numpy()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            padding_side="left",
            revision=self.version,
        )
        self._model = AutoModel.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            dtype=dtype,
            revision=self.version,
        ).to(self.device)
        self._model.eval()
