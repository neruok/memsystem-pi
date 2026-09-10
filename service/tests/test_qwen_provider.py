from memsystem.embeddings import ACTIVE_PROFILE
from memsystem.qwen_provider import QUERY_TASK, QwenEmbeddingProvider


def test_qwen_provider_uses_pinned_profile_and_query_instruction(monkeypatch):
    provider = QwenEmbeddingProvider(device="cpu", cache_dir="/tmp/models")
    seen = []
    monkeypatch.setattr(provider, "_embed", lambda text: seen.append(text) or text)

    assert provider.embed("document") == "document"
    assert provider.embed_query("question") == f"Instruct: {QUERY_TASK}\nQuery:question"
    assert (provider.provider, provider.model, provider.version, provider.dimension) == (
        ACTIVE_PROFILE.embedding_provider,
        ACTIVE_PROFILE.embedding_model,
        ACTIVE_PROFILE.embedding_version,
        ACTIVE_PROFILE.embedding_dimension,
    )
