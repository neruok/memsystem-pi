from uuid import uuid4

import pytest

from memsystem.context_token import ContextTokenCodec, ContextTokenError
from memsystem.request_context import ResolvedContext


@pytest.fixture
def resolved_context():
    return ResolvedContext(
        tenant_id=uuid4(),
        user_id=uuid4(),
        agent_id=uuid4(),
        agent_run_id=uuid4(),
        workspace_id=uuid4(),
        project_id=uuid4(),
    )


def test_context_token_round_trip(resolved_context):
    codec = ContextTokenCodec("x" * 32, ttl_seconds=300)

    claims = codec.verify(
        codec.issue(resolved_context, now=1_000),
        resolved_context.user_id,
        now=1_001,
    )

    assert claims.tenant_id == resolved_context.tenant_id
    assert claims.user_id == resolved_context.user_id
    assert claims.agent_id == resolved_context.agent_id
    assert claims.agent_run_id == resolved_context.agent_run_id
    assert claims.workspace_id == resolved_context.workspace_id
    assert claims.project_id == resolved_context.project_id
    assert claims.expires_at == 1_300


def test_context_token_rejects_tampering(resolved_context):
    codec = ContextTokenCodec("x" * 32)
    version, payload, signature = codec.issue(resolved_context, now=1_000).split(".")
    replacement = "A" if payload[0] != "A" else "B"
    tampered = f"{version}.{replacement}{payload[1:]}.{signature}"

    with pytest.raises(ContextTokenError):
        codec.verify(tampered, resolved_context.user_id, now=1_001)


def test_context_token_rejects_expiry(resolved_context):
    codec = ContextTokenCodec("x" * 32, ttl_seconds=60)
    token = codec.issue(resolved_context, now=1_000)

    with pytest.raises(ContextTokenError):
        codec.verify(token, resolved_context.user_id, now=1_060)


def test_context_token_is_bound_to_authenticated_user(resolved_context):
    codec = ContextTokenCodec("x" * 32)
    token = codec.issue(resolved_context, now=1_000)

    with pytest.raises(ContextTokenError):
        codec.verify(token, uuid4(), now=1_001)


@pytest.mark.parametrize("ttl", [0, 901])
def test_context_token_rejects_non_short_lived_ttl(ttl):
    with pytest.raises(ValueError, match="TTL"):
        ContextTokenCodec("x" * 32, ttl_seconds=ttl)


def test_context_token_requires_long_key():
    with pytest.raises(ValueError, match="32 bytes"):
        ContextTokenCodec("short")
