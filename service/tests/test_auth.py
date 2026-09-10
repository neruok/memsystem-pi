import time
from uuid import uuid4

import pytest
from mcp.server.auth.middleware import auth_context
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.mcpserver.exceptions import ToolError

from memsystem.auth import EnvironmentTokenVerifier, authenticated_user_id


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def configured_token(monkeypatch: pytest.MonkeyPatch):
    user_id = uuid4()
    monkeypatch.setenv("MEMSYSTEM_API_TOKEN", "secret")
    monkeypatch.setenv("MEMSYSTEM_API_TOKEN_USER_ID", str(user_id))
    monkeypatch.setenv("MEMSYSTEM_API_TOKEN_EXPIRES_AT", str(int(time.time()) + 60))
    return user_id


@pytest.mark.anyio
async def test_environment_token_verifier_accepts_configured_credential(configured_token):
    token = await EnvironmentTokenVerifier().verify_token("secret")

    assert token is not None
    assert token.subject == str(configured_token)
    assert token.scopes == ["memory"]


@pytest.mark.anyio
@pytest.mark.parametrize("candidate", ["wrong", ""])
async def test_environment_token_verifier_rejects_other_credentials(configured_token, candidate):
    assert await EnvironmentTokenVerifier().verify_token(candidate) is None


@pytest.mark.anyio
async def test_environment_token_verifier_rejects_expired_credential(
    monkeypatch: pytest.MonkeyPatch, configured_token
):
    monkeypatch.setenv("MEMSYSTEM_API_TOKEN_EXPIRES_AT", str(int(time.time()) - 1))

    assert await EnvironmentTokenVerifier().verify_token("secret") is None


def test_authenticated_user_id_requires_verified_request():
    with pytest.raises(ToolError, match="Authentication required"):
        authenticated_user_id()


def test_authenticated_user_id_returns_token_subject():
    user_id = uuid4()
    access_token = AccessToken(
        token="secret",
        client_id="test",
        scopes=["memory"],
        subject=str(user_id),
    )
    marker = auth_context.auth_context_var.set(AuthenticatedUser(access_token))
    try:
        assert authenticated_user_id() == user_id
    finally:
        auth_context.auth_context_var.reset(marker)
