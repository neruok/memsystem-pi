"""Bearer authentication for the MCP HTTP transport."""

import hmac
import os
import time
from uuid import UUID

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ToolError

_SCOPE = "memory"


class EnvironmentTokenVerifier:
    """Verify one operator-managed bearer credential from the environment."""

    async def verify_token(self, token: str) -> AccessToken | None:
        expected = os.getenv("MEMSYSTEM_API_TOKEN")
        subject = os.getenv("MEMSYSTEM_API_TOKEN_USER_ID")
        expires = os.getenv("MEMSYSTEM_API_TOKEN_EXPIRES_AT")
        if not expected or not subject or not expires or not hmac.compare_digest(token, expected):
            return None

        try:
            user_id = UUID(subject)
            expires_at = int(expires)
        except ValueError:
            return None
        if expires_at <= int(time.time()):
            return None

        return AccessToken(
            token=token,
            client_id="memsystem-pi",
            scopes=[_SCOPE],
            expires_at=expires_at,
            subject=str(user_id),
            claims={"iss": os.getenv("MEMSYSTEM_ISSUER_URL", "http://127.0.0.1:8000")},
        )


def auth_settings() -> AuthSettings:
    """Build resource-server settings without enabling token issuance."""
    return AuthSettings(
        issuer_url=os.getenv("MEMSYSTEM_ISSUER_URL", "http://127.0.0.1:8000"),
        resource_server_url=os.getenv("MEMSYSTEM_RESOURCE_URL", "http://127.0.0.1:8000/mcp"),
        required_scopes=[_SCOPE],
    )


def authenticated_user_id() -> UUID:
    """Return the verified user identity for the active HTTP request."""
    token = get_access_token()
    if token is None or token.subject is None:
        raise ToolError("Authentication required")
    try:
        return UUID(token.subject)
    except ValueError as error:
        raise ToolError("Authenticated subject is invalid") from error
