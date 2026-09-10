"""Short-lived signed tokens for resolved request context."""

import base64
import hmac
import json
import os
import time
from dataclasses import dataclass
from uuid import UUID

from memsystem.request_context import ResolvedContext

_MAX_TTL_SECONDS = 900


class ContextTokenError(ValueError):
    """Context token is missing, invalid, or expired."""


@dataclass(frozen=True)
class ContextTokenClaims:
    tenant_id: UUID
    user_id: UUID
    agent_id: UUID | None
    agent_run_id: UUID | None
    workspace_id: UUID | None
    project_id: UUID | None
    issued_at: int
    expires_at: int


class ContextTokenCodec:
    def __init__(self, key: str, ttl_seconds: int = 300):
        if len(key.encode()) < 32:
            raise ValueError("context token key must contain at least 32 bytes")
        if not 1 <= ttl_seconds <= _MAX_TTL_SECONDS:
            raise ValueError("context token TTL must be from 1 through 900 seconds")
        self._key = key.encode()
        self._ttl_seconds = ttl_seconds

    @classmethod
    def from_environment(cls) -> "ContextTokenCodec":
        key = os.getenv("MEMSYSTEM_CONTEXT_TOKEN_KEY")
        if key is None:
            raise ValueError("MEMSYSTEM_CONTEXT_TOKEN_KEY is required")
        try:
            ttl = int(os.getenv("MEMSYSTEM_CONTEXT_TOKEN_TTL", "300"))
        except ValueError as error:
            raise ValueError("MEMSYSTEM_CONTEXT_TOKEN_TTL must be an integer") from error
        return cls(key, ttl)

    def issue(self, context: ResolvedContext, *, now: int | None = None) -> str:
        issued_at = int(time.time()) if now is None else now
        payload = json.dumps(
            {
                "agent_id": str(context.agent_id) if context.agent_id else None,
                "agent_run_id": str(context.agent_run_id) if context.agent_run_id else None,
                "aud": "memsystem-mcp",
                "exp": issued_at + self._ttl_seconds,
                "iat": issued_at,
                "iss": "memsystem",
                "project_id": str(context.project_id) if context.project_id else None,
                "tenant_id": str(context.tenant_id),
                "user_id": str(context.user_id),
                "workspace_id": str(context.workspace_id) if context.workspace_id else None,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        encoded = _encode(payload)
        signature = _encode(hmac.digest(self._key, b"v1." + encoded, "sha256"))
        return f"v1.{encoded.decode()}.{signature.decode()}"

    def verify(
        self,
        token: str,
        authenticated_user_id: UUID | str,
        *,
        now: int | None = None,
    ) -> ContextTokenClaims:
        if len(token) > 4096:
            raise ContextTokenError("Context token is invalid")
        try:
            version, payload_text, signature_text = token.split(".")
            payload_encoded = payload_text.encode("ascii")
            signature = _decode(signature_text)
        except (ValueError, UnicodeEncodeError):
            raise ContextTokenError("Context token is invalid") from None
        expected = hmac.digest(self._key, b"v1." + payload_encoded, "sha256")
        if version != "v1" or not hmac.compare_digest(signature, expected):
            raise ContextTokenError("Context token is invalid")

        try:
            payload = json.loads(_decode(payload_text))
            if set(payload) != {
                "agent_id", "agent_run_id", "aud", "exp", "iat", "iss", "project_id",
                "tenant_id", "user_id", "workspace_id",
            }:
                raise ValueError
            if payload["iss"] != "memsystem" or payload["aud"] != "memsystem-mcp":
                raise ValueError
            issued_at, expires_at = payload["iat"], payload["exp"]
            if type(issued_at) is not int or type(expires_at) is not int:
                raise ValueError
            user_id = UUID(payload["user_id"])
            claims = ContextTokenClaims(
                tenant_id=UUID(payload["tenant_id"]),
                user_id=user_id,
                agent_id=_optional_uuid(payload["agent_id"]),
                agent_run_id=_optional_uuid(payload["agent_run_id"]),
                workspace_id=_optional_uuid(payload["workspace_id"]),
                project_id=_optional_uuid(payload["project_id"]),
                issued_at=issued_at,
                expires_at=expires_at,
            )
            if (claims.agent_id is None) != (claims.agent_run_id is None):
                raise ValueError
            if claims.project_id is not None and claims.workspace_id is None:
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise ContextTokenError("Context token is invalid") from None

        current_time = int(time.time()) if now is None else now
        try:
            identity_matches = user_id == UUID(str(authenticated_user_id))
        except ValueError:
            raise ContextTokenError("Context token is invalid") from None
        if (
            not identity_matches
            or issued_at > current_time
            or expires_at <= current_time
            or expires_at <= issued_at
            or expires_at - issued_at > _MAX_TTL_SECONDS
        ):
            raise ContextTokenError("Context token is invalid")
        return claims


def _encode(value: bytes) -> bytes:
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _decode(value: str) -> bytes:
    encoded = value.encode("ascii")
    return base64.b64decode(encoded + b"=" * (-len(encoded) % 4), altchars=b"-_", validate=True)


def _optional_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError
    return UUID(value)
