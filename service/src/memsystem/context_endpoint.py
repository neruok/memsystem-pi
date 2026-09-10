"""Authenticated HTTP endpoint for context resolution and token issuance."""

import json
import os
from uuid import UUID

import psycopg
from mcp.server.mcpserver.exceptions import ToolError
from starlette.requests import Request
from starlette.responses import JSONResponse

from memsystem.auth import authenticated_user_id
from memsystem.context_token import ContextTokenCodec
from memsystem.request_context import ContextResolutionError, resolve_context

_FIELDS = {"tenantId", "agentKey", "sessionKey", "workspaceKey", "projectKey"}


async def resolve_context_route(request: Request) -> JSONResponse:
    try:
        user_id = authenticated_user_id()
    except ToolError:
        return JSONResponse(
            {"error": "Authentication required"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        body = await request.body()
        if len(body) > 8192:
            raise ValueError
        data = json.loads(body)
        if not isinstance(data, dict) or set(data) - _FIELDS or not isinstance(data.get("tenantId"), str):
            raise ValueError
        tenant_id = UUID(data["tenantId"])
        hints = {
            name: _optional_string(data.get(field), 500 if field == "sessionKey" else 200)
            for field, name in (
                ("agentKey", "agent_key"),
                ("sessionKey", "session_key"),
                ("workspaceKey", "workspace_key"),
                ("projectKey", "project_key"),
            )
        }
        if (hints["agent_key"] is None) != (hints["session_key"] is None):
            raise ValueError
        if hints["project_key"] is not None and hints["workspace_key"] is None:
            raise ValueError
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return JSONResponse({"error": "Invalid context request"}, status_code=400)

    database_url = os.getenv("MEMSYSTEM_DATABASE_URL")
    try:
        codec = ContextTokenCodec.from_environment()
        if database_url is None:
            raise ValueError
        with psycopg.connect(database_url, autocommit=True) as connection:
            context = resolve_context(connection, user_id, tenant_id, **hints)
    except ContextResolutionError:
        return JSONResponse({"error": "Context could not be resolved"}, status_code=403)
    except (psycopg.Error, ValueError):
        return JSONResponse({"error": "Context service is unavailable"}, status_code=503)

    token = codec.issue(context)
    return JSONResponse(
        {
            "contextToken": token,
            "tenantId": str(context.tenant_id),
            "userId": str(context.user_id),
            "agentId": str(context.agent_id) if context.agent_id else None,
            "agentCanWrite": context.agent_can_write,
            "workspaceId": str(context.workspace_id) if context.workspace_id else None,
            "workspaceCanRead": context.workspace_can_read,
            "workspaceCanWrite": context.workspace_can_write,
            "projectId": str(context.project_id) if context.project_id else None,
            "projectCanWrite": context.project_can_write,
        }
    )


def _optional_string(value: object, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= limit:
        raise ValueError
    return value
