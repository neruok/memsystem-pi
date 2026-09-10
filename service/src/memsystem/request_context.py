"""Resolve trusted request context from authenticated identity and untrusted hints."""

from dataclasses import dataclass
from uuid import UUID

from psycopg import Connection

from memsystem.database import tenant_transaction


class ContextResolutionError(ValueError):
    """Requested context is unavailable to the authenticated user."""


@dataclass(frozen=True)
class ResolvedContext:
    tenant_id: UUID
    user_id: UUID
    agent_id: UUID | None = None
    agent_run_id: UUID | None = None
    agent_can_write: bool = False
    workspace_id: UUID | None = None
    workspace_can_read: bool = False
    workspace_can_write: bool = False
    project_id: UUID | None = None
    project_can_write: bool = False


def resolve_context(
    connection: Connection,
    user_id: UUID | str,
    tenant_id: UUID | str,
    *,
    agent_key: str | None = None,
    session_key: str | None = None,
    workspace_key: str | None = None,
    project_key: str | None = None,
) -> ResolvedContext:
    """Resolve canonical IDs only after database-backed authorization checks."""
    user_id, tenant_id = UUID(str(user_id)), UUID(str(tenant_id))
    if (agent_key is None) != (session_key is None):
        raise ValueError("agent_key and session_key must be supplied together")
    if project_key is not None and workspace_key is None:
        raise ValueError("project_key requires workspace_key")

    with tenant_transaction(connection, tenant_id):
        membership = connection.execute(
            """SELECT 1 FROM tenant_memberships
               WHERE tenant_id = %s AND user_id = %s AND state = 'active'""",
            (tenant_id, user_id),
        ).fetchone()
        if membership is None:
            raise ContextResolutionError("Context could not be resolved")

        agent_id = agent_run_id = None
        agent_can_write = False
        if agent_key is not None:
            agent = connection.execute(
                """SELECT a.id, r.id, bool_or(d.can_write)
                   FROM agents a
                   JOIN agent_runs r
                     ON (r.tenant_id, r.agent_id) = (a.tenant_id, a.id)
                   JOIN agent_delegations d
                     ON (d.tenant_id, d.agent_id) = (a.tenant_id, a.id)
                   WHERE a.tenant_id = %s AND a.key = %s AND a.archived_at IS NULL
                     AND r.session_key = %s AND r.state = 'active' AND r.expires_at > now()
                     AND d.user_id = %s AND d.can_read
                     AND d.revoked_at IS NULL AND d.expires_at > now()
                   GROUP BY a.id, r.id""",
                (tenant_id, agent_key, session_key, user_id),
            ).fetchone()
            if agent is None:
                raise ContextResolutionError("Context could not be resolved")
            agent_id, agent_run_id, agent_can_write = agent

        workspace_id = None
        workspace_can_read = workspace_can_write = False
        if workspace_key is not None:
            workspace = connection.execute(
                """SELECT w.id, COALESCE(m.can_read, false), COALESCE(m.can_write, false)
                   FROM workspaces w
                   LEFT JOIN workspace_memberships m
                     ON (m.tenant_id, m.workspace_id, m.user_id) = (w.tenant_id, w.id, %s)
                   WHERE w.tenant_id = %s AND w.key = %s AND w.archived_at IS NULL""",
                (user_id, tenant_id, workspace_key),
            ).fetchone()
            if workspace is None:
                raise ContextResolutionError("Context could not be resolved")
            workspace_id, workspace_can_read, workspace_can_write = workspace
            if project_key is None and not workspace_can_read:
                raise ContextResolutionError("Context could not be resolved")

        project_id = None
        project_can_write = False
        if project_key is not None:
            project = connection.execute(
                """SELECT p.id, m.can_write
                   FROM projects p
                   JOIN project_memberships m
                     ON (m.tenant_id, m.project_id) = (p.tenant_id, p.id)
                   WHERE p.tenant_id = %s AND p.workspace_id = %s AND p.key = %s
                     AND p.archived_at IS NULL AND m.user_id = %s AND m.can_read""",
                (tenant_id, workspace_id, project_key, user_id),
            ).fetchone()
            if project is None:
                raise ContextResolutionError("Context could not be resolved")
            project_id, project_can_write = project

    return ResolvedContext(
        tenant_id=tenant_id,
        user_id=user_id,
        agent_id=agent_id,
        agent_run_id=agent_run_id,
        agent_can_write=agent_can_write,
        workspace_id=workspace_id,
        workspace_can_read=workspace_can_read,
        workspace_can_write=workspace_can_write,
        project_id=project_id,
        project_can_write=project_can_write,
    )
