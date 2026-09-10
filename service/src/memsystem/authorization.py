"""Fixed compartment authorization rules."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from psycopg import Connection

from memsystem.context_token import ContextTokenClaims


class AuthorizationError(PermissionError):
    """Requested compartment action is not authorized."""


@dataclass(frozen=True)
class CompartmentPermission:
    compartment_id: UUID
    scope_type: str
    can_read: bool
    can_write: bool


COMPARTMENT_PERMISSIONS_SQL = """WITH active_member AS (
       SELECT 1 FROM tenant_memberships
       WHERE tenant_id = %(tenant_id)s AND user_id = %(user_id)s
         AND state = 'active'
   ),
   delegation AS (
       SELECT bool_or(d.can_write) AS can_write
       FROM agent_delegations d, active_member
       WHERE %(agent_id)s::uuid IS NOT NULL
         AND d.tenant_id = %(tenant_id)s AND d.agent_id = %(agent_id)s
         AND d.user_id = %(user_id)s AND d.can_read
         AND d.revoked_at IS NULL AND d.expires_at > now()
       HAVING count(*) > 0
   ),
   active_agent AS (
       SELECT 1
       FROM agents a
       JOIN agent_runs r
         ON (r.tenant_id, r.agent_id) = (a.tenant_id, a.id)
       JOIN active_member ON true
       WHERE a.tenant_id = %(tenant_id)s AND a.id = %(agent_id)s
         AND a.archived_at IS NULL AND r.id = %(agent_run_id)s
         AND r.state = 'active' AND r.expires_at > now()
   ),
   active_principal AS (
       SELECT 1 FROM active_member WHERE %(agent_id)s::uuid IS NULL
       UNION ALL
       SELECT 1 FROM active_member, active_agent, delegation
   ),
   scopes(scope_type, scope_id, can_read, can_write) AS (
       SELECT 'global'::compartment_scope, NULL::uuid, true, false
       FROM active_principal
       UNION ALL
       SELECT 'user', %(user_id)s, true, true
       FROM active_principal
       WHERE %(agent_id)s::uuid IS NULL
       UNION ALL
       SELECT 'user', %(user_id)s, true, d.can_write
       FROM delegation d, active_principal
       UNION ALL
       SELECT 'agent', %(agent_id)s, true, true
       FROM active_agent, delegation
       UNION ALL
       SELECT 'workspace', w.id, m.can_read, m.can_write
       FROM workspace_memberships m
       JOIN workspaces w
         ON (w.tenant_id, w.id) = (m.tenant_id, m.workspace_id)
       JOIN active_principal ON true
       WHERE w.tenant_id = %(tenant_id)s AND w.id = %(workspace_id)s
         AND w.archived_at IS NULL AND m.user_id = %(user_id)s AND m.can_read
       UNION ALL
       SELECT 'project', p.id, m.can_read, m.can_write
       FROM project_memberships m
       JOIN projects p
         ON (p.tenant_id, p.id) = (m.tenant_id, m.project_id)
       JOIN workspaces w
         ON (w.tenant_id, w.id) = (p.tenant_id, p.workspace_id)
       JOIN active_principal ON true
       WHERE p.tenant_id = %(tenant_id)s AND p.id = %(project_id)s
         AND p.workspace_id = %(workspace_id)s AND p.archived_at IS NULL
         AND w.archived_at IS NULL
         AND m.user_id = %(user_id)s AND m.can_read
   )
   SELECT c.id, c.scope_type::text, s.can_read, s.can_write
   FROM scopes s
   JOIN compartments c
     ON c.tenant_id = %(tenant_id)s AND c.scope_type = s.scope_type
    AND c.scope_id IS NOT DISTINCT FROM s.scope_id"""


def permission_parameters(context: ContextTokenClaims) -> dict[str, UUID | None]:
    return {
        "tenant_id": context.tenant_id,
        "user_id": context.user_id,
        "agent_id": context.agent_id,
        "agent_run_id": context.agent_run_id,
        "workspace_id": context.workspace_id,
        "project_id": context.project_id,
    }


def compartment_permissions(
    connection: Connection,
    context: ContextTokenClaims,
) -> dict[UUID, CompartmentPermission]:
    """Return current permissions. Caller must own the tenant transaction."""
    rows = connection.execute(
        COMPARTMENT_PERMISSIONS_SQL, permission_parameters(context)
    ).fetchall()
    return {
        compartment_id: CompartmentPermission(
            compartment_id, scope_type, can_read, can_write
        )
        for compartment_id, scope_type, can_read, can_write in rows
    }


def require_compartment_access(
    connection: Connection,
    context: ContextTokenClaims,
    compartment_id: UUID | str,
    action: Literal["read", "write"],
) -> CompartmentPermission:
    if action not in ("read", "write"):
        raise ValueError("action must be read or write")
    permission = compartment_permissions(connection, context).get(UUID(str(compartment_id)))
    if permission is None or not getattr(permission, f"can_{action}"):
        raise AuthorizationError("Compartment access denied")
    return permission
