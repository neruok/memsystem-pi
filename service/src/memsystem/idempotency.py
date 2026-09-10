"""Atomic idempotency for tenant-scoped mutations."""

import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

from psycopg import Connection
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb

from memsystem.context_token import ContextTokenClaims


class IdempotencyConflict(RuntimeError):
    """An idempotency key was reused for a different request."""


def run_idempotent(
    connection: Connection,
    context: ContextTokenClaims,
    key: str,
    operation: str,
    request: dict[str, Any],
    mutation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run one mutation or replay its committed response."""
    if not 1 <= len(key) <= 200:
        raise ValueError("idempotency key must contain from 1 through 200 characters")
    if not 1 <= len(operation) <= 100:
        raise ValueError("operation must contain from 1 through 100 characters")
    try:
        encoded_request = json.dumps(
            {
                "agentId": str(context.agent_id) if context.agent_id else None,
                "operation": operation,
                "request": request,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("idempotent request must contain JSON values") from error
    request_hash = hashlib.sha256(encoded_request).digest()
    if connection.info.transaction_status is TransactionStatus.IDLE:
        raise RuntimeError("run_idempotent requires an active transaction")
    with connection.transaction():
        return _store_or_replay(
            connection, context, key, operation, request_hash, mutation
        )


def _store_or_replay(
    connection: Connection,
    context: ContextTokenClaims,
    key: str,
    operation: str,
    request_hash: bytes,
    mutation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    inserted = connection.execute(
        """INSERT INTO mutation_idempotency
           (tenant_id, user_id, key, operation, request_hash)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (tenant_id, user_id, key) DO NOTHING
           RETURNING 1""",
        (context.tenant_id, context.user_id, key, operation, request_hash),
    ).fetchone()
    if inserted is None:
        stored = connection.execute(
            """SELECT request_hash, response FROM mutation_idempotency
               WHERE tenant_id = %s AND user_id = %s AND key = %s""",
            (context.tenant_id, context.user_id, key),
        ).fetchone()
        if stored is None or not hmac.compare_digest(bytes(stored[0]), request_hash):
            raise IdempotencyConflict("Idempotency key conflicts with another request")
        if stored[1] is None:
            raise RuntimeError("Idempotent mutation has no response")
        return stored[1]

    response = mutation()
    try:
        encoded_response = json.dumps(
            response, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("idempotent response must contain JSON values") from error
    if len(encoded_response) > 65_536:
        raise ValueError("idempotent response must not exceed 65536 bytes")
    connection.execute(
        """UPDATE mutation_idempotency SET response = %s
           WHERE tenant_id = %s AND user_id = %s AND key = %s""",
        (Jsonb(response), context.tenant_id, context.user_id, key),
    )
    return response
