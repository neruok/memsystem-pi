"""Authorized typed document links and backlinks."""

import math
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from psycopg import Connection

from memsystem.authorization import compartment_permissions
from memsystem.context_token import ContextTokenClaims
from memsystem.documents import DocumentNotFound
from memsystem.idempotency import run_idempotent
from memsystem.navigation import _decode_cursor, _encode_cursor

LinkType = Literal["references", "supports", "contradicts", "supersedes", "related"]
_LINK_TYPES = {"references", "supports", "contradicts", "supersedes", "related"}


@dataclass(frozen=True)
class DocumentLink:
    source_document_id: UUID
    target_document_id: UUID
    link_type: str
    weight: float


@dataclass(frozen=True)
class BacklinkPage:
    items: list[DocumentLink]
    next_cursor: str | None


def set_document_link(
    connection: Connection,
    context: ContextTokenClaims,
    source_document_id: UUID | str,
    target_document_id: UUID | str,
    link_type: LinkType,
    *,
    idempotency_key: str,
    weight: float = 1.0,
) -> DocumentLink:
    """Create or update one authorized typed link once."""
    source_document_id = UUID(str(source_document_id))
    target_document_id = UUID(str(target_document_id))
    _validate_link(source_document_id, target_document_id, link_type, weight)
    readable, writable = _document_compartments(connection, context)
    _lock_link_documents(
        connection,
        context.tenant_id,
        (source_document_id, writable),
        (target_document_id, readable),
    )

    def mutate() -> dict[str, object]:
        connection.execute(
            """INSERT INTO document_links
               (tenant_id, source_document_id, target_document_id, type, weight)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (tenant_id, source_document_id, target_document_id, type)
               DO UPDATE SET weight = EXCLUDED.weight""",
            (
                context.tenant_id,
                source_document_id,
                target_document_id,
                link_type,
                weight,
            ),
        )
        return {
            "sourceDocumentId": str(source_document_id),
            "targetDocumentId": str(target_document_id),
            "type": link_type,
            "weight": weight,
        }

    response = run_idempotent(
        connection,
        context,
        idempotency_key,
        "document.link.set",
        {
            "sourceDocumentId": str(source_document_id),
            "targetDocumentId": str(target_document_id),
            "type": link_type,
            "weight": weight,
        },
        mutate,
    )
    return _link_result(response)


def delete_document_link(
    connection: Connection,
    context: ContextTokenClaims,
    source_document_id: UUID | str,
    target_document_id: UUID | str,
    link_type: LinkType,
    *,
    idempotency_key: str,
) -> bool:
    """Delete one link from an authorized live source once."""
    source_document_id = UUID(str(source_document_id))
    target_document_id = UUID(str(target_document_id))
    _validate_link(source_document_id, target_document_id, link_type, 1.0)
    _, writable = _document_compartments(connection, context)
    _require_live_document(
        connection, context.tenant_id, source_document_id, writable, lock=True
    )

    def mutate() -> dict[str, object]:
        removed = connection.execute(
            """DELETE FROM document_links
               WHERE tenant_id = %s AND source_document_id = %s
                 AND target_document_id = %s AND type = %s
               RETURNING 1""",
            (context.tenant_id, source_document_id, target_document_id, link_type),
        ).fetchone()
        return {"removed": removed is not None}

    response = run_idempotent(
        connection,
        context,
        idempotency_key,
        "document.link.delete",
        {
            "sourceDocumentId": str(source_document_id),
            "targetDocumentId": str(target_document_id),
            "type": link_type,
        },
        mutate,
    )
    removed = response.get("removed")
    if type(removed) is not bool:
        raise RuntimeError("Stored link mutation response is invalid")
    return removed


def read_backlinks(
    connection: Connection,
    context: ContextTokenClaims,
    target_document_id: UUID | str,
    *,
    limit: int = 50,
) -> list[DocumentLink]:
    """Return the first bounded backlink page for compatibility."""
    return read_backlinks_page(
        connection, context, target_document_id, limit=limit
    ).items


def read_backlinks_page(
    connection: Connection,
    context: ContextTokenClaims,
    target_document_id: UUID | str,
    *,
    limit: int = 50,
    cursor: str | None = None,
) -> BacklinkPage:
    """Return one bounded backlink page from readable live sources."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be from 1 through 100")
    target_document_id = UUID(str(target_document_id))
    readable, _ = _document_compartments(connection, context)
    _require_live_document(connection, context.tenant_id, target_document_id, readable)
    after_source, after_type = UUID(int=0), ""
    if cursor is not None:
        value = _decode_cursor(cursor, "backlinks", target_document_id)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not isinstance(value[1], str)
        ):
            raise ValueError("invalid cursor")
        try:
            after_source, after_type = UUID(value[0]), value[1]
        except ValueError:
            raise ValueError("invalid cursor") from None
    rows = connection.execute(
        """SELECT l.source_document_id, l.target_document_id, l.type::text, l.weight
           FROM document_links l
           JOIN documents source
             ON (source.tenant_id, source.id) = (l.tenant_id, l.source_document_id)
           WHERE l.tenant_id = %s AND l.target_document_id = %s
             AND source.compartment_id = ANY(%s) AND source.deleted_at IS NULL
             AND (source.expires_at IS NULL OR source.expires_at > now())
             AND (l.source_document_id, l.type::text) > (%s, %s)
           ORDER BY l.source_document_id, l.type::text
           LIMIT %s""",
        (
            context.tenant_id, target_document_id, readable,
            after_source, after_type, limit + 1,
        ),
    ).fetchall()
    items = [DocumentLink(*row) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = _encode_cursor(
            "backlinks", target_document_id,
            [str(last.source_document_id), last.link_type],
        )
    return BacklinkPage(items, next_cursor)


def _document_compartments(
    connection: Connection,
    context: ContextTokenClaims,
) -> tuple[list[UUID], list[UUID]]:
    permissions = compartment_permissions(connection, context)
    return (
        [identifier for identifier, permission in permissions.items() if permission.can_read],
        [identifier for identifier, permission in permissions.items() if permission.can_write],
    )


def _lock_link_documents(
    connection: Connection,
    tenant_id: UUID,
    *requirements: tuple[UUID, list[UUID]],
) -> None:
    for document_id, compartment_ids in sorted(requirements, key=lambda item: item[0].int):
        _require_live_document(
            connection, tenant_id, document_id, compartment_ids, lock=True
        )


def _require_live_document(
    connection: Connection,
    tenant_id: UUID,
    document_id: UUID,
    compartment_ids: list[UUID],
    *,
    lock: bool = False,
) -> None:
    lock_clause = " FOR SHARE" if lock else ""
    if not compartment_ids or connection.execute(
        """SELECT 1 FROM documents
           WHERE tenant_id = %s AND id = %s AND compartment_id = ANY(%s)
             AND deleted_at IS NULL AND (expires_at IS NULL OR expires_at > now())"""
        + lock_clause,
        (tenant_id, document_id, compartment_ids),
    ).fetchone() is None:
        raise DocumentNotFound("Document not found")


def _validate_link(
    source_document_id: UUID,
    target_document_id: UUID,
    link_type: str,
    weight: float,
) -> None:
    if source_document_id == target_document_id:
        raise ValueError("document cannot link to itself")
    if link_type not in _LINK_TYPES:
        raise ValueError("invalid link type")
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be finite and from 0 through 1")


def _link_result(response: dict[str, object]) -> DocumentLink:
    source = response.get("sourceDocumentId")
    target = response.get("targetDocumentId")
    link_type = response.get("type")
    weight = response.get("weight")
    if (
        not isinstance(source, str)
        or not isinstance(target, str)
        or not isinstance(link_type, str)
        or type(weight) not in (int, float)
    ):
        raise RuntimeError("Stored link mutation response is invalid")
    return DocumentLink(UUID(source), UUID(target), link_type, float(weight))
