"""Bounded collection navigation and revision history."""

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from psycopg import Connection

from memsystem.context_token import ContextTokenClaims
from memsystem.documents import DocumentNotFound, read_document


@dataclass(frozen=True)
class ChildDocument:
    document_id: UUID
    kind: str
    slug: str
    revision: int
    title: str
    updated_at: datetime


@dataclass(frozen=True)
class RevisionSummary:
    revision: int
    title: str
    author_type: str
    created_at: datetime


@dataclass(frozen=True)
class CollectionPage:
    items: list[ChildDocument]
    next_cursor: str | None


@dataclass(frozen=True)
class ContentPage:
    document_id: UUID
    revision: int
    title: str
    markdown: str
    next_cursor: str | None


@dataclass(frozen=True)
class RevisionPage:
    items: list[RevisionSummary]
    next_cursor: str | None


def read_document_content(
    connection: Connection,
    context: ContextTokenClaims,
    document_id: UUID | str,
    *,
    revision: int | None = None,
    cursor: str | None = None,
    max_bytes: int = 32_768,
) -> ContentPage:
    """Read one byte-bounded page from an authorized document revision."""
    if not 4 <= max_bytes <= 65_536:
        raise ValueError("max_bytes must be from 4 through 65536")
    document = read_document(connection, context, document_id, revision)
    offset = 0
    if cursor is not None:
        value = _decode_cursor(cursor, "content", document.id)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or type(value[0]) is not int
            or type(value[1]) is not int
            or value[0] != document.revision
            or not 0 <= value[1] <= len(document.markdown)
        ):
            raise ValueError("invalid cursor")
        offset = value[1]

    low, high = offset, min(offset + max_bytes, len(document.markdown))
    while low < high:
        middle = (low + high + 1) // 2
        if len(document.markdown[offset:middle].encode()) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    end = low
    next_cursor = None
    if end < len(document.markdown):
        next_cursor = _encode_cursor("content", document.id, [document.revision, end])
    return ContentPage(
        document.id, document.revision, document.title,
        document.markdown[offset:end], next_cursor,
    )


def read_collection_children(
    connection: Connection,
    context: ContextTokenClaims,
    collection_id: UUID | str,
    *,
    limit: int = 50,
    cursor: str | None = None,
) -> CollectionPage:
    """Read one page of active children from an authorized collection."""
    _validate_limit(limit)
    collection = read_document(connection, context, collection_id)
    if collection.kind != "collection":
        raise DocumentNotFound("Collection not found")

    last_slug, last_id = "", UUID(int=0)
    if cursor is not None:
        value = _decode_cursor(cursor, "children", collection.id)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not isinstance(value[1], str)
        ):
            raise ValueError("invalid cursor")
        try:
            last_slug, last_id = value[0], UUID(value[1])
        except ValueError:
            raise ValueError("invalid cursor") from None

    rows = connection.execute(
        """SELECT d.id, d.kind::text, d.slug, r.revision, r.title, d.updated_at
           FROM documents d
           JOIN document_revisions r
             ON (r.tenant_id, r.document_id, r.id) =
                (d.tenant_id, d.id, d.current_revision_id)
           WHERE d.tenant_id = %s AND d.compartment_id = %s AND d.parent_id = %s
             AND (d.slug, d.id) > (%s, %s)
             AND d.deleted_at IS NULL AND (d.expires_at IS NULL OR d.expires_at > now())
           ORDER BY d.slug, d.id
           LIMIT %s""",
        (
            context.tenant_id,
            collection.compartment_id,
            collection.id,
            last_slug,
            last_id,
            limit + 1,
        ),
    ).fetchall()
    items = [ChildDocument(*row) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = _encode_cursor(
            "children", collection.id, [last.slug, str(last.document_id)]
        )
    return CollectionPage(items, next_cursor)


def read_revision_history(
    connection: Connection,
    context: ContextTokenClaims,
    document_id: UUID | str,
    *,
    limit: int = 50,
    cursor: str | None = None,
) -> RevisionPage:
    """Read one newest-first page of immutable revision metadata."""
    _validate_limit(limit)
    document = read_document(connection, context, document_id)
    before_revision = document.revision + 1
    if cursor is not None:
        value = _decode_cursor(cursor, "history", document.id)
        if type(value) is not int or value < 1:
            raise ValueError("invalid cursor")
        before_revision = value

    rows = connection.execute(
        """SELECT revision, title, created_by_type::text, created_at
           FROM document_revisions
           WHERE tenant_id = %s AND document_id = %s AND revision < %s
           ORDER BY revision DESC
           LIMIT %s""",
        (context.tenant_id, document.id, before_revision, limit + 1),
    ).fetchall()
    items = [RevisionSummary(*row) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = _encode_cursor("history", document.id, items[-1].revision)
    return RevisionPage(items, next_cursor)


def _encode_cursor(kind: str, document_id: UUID, value: object) -> str:
    payload = json.dumps(
        {"d": str(document_id), "k": kind, "v": value},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode()


def _decode_cursor(cursor: str, kind: str, document_id: UUID) -> object:
    if not isinstance(cursor, str) or len(cursor) > 1024:
        raise ValueError("invalid cursor")
    try:
        encoded = cursor.encode("ascii")
        payload = json.loads(
            base64.b64decode(
                encoded + b"=" * (-len(encoded) % 4),
                altchars=b"-_",
                validate=True,
            )
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != {"d", "k", "v"}
            or payload["d"] != str(document_id)
            or payload["k"] != kind
        ):
            raise ValueError
        return payload["v"]
    except (UnicodeError, ValueError, json.JSONDecodeError, TypeError):
        raise ValueError("invalid cursor") from None


def _validate_limit(limit: int) -> None:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be from 1 through 100")
