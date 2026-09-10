"""Authorized document storage operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from psycopg import Connection

from memsystem.authorization import compartment_permissions, require_compartment_access
from memsystem.chunking import split_markdown
from memsystem.context_token import ContextTokenClaims
from memsystem.embeddings import ACTIVE_PROFILE
from memsystem.idempotency import run_idempotent
from memsystem.jobs import lock_index_projection

DocumentKind = Literal["collection", "page", "journal"]


class DocumentNotFound(LookupError):
    """Document is absent, inactive, or unauthorized."""


class RevisionConflict(RuntimeError):
    def __init__(self, current_revision: int):
        super().__init__(f"Expected revision does not match current revision {current_revision}")
        self.current_revision = current_revision


@dataclass(frozen=True)
class Document:
    id: UUID
    compartment_id: UUID
    parent_id: UUID | None
    kind: str
    slug: str
    revision: int
    title: str
    markdown: str
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DocumentMutation:
    document_id: UUID
    revision: int


def create_document(
    connection: Connection,
    context: ContextTokenClaims,
    compartment_id: UUID | str,
    *,
    kind: DocumentKind,
    slug: str,
    title: str,
    markdown: str,
    parent_id: UUID | str | None = None,
    expires_at: datetime | None = None,
    idempotency_key: str,
) -> DocumentMutation:
    """Create a document and its first immutable revision once."""
    _validate_content(kind, slug, title, markdown, expires_at)
    compartment_id = UUID(str(compartment_id))
    parent_id = UUID(str(parent_id)) if parent_id is not None else None
    require_compartment_access(connection, context, compartment_id, "write")

    def mutate() -> dict[str, object]:
        if parent_id is not None:
            parent = connection.execute(
                """SELECT 1 FROM documents
                   WHERE tenant_id = %s AND id = %s AND compartment_id = %s
                     AND kind = 'collection' AND deleted_at IS NULL
                     AND (expires_at IS NULL OR expires_at > now())""",
                (context.tenant_id, parent_id, compartment_id),
            ).fetchone()
            if parent is None:
                raise DocumentNotFound("Parent document not found")

        document_id = connection.execute(
            """INSERT INTO documents
               (tenant_id, compartment_id, parent_id, kind, slug, expires_at)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (context.tenant_id, compartment_id, parent_id, kind, slug, expires_at),
        ).fetchone()[0]
        revision_id = connection.execute(
            """INSERT INTO document_revisions
               (tenant_id, document_id, revision, title, markdown, created_by_type, created_by_id)
               VALUES (%s, %s, 1, %s, %s, %s, %s)
               RETURNING id""",
            (context.tenant_id, document_id, title, markdown, *_author(context)),
        ).fetchone()[0]
        _insert_chunks(connection, context.tenant_id, document_id, revision_id, title, markdown)
        connection.execute(
            """UPDATE documents SET current_revision_id = %s
               WHERE tenant_id = %s AND id = %s""",
            (revision_id, context.tenant_id, document_id),
        )
        return {"documentId": str(document_id), "revision": 1}

    response = run_idempotent(
        connection,
        context,
        idempotency_key,
        "document.create",
        {
            "compartmentId": str(compartment_id),
            "expiresAt": expires_at.isoformat() if expires_at else None,
            "kind": kind,
            "markdown": markdown,
            "parentId": str(parent_id) if parent_id else None,
            "slug": slug,
            "title": title,
        },
        mutate,
    )
    return _mutation_result(response)


def read_document(
    connection: Connection,
    context: ContextTokenClaims,
    document_id: UUID | str,
    revision: int | None = None,
) -> Document:
    """Read one live authorized document revision."""
    if revision is not None and revision < 1:
        raise ValueError("revision must be positive")
    readable = [
        compartment_id
        for compartment_id, permission in compartment_permissions(connection, context).items()
        if permission.can_read
    ]
    if not readable:
        raise DocumentNotFound("Document not found")
    return _fetch_document(
        connection, context.tenant_id, UUID(str(document_id)), readable, revision
    )


def update_document(
    connection: Connection,
    context: ContextTokenClaims,
    document_id: UUID | str,
    *,
    expected_revision: int,
    title: str,
    markdown: str,
    idempotency_key: str,
) -> DocumentMutation:
    """Lock and replace current content once."""
    _validate_revision(expected_revision)
    _validate_title_markdown(title, markdown)
    document_id = UUID(str(document_id))
    writable = _require_writable_document(connection, context, document_id)

    def mutate() -> dict[str, object]:
        row = _lock_document(connection, context.tenant_id, document_id, writable)
        if row is None:
            raise DocumentNotFound("Document not found")
        _, current_revision = row
        if current_revision != expected_revision:
            raise RevisionConflict(current_revision)

        next_revision = current_revision + 1
        revision_id = connection.execute(
            """INSERT INTO document_revisions
               (tenant_id, document_id, revision, title, markdown, created_by_type, created_by_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                context.tenant_id, document_id, next_revision, title, markdown,
                *_author(context),
            ),
        ).fetchone()[0]
        _insert_chunks(connection, context.tenant_id, document_id, revision_id, title, markdown)
        _enqueue_index_removals(connection, context.tenant_id, document_id)
        connection.execute(
            """UPDATE documents
               SET current_revision_id = %s, updated_at = now()
               WHERE tenant_id = %s AND id = %s""",
            (revision_id, context.tenant_id, document_id),
        )
        return {"documentId": str(document_id), "revision": next_revision}

    response = run_idempotent(
        connection,
        context,
        idempotency_key,
        "document.update",
        {
            "documentId": str(document_id),
            "expectedRevision": expected_revision,
            "markdown": markdown,
            "title": title,
        },
        mutate,
    )
    return _mutation_result(response)


def soft_delete_document(
    connection: Connection,
    context: ContextTokenClaims,
    document_id: UUID | str,
    *,
    expected_revision: int,
    idempotency_key: str,
) -> DocumentMutation:
    """Lock and soft-delete a live document once."""
    _validate_revision(expected_revision)
    document_id = UUID(str(document_id))
    writable = _require_writable_document(connection, context, document_id)

    def mutate() -> dict[str, object]:
        row = _lock_document(connection, context.tenant_id, document_id, writable)
        if row is None:
            raise DocumentNotFound("Document not found")
        _, current_revision = row
        if current_revision != expected_revision:
            raise RevisionConflict(current_revision)
        lock_index_projection(connection, context.tenant_id, shared=True)
        _enqueue_index_removals(connection, context.tenant_id, document_id)
        connection.execute(
            """UPDATE documents SET deleted_at = now(), updated_at = now()
               WHERE tenant_id = %s AND id = %s""",
            (context.tenant_id, document_id),
        )
        return {"documentId": str(document_id), "revision": current_revision}

    response = run_idempotent(
        connection,
        context,
        idempotency_key,
        "document.delete",
        {"documentId": str(document_id), "expectedRevision": expected_revision},
        mutate,
    )
    return _mutation_result(response)


def _require_writable_document(
    connection: Connection,
    context: ContextTokenClaims,
    document_id: UUID,
) -> list[UUID]:
    writable = [
        compartment_id
        for compartment_id, permission in compartment_permissions(connection, context).items()
        if permission.can_write
    ]
    if not writable or connection.execute(
        """SELECT 1 FROM documents
           WHERE tenant_id = %s AND id = %s AND compartment_id = ANY(%s)""",
        (context.tenant_id, document_id, writable),
    ).fetchone() is None:
        raise DocumentNotFound("Document not found")
    return writable


def _mutation_result(response: dict[str, object]) -> DocumentMutation:
    document_id = response.get("documentId")
    revision = response.get("revision")
    if not isinstance(document_id, str) or type(revision) is not int:
        raise RuntimeError("Stored mutation response is invalid")
    return DocumentMutation(UUID(document_id), revision)


def _fetch_document(
    connection: Connection,
    tenant_id: UUID,
    document_id: UUID,
    compartment_ids: list[UUID],
    revision: int | None = None,
) -> Document:
    revision_filter = (
        "r.id = d.current_revision_id"
        if revision is None
        else "r.revision = %(revision)s"
    )
    row = connection.execute(
        f"""SELECT d.id, d.compartment_id, d.parent_id, d.kind::text, d.slug,
                   r.revision, r.title, r.markdown, d.expires_at, d.created_at, d.updated_at
            FROM documents d
            JOIN document_revisions r
              ON (r.tenant_id, r.document_id) = (d.tenant_id, d.id) AND {revision_filter}
            WHERE d.tenant_id = %(tenant_id)s AND d.id = %(document_id)s
              AND d.compartment_id = ANY(%(compartment_ids)s)
              AND d.deleted_at IS NULL AND (d.expires_at IS NULL OR d.expires_at > now())""",
        {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "compartment_ids": compartment_ids,
            "revision": revision,
        },
    ).fetchone()
    if row is None:
        raise DocumentNotFound("Document not found")
    return Document(*row)


def _lock_document(
    connection: Connection,
    tenant_id: UUID,
    document_id: UUID,
    writable: list[UUID],
) -> tuple[UUID, int] | None:
    if not writable:
        return None
    document = connection.execute(
        """SELECT compartment_id FROM documents
           WHERE tenant_id = %s AND id = %s AND compartment_id = ANY(%s)
             AND deleted_at IS NULL AND (expires_at IS NULL OR expires_at > now())
           FOR UPDATE""",
        (tenant_id, document_id, writable),
    ).fetchone()
    if document is None:
        return None
    revision = connection.execute(
        """SELECT r.revision
           FROM documents d
           JOIN document_revisions r
             ON (r.tenant_id, r.document_id, r.id) =
                (d.tenant_id, d.id, d.current_revision_id)
           WHERE d.tenant_id = %s AND d.id = %s""",
        (tenant_id, document_id),
    ).fetchone()[0]
    return document[0], revision


def _insert_chunks(
    connection: Connection,
    tenant_id: UUID,
    document_id: UUID,
    revision_id: UUID,
    title: str,
    markdown: str,
) -> None:
    lock_index_projection(connection, tenant_id, shared=True)
    with connection.cursor() as cursor:
        cursor.executemany(
            """INSERT INTO chunks
           (tenant_id, document_id, revision_id, heading_path, text, position,
            source_start, source_end, chunk_profile, search_document)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                   setweight(to_tsvector(%s::regconfig, %s), 'A') ||
                   setweight(to_tsvector(%s::regconfig, array_to_string(%s::text[], ' ')), 'B') ||
                   setweight(to_tsvector(%s::regconfig, %s), 'C'))""",
            [
                (
                    tenant_id,
                    document_id,
                    revision_id,
                    list(chunk.heading_path),
                    chunk.text,
                    position,
                    chunk.source_start,
                    chunk.source_end,
                    ACTIVE_PROFILE.chunk_profile,
                    ACTIVE_PROFILE.text_search_config,
                    title,
                    ACTIVE_PROFILE.text_search_config,
                    list(chunk.heading_path),
                    ACTIVE_PROFILE.text_search_config,
                    chunk.text,
                )
                for position, chunk in enumerate(split_markdown(markdown))
            ],
        )
    connection.execute(
        """INSERT INTO jobs
           (tenant_id, document_id, revision_id, chunk_id, vector_id, profile, kind)
           SELECT tenant_id, document_id, revision_id, id, vector_id, %s, 'embed'
           FROM chunks
           WHERE tenant_id = %s AND revision_id = %s""",
        (ACTIVE_PROFILE.name, tenant_id, revision_id),
    )


def _enqueue_index_removals(
    connection: Connection, tenant_id: UUID, document_id: UUID
) -> None:
    connection.execute(
        """INSERT INTO jobs (tenant_id, vector_id, profile, kind)
           SELECT c.tenant_id, c.vector_id, %s, 'index_remove'
           FROM chunks c
           JOIN documents d
             ON (d.tenant_id, d.id, d.current_revision_id) =
                (c.tenant_id, c.document_id, c.revision_id)
           WHERE c.tenant_id = %s AND c.document_id = %s""",
        (ACTIVE_PROFILE.name, tenant_id, document_id),
    )


def _author(context: ContextTokenClaims) -> tuple[str, UUID]:
    if context.agent_id is not None:
        return "agent", context.agent_id
    return "user", context.user_id


def _validate_content(
    kind: str,
    slug: str,
    title: str,
    markdown: str,
    expires_at: datetime | None,
) -> None:
    if kind not in ("collection", "page", "journal"):
        raise ValueError("invalid document kind")
    if not 1 <= len(slug) <= 200:
        raise ValueError("slug must contain from 1 through 200 characters")
    if expires_at is not None and expires_at.utcoffset() is None:
        raise ValueError("expires_at must include a timezone")
    _validate_title_markdown(title, markdown)


def _validate_title_markdown(title: str, markdown: str) -> None:
    if not 1 <= len(title) <= 500:
        raise ValueError("title must contain from 1 through 500 characters")
    if len(markdown.encode()) > 1_048_576:
        raise ValueError("markdown must not exceed 1048576 bytes")


def _validate_revision(revision: int) -> None:
    if revision < 1:
        raise ValueError("expected_revision must be positive")
