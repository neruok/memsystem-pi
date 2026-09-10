"""Authorized PostgreSQL full-text retrieval."""

import base64
from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Sequence
from uuid import UUID

from pgvector import Vector
from psycopg import Connection
from psycopg.pq import TransactionStatus

from memsystem.authorization import (
    COMPARTMENT_PERMISSIONS_SQL,
    compartment_permissions,
    permission_parameters,
)
from memsystem.context_token import ContextTokenClaims
from memsystem.embeddings import ACTIVE_PROFILE, normalize_embedding
from memsystem.jobs import lock_index_projection
from memsystem.vector_index import IndexSlot

RRF_K = 60


@dataclass(frozen=True)
class LexicalResult:
    chunk_id: UUID
    document_id: UUID
    revision: int
    title: str
    heading_path: list[str]
    text: str
    source_start: int
    source_end: int
    score: float


@dataclass(frozen=True)
class LexicalPage:
    items: list[LexicalResult]
    next_cursor: str | None


@dataclass(frozen=True)
class VectorResult:
    chunk_id: UUID
    document_id: UUID
    revision: int
    title: str
    heading_path: list[str]
    text: str
    source_start: int
    source_end: int
    score: float


@dataclass(frozen=True)
class HybridResult:
    chunk_id: UUID
    document_id: UUID
    revision: int
    title: str
    heading_path: list[str]
    text: str
    source_start: int
    source_end: int
    lexical_score: float | None
    vector_score: float | None
    fused_score: float


def search_lexical(
    connection: Connection,
    context: ContextTokenClaims,
    query: str,
    *,
    limit: int = 10,
    cursor: str | None = None,
    project_only: bool = False,
    subsystem_keys: list[str] | None = None,
) -> LexicalPage:
    """Rank current authorized chunks with stable keyset pagination."""
    if not isinstance(query, str):
        raise ValueError("query must contain from 1 through 1000 characters")
    query = query.strip()
    if not query or len(query) > 1_000:
        raise ValueError("query must contain from 1 through 1000 characters")
    _validate_limit(limit)
    keys = _validate_filters(context, project_only, subsystem_keys)

    after_score, after_id = None, UUID(int=0)
    if cursor is not None:
        after_score, after_id = _decode_cursor(
            cursor, query, project_only, keys, context
        )

    readable = _readable_compartments(connection, context)
    if not readable:
        return LexicalPage([], None)

    rows = connection.execute(
        """WITH query AS (
               SELECT websearch_to_tsquery(%(text_config)s::regconfig, %(query)s) AS value
           ), ranked AS (
               SELECT c.id AS chunk_id, c.document_id, r.revision, r.title,
                      c.heading_path, c.text, c.source_start, c.source_end,
                      ts_rank_cd(c.search_document, query.value) AS score
               FROM query, chunks c
               JOIN documents d
                 ON (d.tenant_id, d.id, d.current_revision_id) =
                    (c.tenant_id, c.document_id, c.revision_id)
               JOIN document_revisions r
                 ON (r.tenant_id, r.document_id, r.id) =
                    (c.tenant_id, c.document_id, c.revision_id)
               WHERE c.tenant_id = %(tenant_id)s
                 AND d.compartment_id = ANY(%(compartment_ids)s)
                 AND (
                     NOT %(project_only)s OR EXISTS (
                         SELECT 1 FROM compartments project_scope
                         WHERE project_scope.tenant_id = d.tenant_id
                           AND project_scope.id = d.compartment_id
                           AND project_scope.scope_type = 'project'
                           AND project_scope.scope_id = %(project_id)s
                     )
                 )
                 AND (
                     cardinality(%(subsystem_keys)s::text[]) = 0 OR EXISTS (
                         SELECT 1
                         FROM document_subsystems ds
                         JOIN subsystems s
                           ON (s.tenant_id, s.id) = (ds.tenant_id, ds.subsystem_id)
                         WHERE ds.tenant_id = d.tenant_id
                           AND ds.document_id = d.id
                           AND s.workspace_id = %(workspace_id)s
                           AND s.archived_at IS NULL
                           AND s.key = ANY(%(subsystem_keys)s::text[])
                     )
                 )
                 AND d.deleted_at IS NULL
                 AND (d.expires_at IS NULL OR d.expires_at > now())
                 AND c.chunk_profile = %(profile)s
                 AND c.search_document @@ query.value
           )
           SELECT chunk_id, document_id, revision, title, heading_path, text,
                  source_start, source_end, score
           FROM ranked
           WHERE %(after_score)s::real IS NULL
              OR score < %(after_score)s::real
              OR (score = %(after_score)s::real AND chunk_id > %(after_id)s)
           ORDER BY score DESC, chunk_id
           LIMIT %(limit)s""",
        {
            "query": query,
            "tenant_id": context.tenant_id,
            "compartment_ids": readable,
            "profile": ACTIVE_PROFILE.chunk_profile,
            "text_config": ACTIVE_PROFILE.text_search_config,
            "project_only": project_only,
            "project_id": context.project_id,
            "subsystem_keys": keys,
            "workspace_id": context.workspace_id,
            "after_score": after_score,
            "after_id": after_id,
            "limit": limit + 1,
        },
    ).fetchall()
    items = [LexicalResult(*row) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = _encode_cursor(
            query, last.score, last.chunk_id, project_only, keys, context
        )
    return LexicalPage(items, next_cursor)


def search_vector(
    connection: Connection,
    context: ContextTokenClaims,
    slot: IndexSlot,
    query_embedding: Sequence[float],
    *,
    limit: int = 10,
    project_only: bool = False,
    subsystem_keys: list[str] | None = None,
) -> list[VectorResult]:
    """Rerank authorized current TurboVec candidates with exact inner products."""
    if connection.info.transaction_status is TransactionStatus.IDLE:
        raise RuntimeError("vector search requires an active tenant transaction")
    _validate_limit(limit)
    keys = _validate_filters(context, project_only, subsystem_keys)
    query = normalize_embedding(query_embedding)

    lock_index_projection(connection, context.tenant_id, shared=True)
    owner = slot.current(context.tenant_id)
    if not owner.healthy:
        raise RuntimeError("vector index owner does not match request context")
    state = connection.execute(
        """SELECT generation, generation_path
           FROM vector_index_state
           WHERE tenant_id = %s AND profile = %s""",
        (context.tenant_id, ACTIVE_PROFILE.name),
    ).fetchone()
    if state is None:
        return []
    if state != (owner.generation, str(owner.path)):
        raise RuntimeError("vector index owner does not match active generation")

    configured_limit = _vector_allowlist_limit()
    params = _vector_params(
        context, keys, project_only, owner.generation, configured_limit
    )
    params["query_embedding"] = Vector(query).to_text()
    candidate_ids = None
    for _ in range(2):
        allowlist = [row[0] for row in _vector_rows(connection, params)]
        if not allowlist or (
            configured_limit is not None and len(allowlist) > configured_limit
        ):
            return []
        candidate_limit = min(limit * 3, len(allowlist))
        try:
            _, candidate_ids = owner.search(query, candidate_limit, allowlist)
            break
        except KeyError:
            continue
    if candidate_ids is None:
        return []

    if len(candidate_ids[0]) > candidate_limit:
        raise RuntimeError("vector index returned too many candidates")
    rows = _vector_rows(connection, params, [int(value) for value in candidate_ids[0]])
    return [VectorResult(*row[1:]) for row in rows[:limit]]


def fuse_ranked_results(
    lexical: Sequence[LexicalResult],
    vector: Sequence[VectorResult] | None = None,
    *,
    limit: int = 10,
) -> list[HybridResult]:
    """Combine ranked chunk lists with reciprocal rank fusion."""
    _validate_limit(limit)
    vector = vector or ()
    items = {item.chunk_id: item for item in vector}
    items.update({item.chunk_id: item for item in lexical})
    lexical_scores = {item.chunk_id: item.score for item in lexical}
    vector_scores = {item.chunk_id: item.score for item in vector}
    fused_scores: dict[UUID, float] = {}
    for ranked in (lexical, vector):
        chunk_ids = dict.fromkeys(item.chunk_id for item in ranked)
        for rank, chunk_id in enumerate(chunk_ids, 1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)

    results = []
    for chunk_id, fused_score in fused_scores.items():
        item = items[chunk_id]
        results.append(HybridResult(
            item.chunk_id, item.document_id, item.revision, item.title,
            item.heading_path, item.text, item.source_start, item.source_end,
            lexical_scores.get(chunk_id), vector_scores.get(chunk_id), fused_score,
        ))
    return sorted(results, key=lambda item: (-item.fused_score, item.chunk_id.int))[:limit]


def _readable_compartments(
    connection: Connection, context: ContextTokenClaims
) -> list[UUID]:
    return [
        compartment_id
        for compartment_id, permission in compartment_permissions(connection, context).items()
        if permission.can_read
    ]


def _vector_params(
    context: ContextTokenClaims,
    subsystem_keys: list[str],
    project_only: bool,
    generation: int,
    allowlist_limit: int | None,
) -> dict[str, object]:
    return {
        **permission_parameters(context),
        "project_only": project_only,
        "project_id": context.project_id,
        "subsystem_keys": subsystem_keys,
        "workspace_id": context.workspace_id,
        "chunk_profile": ACTIVE_PROFILE.chunk_profile,
        "embedding_model": ACTIVE_PROFILE.embedding_model,
        "embedding_version": ACTIVE_PROFILE.embedding_version,
        "embedding_dimension": ACTIVE_PROFILE.embedding_dimension,
        "generation": generation,
        "profile": ACTIVE_PROFILE.name,
        "allowlist_limit": allowlist_limit,
        "allowlist_query_limit": allowlist_limit + 1 if allowlist_limit else None,
    }


def _vector_rows(
    connection: Connection,
    params: dict[str, object],
    vector_ids: list[int] | None = None,
):
    details = vector_ids is not None
    params = {**params, "vector_ids": vector_ids or []}
    columns = (
        "c.vector_id, c.id, c.document_id, r.revision, r.title, c.heading_path, "
        "c.text, c.source_start, c.source_end, "
        "-(c.embedding <#> %(query_embedding)s::vector) AS score"
        if details
        else "c.vector_id"
    )
    candidate_filter = "AND c.vector_id = ANY(%(vector_ids)s)" if details else ""
    limit = (
        "ORDER BY score DESC, c.id"
        if details
        else "ORDER BY c.vector_id LIMIT %(allowlist_query_limit)s"
    )
    return connection.execute(
        f"""WITH readable_compartments AS (
                {COMPARTMENT_PERMISSIONS_SQL}
            )
            SELECT {columns}
            FROM chunks c
            JOIN documents d
              ON (d.tenant_id, d.id, d.current_revision_id) =
                 (c.tenant_id, c.document_id, c.revision_id)
            JOIN document_revisions r
              ON (r.tenant_id, r.document_id, r.id) =
                 (c.tenant_id, c.document_id, c.revision_id)
            JOIN vector_index_state state
              ON state.tenant_id = c.tenant_id AND state.profile = %(profile)s
            JOIN readable_compartments access
              ON access.id = d.compartment_id AND access.can_read
            WHERE c.tenant_id = %(tenant_id)s
              AND (
                  NOT %(project_only)s OR EXISTS (
                      SELECT 1 FROM compartments project_scope
                      WHERE project_scope.tenant_id = d.tenant_id
                        AND project_scope.id = d.compartment_id
                        AND project_scope.scope_type = 'project'
                        AND project_scope.scope_id = %(project_id)s
                  )
              )
              AND (
                  cardinality(%(subsystem_keys)s::text[]) = 0 OR EXISTS (
                      SELECT 1
                      FROM document_subsystems ds
                      JOIN subsystems s
                        ON (s.tenant_id, s.id) = (ds.tenant_id, ds.subsystem_id)
                      WHERE ds.tenant_id = d.tenant_id
                        AND ds.document_id = d.id
                        AND s.workspace_id = %(workspace_id)s
                        AND s.archived_at IS NULL
                        AND s.key = ANY(%(subsystem_keys)s::text[])
                  )
              )
              AND d.deleted_at IS NULL
              AND (d.expires_at IS NULL OR d.expires_at > now())
              AND c.chunk_profile = %(chunk_profile)s
              AND c.embedding_state = 'ready'
              AND c.embedding_model = %(embedding_model)s
              AND c.embedding_version = %(embedding_version)s
              AND c.embedding_dimension = %(embedding_dimension)s
              AND c.indexed_revision = r.revision
              AND c.indexed_generation = %(generation)s
              AND state.generation = %(generation)s
              {candidate_filter}
            {limit}""",
        params,
    ).fetchall()


def _vector_allowlist_limit() -> int | None:
    value = os.getenv("MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT")
    if value is None or not value.strip():
        return None
    try:
        limit = int(value)
    except ValueError:
        raise RuntimeError("MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT must be a positive integer") from None
    if limit < 1:
        raise RuntimeError("MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT must be a positive integer")
    return limit


def _validate_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be from 1 through 100")


def _validate_filters(
    context: ContextTokenClaims,
    project_only: bool,
    subsystem_keys: list[str] | None,
) -> list[str]:
    if type(project_only) is not bool:
        raise ValueError("project_only must be a boolean")
    if project_only and context.project_id is None:
        raise ValueError("project filter requires project context")
    if subsystem_keys is None:
        keys = []
    elif (
        not isinstance(subsystem_keys, list)
        or len(subsystem_keys) > 50
        or any(
            not isinstance(key, str) or not key or len(key) > 200
            for key in subsystem_keys
        )
    ):
        raise ValueError("subsystem_keys must contain up to 50 nonempty keys")
    else:
        keys = sorted(set(subsystem_keys))
    if keys and context.workspace_id is None:
        raise ValueError("subsystem filter requires workspace context")
    return keys


def _encode_cursor(
    query: str,
    score: float,
    chunk_id: UUID,
    project_only: bool,
    subsystem_keys: list[str],
    context: ContextTokenClaims,
) -> str:
    payload = json.dumps(
        {
            "c": str(chunk_id),
            "q": _search_hash(query, project_only, subsystem_keys, context),
            "s": score,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode()


def _decode_cursor(
    cursor: str,
    query: str,
    project_only: bool,
    subsystem_keys: list[str],
    context: ContextTokenClaims,
) -> tuple[float, UUID]:
    if not isinstance(cursor, str) or len(cursor) > 1024:
        raise ValueError("invalid cursor")
    try:
        encoded = cursor.encode("ascii")
        payload = json.loads(
            base64.b64decode(encoded + b"=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != {"c", "q", "s"}
            or payload["q"] != _search_hash(
                query, project_only, subsystem_keys, context
            )
            or not isinstance(payload["c"], str)
            or type(payload["s"]) not in (int, float)
            or not math.isfinite(payload["s"])
        ):
            raise ValueError
        return float(payload["s"]), UUID(payload["c"])
    except (UnicodeError, ValueError, json.JSONDecodeError, TypeError):
        raise ValueError("invalid cursor") from None


def _search_hash(
    query: str,
    project_only: bool,
    subsystem_keys: list[str],
    context: ContextTokenClaims,
) -> str:
    value = json.dumps(
        [
            query,
            project_only,
            subsystem_keys,
            str(context.tenant_id),
            str(context.user_id),
            str(context.agent_id) if context.agent_id else None,
            str(context.workspace_id) if context.workspace_id else None,
            str(context.project_id) if context.project_id else None,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(value.encode()).hexdigest()[:16]
