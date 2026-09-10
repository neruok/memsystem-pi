"""Tenant-scoped embedding job worker."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from pgvector.psycopg import register_vector
from psycopg import Connection

from memsystem.database import tenant_transaction
from memsystem.embeddings import ACTIVE_PROFILE, EmbeddingProvider, normalize_embedding

MAX_ATTEMPTS = 5
CLAIM_LEASE_SECONDS = 300


def _index_lock_name(tenant_id: UUID | str) -> str:
    return f"{UUID(str(tenant_id))}:{ACTIVE_PROFILE.name}"


def lock_index_projection(
    connection: Connection, tenant_id: UUID | str, *, shared: bool
) -> None:
    """Lock one tenant/profile projection for the current transaction."""
    function = "pg_advisory_xact_lock_shared" if shared else "pg_advisory_xact_lock"
    connection.execute(
        f"SELECT {function}(hashtextextended(%s, 0))",
        (_index_lock_name(tenant_id),),
    )


def lock_index_projection_session(
    connection: Connection, tenant_id: UUID | str
) -> None:
    if not connection.autocommit:
        raise RuntimeError("session index lock requires an autocommit connection")
    connection.execute(
        "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
        (_index_lock_name(tenant_id),),
    )


def unlock_index_projection_session(
    connection: Connection, tenant_id: UUID | str
) -> None:
    unlocked = connection.execute(
        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
        (_index_lock_name(tenant_id),),
    ).fetchone()[0]
    if not unlocked:
        raise RuntimeError("session index lock was not held")


@dataclass(frozen=True)
class EmbeddingJob:
    id: UUID
    chunk_id: UUID
    document_id: UUID
    revision_id: UUID
    vector_id: int
    text: str
    attempts: int
    claimed_at: datetime
    active: bool


def claim_embedding_job(connection: Connection, tenant_id: UUID | str) -> EmbeddingJob | None:
    """Claim the oldest eligible embedding job without blocking another worker."""
    tenant_id = UUID(str(tenant_id))
    with tenant_transaction(connection, tenant_id):
        connection.execute(
            """UPDATE jobs
               SET state = 'failed', error = 'claim lease expired after maximum attempts',
                   updated_at = now()
               WHERE tenant_id = %s AND profile = %s AND kind = 'embed'
                 AND state = 'running' AND attempts >= %s
                 AND claimed_at <= now() - make_interval(secs => %s)""",
            (tenant_id, ACTIVE_PROFILE.name, MAX_ATTEMPTS, CLAIM_LEASE_SECONDS),
        )
        row = connection.execute(
            """WITH candidate AS (
                   SELECT j.id
                   FROM jobs j
                   WHERE j.tenant_id = %s AND j.profile = %s AND j.kind = 'embed'
                     AND j.attempts < %s
                     AND (
                         (j.state IN ('pending', 'failed') AND j.next_attempt_at <= now())
                         OR (j.state = 'running' AND j.claimed_at <=
                             now() - make_interval(secs => %s))
                     )
                   ORDER BY j.next_attempt_at, j.created_at, j.id
                   FOR UPDATE SKIP LOCKED
                   LIMIT 1
               ), claimed AS (
                   UPDATE jobs j
                   SET state = 'running',
                       attempts = j.attempts + CASE WHEN j.state = 'running' THEN 1 ELSE 0 END,
                       claimed_at = now(), updated_at = now()
                   FROM candidate
                   WHERE j.tenant_id = %s AND j.id = candidate.id
                   RETURNING j.id, j.chunk_id, j.document_id, j.revision_id,
                             j.vector_id, j.attempts, j.claimed_at
               )
               SELECT claimed.id, claimed.chunk_id, claimed.document_id,
                      claimed.revision_id, claimed.vector_id, c.text, claimed.attempts,
                      claimed.claimed_at,
                      (d.current_revision_id = c.revision_id
                       AND d.deleted_at IS NULL
                       AND (d.expires_at IS NULL OR d.expires_at > now())) AS active
               FROM claimed
               JOIN chunks c ON c.tenant_id = %s AND c.id = claimed.chunk_id
               JOIN documents d ON d.tenant_id = c.tenant_id AND d.id = c.document_id""",
            (
                tenant_id,
                ACTIVE_PROFILE.name,
                MAX_ATTEMPTS,
                CLAIM_LEASE_SECONDS,
                tenant_id,
                tenant_id,
            ),
        ).fetchone()
    return EmbeddingJob(*row) if row else None


def process_embedding_job(
    connection: Connection,
    tenant_id: UUID | str,
    provider: EmbeddingProvider,
) -> bool:
    """Process one embedding job. Return false when no work is ready."""
    _validate_provider(provider)
    tenant_id = UUID(str(tenant_id))
    job = claim_embedding_job(connection, tenant_id)
    if job is None:
        return False
    if not job.active:
        _complete_claim(connection, tenant_id, job)
        return True

    try:
        embedding = normalize_embedding(provider.embed(job.text))
    except Exception as error:
        _record_failure(connection, tenant_id, job, error)
        return True

    with tenant_transaction(connection, tenant_id):
        lock_index_projection(connection, tenant_id, shared=True)
        owned = connection.execute(
            """SELECT 1 FROM jobs
               WHERE tenant_id = %s AND id = %s AND state = 'running'
                 AND claimed_at = %s
               FOR UPDATE""",
            (tenant_id, job.id, job.claimed_at),
        ).fetchone()
        if not owned:
            return True

        register_vector(connection)
        current = connection.execute(
            """UPDATE chunks c
               SET embedding = %s, embedding_model = %s, embedding_version = %s,
                   embedding_dimension = %s, embedding_state = 'ready',
                   embedding_error = NULL, embedding_attempted_at = now()
               FROM documents d
               WHERE c.tenant_id = %s AND c.id = %s
                 AND d.tenant_id = c.tenant_id AND d.id = c.document_id
                 AND d.current_revision_id = c.revision_id
                 AND d.deleted_at IS NULL
                 AND (d.expires_at IS NULL OR d.expires_at > now())
               RETURNING c.document_id, c.revision_id, c.id, c.vector_id""",
            (
                embedding,
                ACTIVE_PROFILE.embedding_model,
                ACTIVE_PROFILE.embedding_version,
                ACTIVE_PROFILE.embedding_dimension,
                tenant_id,
                job.chunk_id,
            ),
        ).fetchone()
        if current:
            connection.execute(
                """INSERT INTO jobs
                   (tenant_id, document_id, revision_id, chunk_id, vector_id, profile, kind)
                   VALUES (%s, %s, %s, %s, %s, %s, 'index_add')""",
                (tenant_id, *current, ACTIVE_PROFILE.name),
            )
        _complete_claim_in_transaction(connection, tenant_id, job)
    return True


def _validate_provider(provider: EmbeddingProvider) -> None:
    expected = (
        ACTIVE_PROFILE.embedding_provider,
        ACTIVE_PROFILE.embedding_model,
        ACTIVE_PROFILE.embedding_version,
        ACTIVE_PROFILE.embedding_dimension,
    )
    actual = (provider.provider, provider.model, provider.version, provider.dimension)
    if actual != expected:
        raise ValueError("embedding provider does not match the active retrieval profile")


def _complete_claim(connection: Connection, tenant_id: UUID, job: EmbeddingJob) -> None:
    with tenant_transaction(connection, tenant_id):
        _complete_claim_in_transaction(connection, tenant_id, job)


def _complete_claim_in_transaction(
    connection: Connection, tenant_id: UUID, job: EmbeddingJob
) -> None:
    connection.execute(
        """UPDATE jobs SET state = 'complete', error = NULL, updated_at = now()
           WHERE tenant_id = %s AND id = %s AND state = 'running'
             AND claimed_at = %s""",
        (tenant_id, job.id, job.claimed_at),
    )


def _record_failure(
    connection: Connection,
    tenant_id: UUID,
    job: EmbeddingJob,
    error: Exception,
) -> None:
    message = str(error)[:2000]
    retry_seconds = min(300, 2 ** (job.attempts + 1))
    with tenant_transaction(connection, tenant_id):
        failed = connection.execute(
            """UPDATE jobs
               SET state = 'failed', attempts = attempts + 1, error = %s,
                   next_attempt_at = now() + make_interval(secs => %s),
                   updated_at = now()
               WHERE tenant_id = %s AND id = %s AND state = 'running'
                 AND claimed_at = %s
               RETURNING 1""",
            (message, retry_seconds, tenant_id, job.id, job.claimed_at),
        ).fetchone()
        if failed:
            connection.execute(
                """UPDATE chunks
                   SET embedding_state = 'failed', embedding_error = %s,
                       embedding_attempted_at = now()
                   WHERE tenant_id = %s AND id = %s""",
                (message, tenant_id, job.chunk_id),
            )
