"""Durable tenant-scoped TurboVec projection worker."""

from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import os
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection
from turbovec import IdMapIndex

from memsystem.database import tenant_transaction
from memsystem.embeddings import ACTIVE_PROFILE
from memsystem.jobs import (
    CLAIM_LEASE_SECONDS,
    MAX_ATTEMPTS,
    lock_index_projection,
    lock_index_projection_session,
    unlock_index_projection_session,
)

IndexJobKind = Literal["index_add", "index_remove"]


@dataclass(frozen=True)
class IndexJob:
    id: UUID
    kind: IndexJobKind
    vector_id: int
    attempts: int
    claimed_at: datetime


class TurboVecIndex:
    """Own one tenant's mutable TurboVec generation file."""

    def __init__(
        self, tenant_id: UUID | str, path: str | Path, generation: int = 1
    ):
        if generation < 1:
            raise ValueError("generation must be positive")
        self.tenant_id = UUID(str(tenant_id))
        self.path = Path(path).resolve()
        self.generation = generation
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.path.with_suffix(self.path.suffix + ".lock").open("a+b")
        self._thread_lock = RLock()
        self._healthy = False
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.reload()
        except Exception:
            self._lock_file.close()
            raise

    @property
    def healthy(self) -> bool:
        return self._healthy

    def reload(self) -> None:
        with self._thread_lock:
            if self._lock_file.closed:
                raise RuntimeError("closed index owner cannot be reloaded")
            self._healthy = False
            index = (
                IdMapIndex.load(str(self.path))
                if self.path.exists()
                else IdMapIndex(
                    dim=ACTIVE_PROFILE.embedding_dimension,
                    bit_width=ACTIVE_PROFILE.turbovec_bits,
                )
            )
            expected_calibration = (
                "calibrated" if ACTIVE_PROFILE.turbovec_calibrated else "uncalibrated"
            )
            if (
                index.dim != ACTIVE_PROFILE.embedding_dimension
                or index.bit_width != ACTIVE_PROFILE.turbovec_bits
                or index.calibration_state != expected_calibration
            ):
                raise ValueError("TurboVec index does not match the active retrieval profile")
            index.prepare()
            self.index = index
            self._healthy = True

    def close(self) -> None:
        with self._thread_lock:
            self._healthy = False
            if not self._lock_file.closed:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()

    def add(self, vector_id: int, embedding) -> None:
        with self._thread_lock:
            self._require_healthy()
            self.index.remove(vector_id)
            vectors = np.ascontiguousarray([embedding], dtype=np.float32)
            ids = np.ascontiguousarray([vector_id], dtype=np.uint64)
            self.index.add_with_ids(vectors, ids)

    def add_many(self, vector_ids, embeddings) -> None:
        with self._thread_lock:
            self._require_healthy()
            vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
            ids = np.ascontiguousarray(vector_ids, dtype=np.uint64)
            self.index.add_with_ids(vectors, ids)

    def remove(self, vector_id: int) -> None:
        with self._thread_lock:
            self._require_healthy()
            self.index.remove(vector_id)

    def prepare(self) -> None:
        with self._thread_lock:
            self._require_healthy()
            self.index.prepare()

    def search(self, query, limit: int, allowlist: list[int]):
        with self._thread_lock:
            self._require_healthy()
            queries = np.ascontiguousarray([query], dtype=np.float32)
            allowed = np.ascontiguousarray(allowlist, dtype=np.uint64)
            return self.index.search(queries, limit, allowlist=allowed)

    def checksum(self) -> str:
        with self._thread_lock:
            self._require_healthy()
            return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def sync(self) -> str:
        with self._thread_lock:
            self._require_healthy()
            self.index.sync(str(self.path))
            # ponytail: full-file hash per sync; batch jobs if measured index size makes this costly.
            return self.checksum()

    def _require_healthy(self) -> None:
        if not self._healthy:
            raise RuntimeError("index owner must be reconstructed after reload failure")


class IndexSlot:
    """Hold the current owner for one tenant and profile."""

    def __init__(self, owner: TurboVecIndex):
        self._lock = RLock()
        self._owner = owner

    def current(self, tenant_id: UUID | str) -> TurboVecIndex:
        with self._lock:
            tenant_id = UUID(str(tenant_id))
            if self._owner.tenant_id != tenant_id:
                raise ValueError("index owner tenant does not match worker tenant")
            return self._owner

    def swap(self, owner: TurboVecIndex) -> TurboVecIndex:
        with self._lock:
            if owner.tenant_id != self._owner.tenant_id:
                raise ValueError("replacement index owner has a different tenant")
            previous, self._owner = self._owner, owner
            return previous


def claim_index_job(connection: Connection, tenant_id: UUID | str) -> IndexJob | None:
    """Claim the oldest add or remove job, including expired claims."""
    tenant_id = UUID(str(tenant_id))
    with tenant_transaction(connection, tenant_id):
        connection.execute(
            """UPDATE jobs
               SET state = 'failed', error = 'claim lease expired after maximum attempts',
                   updated_at = now()
               WHERE tenant_id = %s AND profile = %s
                 AND kind IN ('index_add', 'index_remove')
                 AND state = 'running' AND attempts >= %s
                 AND claimed_at <= now() - make_interval(secs => %s)""",
            (tenant_id, ACTIVE_PROFILE.name, MAX_ATTEMPTS, CLAIM_LEASE_SECONDS),
        )
        row = connection.execute(
            """WITH candidate AS (
                   SELECT j.id
                   FROM jobs j
                   WHERE j.tenant_id = %s AND j.profile = %s
                     AND j.kind IN ('index_add', 'index_remove')
                     AND j.attempts < %s
                     AND (
                         (j.state IN ('pending', 'failed') AND j.next_attempt_at <= now())
                         OR (j.state = 'running' AND j.claimed_at <=
                             now() - make_interval(secs => %s))
                     )
                   ORDER BY j.next_attempt_at, j.created_at, j.id
                   FOR UPDATE SKIP LOCKED
                   LIMIT 1
               )
               UPDATE jobs j
               SET state = 'running',
                   attempts = j.attempts + CASE WHEN j.state = 'running' THEN 1 ELSE 0 END,
                   claimed_at = now(), updated_at = now()
               FROM candidate
               WHERE j.tenant_id = %s AND j.id = candidate.id
               RETURNING j.id, j.kind::text, j.vector_id, j.attempts, j.claimed_at""",
            (
                tenant_id,
                ACTIVE_PROFILE.name,
                MAX_ATTEMPTS,
                CLAIM_LEASE_SECONDS,
                tenant_id,
            ),
        ).fetchone()
    return IndexJob(*row) if row else None


def process_index_job(
    connection: Connection,
    tenant_id: UUID | str,
    slot: IndexSlot,
) -> bool:
    """Apply one index job. Return false when no work is ready."""
    tenant_id = UUID(str(tenant_id))
    job = claim_index_job(connection, tenant_id)
    if job is None:
        return False

    owner = None
    try:
        with tenant_transaction(connection, tenant_id):
            lock_index_projection(connection, tenant_id, shared=False)
            owner = slot.current(tenant_id)
            if not owner.healthy:
                raise RuntimeError("index owner must be reconstructed after reload failure")
            owned = connection.execute(
                """SELECT 1 FROM jobs
                   WHERE tenant_id = %s AND id = %s AND state = 'running'
                     AND claimed_at = %s
                   FOR UPDATE""",
                (tenant_id, job.id, job.claimed_at),
            ).fetchone()
            if not owned:
                return True

            state = connection.execute(
                """SELECT generation, generation_path, checksum
                   FROM vector_index_state
                   WHERE tenant_id = %s AND profile = %s
                   FOR UPDATE""",
                (tenant_id, ACTIVE_PROFILE.name),
            ).fetchone()
            if state:
                if state[:2] != (owner.generation, str(owner.path)):
                    raise RuntimeError("index owner does not match active generation")
                if not owner.path.exists() or owner.checksum() != state[2]:
                    raise RuntimeError("active index checksum does not match PostgreSQL")
            elif owner.path.exists():
                raise RuntimeError("index file has no PostgreSQL generation state")

            revision = None
            if job.kind == "index_add":
                register_vector(connection)
                current = connection.execute(
                    """SELECT c.embedding, r.revision
                       FROM jobs j
                       JOIN chunks c
                         ON (c.tenant_id, c.id, c.vector_id) =
                            (j.tenant_id, j.chunk_id, j.vector_id)
                       JOIN documents d
                         ON (d.tenant_id, d.id, d.current_revision_id) =
                            (c.tenant_id, c.document_id, c.revision_id)
                       JOIN document_revisions r
                         ON (r.tenant_id, r.document_id, r.id) =
                            (c.tenant_id, c.document_id, c.revision_id)
                       WHERE j.tenant_id = %s AND j.id = %s
                         AND c.embedding_state = 'ready'
                         AND c.embedding_model = %s AND c.embedding_version = %s
                         AND c.embedding_dimension = %s
                         AND d.deleted_at IS NULL
                         AND (d.expires_at IS NULL OR d.expires_at > now())""",
                    (
                        tenant_id,
                        job.id,
                        ACTIVE_PROFILE.embedding_model,
                        ACTIVE_PROFILE.embedding_version,
                        ACTIVE_PROFILE.embedding_dimension,
                    ),
                ).fetchone()
                if current is None:
                    _complete_index_job(connection, tenant_id, job)
                    return True
                embedding, revision = current
                owner.add(job.vector_id, embedding.to_numpy())
            else:
                owner.remove(job.vector_id)

            checksum = owner.sync()
            connection.execute(
                """INSERT INTO vector_index_state
                   (tenant_id, profile, generation, generation_path, checksum)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (tenant_id, profile) DO UPDATE
                   SET generation = EXCLUDED.generation,
                       generation_path = EXCLUDED.generation_path,
                       checksum = EXCLUDED.checksum,
                       updated_at = now()""",
                (
                    tenant_id,
                    ACTIVE_PROFILE.name,
                    owner.generation,
                    str(owner.path),
                    checksum,
                ),
            )
            if revision is not None:
                connection.execute(
                    """UPDATE chunks
                       SET indexed_revision = %s, indexed_generation = %s
                       WHERE tenant_id = %s AND vector_id = %s""",
                    (revision, owner.generation, tenant_id, job.vector_id),
                )
            elif job.kind == "index_remove":
                connection.execute(
                    """UPDATE chunks
                       SET indexed_revision = NULL, indexed_generation = NULL
                       WHERE tenant_id = %s AND vector_id = %s""",
                    (tenant_id, job.vector_id),
                )
            _complete_index_job(connection, tenant_id, job)
    except Exception as error:
        try:
            if owner is not None:
                owner.reload()
        except Exception:
            pass
        _record_index_failure(connection, tenant_id, job, error)
    return True


def load_index_slot(
    connection: Connection, tenant_id: UUID | str
) -> IndexSlot:
    """Load and verify the published generation for one tenant."""
    tenant_id = UUID(str(tenant_id))
    with tenant_transaction(connection, tenant_id):
        state = connection.execute(
            """SELECT generation, generation_path, checksum
               FROM vector_index_state
               WHERE tenant_id = %s AND profile = %s""",
            (tenant_id, ACTIVE_PROFILE.name),
        ).fetchone()
    if state is None:
        raise RuntimeError("tenant has no published vector index")
    generation, path, checksum = state
    owner = TurboVecIndex(tenant_id, path, generation)
    try:
        if owner.checksum() != checksum:
            raise RuntimeError("active index checksum does not match PostgreSQL")
    except Exception:
        owner.close()
        raise
    return IndexSlot(owner)


def load_or_rebuild_index_slot(
    connection: Connection,
    tenant_id: UUID | str,
    directory: str | Path,
) -> IndexSlot:
    """Load the published generation or rebuild it from PostgreSQL."""
    tenant_id = UUID(str(tenant_id))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    _cleanup_pending_generations(connection, directory, tenant_id)
    try:
        return load_index_slot(connection, tenant_id)
    except (OSError, RuntimeError, ValueError):
        with tenant_transaction(connection, tenant_id):
            state = connection.execute(
                """SELECT generation FROM vector_index_state
                   WHERE tenant_id = %s AND profile = %s""",
                (tenant_id, ACTIVE_PROFILE.name),
            ).fetchone()
        placeholder_path = directory / f".{tenant_id}-{uuid4().hex}.recovery"
        placeholder = TurboVecIndex(
            tenant_id, placeholder_path, state[0] if state else 1
        )
        slot = IndexSlot(placeholder)
        try:
            rebuild_index(connection, tenant_id, slot, directory)
        except Exception:
            placeholder.close()
            raise
        finally:
            placeholder_path.unlink(missing_ok=True)
            placeholder_path.with_suffix(placeholder_path.suffix + ".lock").unlink(
                missing_ok=True
            )
            _fsync_directory(directory)
        return slot


def rebuild_index(
    connection: Connection,
    tenant_id: UUID | str,
    slot: IndexSlot,
    directory: str | Path,
) -> TurboVecIndex:
    """Rebuild one generation from PostgreSQL and publish it under a short fence."""
    tenant_id = UUID(str(tenant_id))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    with tenant_transaction(connection, tenant_id):
        lock_index_projection(connection, tenant_id, shared=False)
        current_owner = slot.current(tenant_id)
        state = connection.execute(
            """SELECT generation FROM vector_index_state
               WHERE tenant_id = %s AND profile = %s FOR UPDATE""",
            (tenant_id, ACTIVE_PROFILE.name),
        ).fetchone()
        old_generation = state[0] if state else current_owner.generation
        marker = connection.execute(
            """SELECT COALESCE(max(enqueue_seq), 0) FROM jobs
               WHERE tenant_id = %s AND profile = %s""",
            (tenant_id, ACTIVE_PROFILE.name),
        ).fetchone()[0]

    generation = old_generation + 1
    stem = f"{tenant_id}-{ACTIVE_PROFILE.name}-g{generation}-{uuid4().hex}"
    temporary_path = directory / f".{stem}.building"
    final_path = directory / f"{stem}.tvim"
    pending_path = directory / f"{stem}.pending"
    pending_lock = pending_path.open("x+b")
    fcntl.flock(pending_lock, fcntl.LOCK_EX)
    os.fsync(pending_lock.fileno())
    _fsync_directory(directory)
    candidate = None
    database_published = published = False
    try:
        candidate = TurboVecIndex(tenant_id, temporary_path, generation)
        with tenant_transaction(connection, tenant_id):
            register_vector(connection)
            ready = _ready_vectors(connection, tenant_id)
            if ready:
                candidate.add_many(
                    [vector_id for vector_id, _, _ in ready],
                    np.vstack([embedding.to_numpy() for _, embedding, _ in ready]),
                )
        candidate.prepare()
        candidate.sync()
        candidate.close()
        os.replace(temporary_path, final_path)
        _fsync_directory(directory)
        candidate = TurboVecIndex(tenant_id, final_path, generation)

        lock_index_projection_session(connection, tenant_id)
        try:
            with tenant_transaction(connection, tenant_id):
                state = connection.execute(
                    """SELECT generation FROM vector_index_state
                       WHERE tenant_id = %s AND profile = %s FOR UPDATE""",
                    (tenant_id, ACTIVE_PROFILE.name),
                ).fetchone()
                if (state[0] if state else current_owner.generation) != old_generation:
                    raise RuntimeError("active generation changed during rebuild")

                register_vector(connection)
                replay = connection.execute(
                    """SELECT kind::text, vector_id
                       FROM jobs
                       WHERE tenant_id = %s AND profile = %s AND enqueue_seq > %s
                       ORDER BY enqueue_seq""",
                    (tenant_id, ACTIVE_PROFILE.name, marker),
                ).fetchall()
                for kind, vector_id in replay:
                    if kind == "index_remove":
                        candidate.remove(vector_id)
                    elif kind == "index_add":
                        current = _ready_vector(connection, tenant_id, vector_id)
                        if current:
                            candidate.add(vector_id, current[0].to_numpy())
                        else:
                            candidate.remove(vector_id)

                candidate.prepare()
                checksum = candidate.sync()
                tail = connection.execute(
                    """SELECT COALESCE(max(enqueue_seq), 0) FROM jobs
                       WHERE tenant_id = %s AND profile = %s""",
                    (tenant_id, ACTIVE_PROFILE.name),
                ).fetchone()[0]
                connection.execute(
                    """UPDATE chunks
                       SET indexed_revision = NULL, indexed_generation = NULL
                       WHERE tenant_id = %s""",
                    (tenant_id,),
                )
                connection.execute(
                    """UPDATE chunks c
                       SET indexed_revision = r.revision, indexed_generation = %s
                       FROM documents d, document_revisions r
                       WHERE c.tenant_id = %s
                         AND (d.tenant_id, d.id, d.current_revision_id) =
                             (c.tenant_id, c.document_id, c.revision_id)
                         AND (r.tenant_id, r.document_id, r.id) =
                             (c.tenant_id, c.document_id, c.revision_id)
                         AND d.deleted_at IS NULL
                         AND (d.expires_at IS NULL OR d.expires_at > now())
                         AND c.chunk_profile = %s AND c.embedding_state = 'ready'
                         AND c.embedding_model = %s AND c.embedding_version = %s
                         AND c.embedding_dimension = %s""",
                    (
                        generation,
                        tenant_id,
                        ACTIVE_PROFILE.chunk_profile,
                        ACTIVE_PROFILE.embedding_model,
                        ACTIVE_PROFILE.embedding_version,
                        ACTIVE_PROFILE.embedding_dimension,
                    ),
                )
                connection.execute(
                    """INSERT INTO vector_index_state
                       (tenant_id, profile, generation, generation_path, checksum,
                        job_high_water_mark)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (tenant_id, profile) DO UPDATE
                       SET generation = EXCLUDED.generation,
                           generation_path = EXCLUDED.generation_path,
                           checksum = EXCLUDED.checksum,
                           job_high_water_mark = EXCLUDED.job_high_water_mark,
                           updated_at = now()""",
                    (
                        tenant_id,
                        ACTIVE_PROFILE.name,
                        generation,
                        str(final_path),
                        checksum,
                        tail,
                    ),
                )
                connection.execute(
                    """UPDATE jobs
                       SET state = 'complete', error = NULL, updated_at = now()
                       WHERE tenant_id = %s AND profile = %s
                         AND kind IN ('index_add', 'index_remove')
                         AND enqueue_seq <= %s""",
                    (tenant_id, ACTIVE_PROFILE.name, tail),
                )
            database_published = True
            previous = slot.swap(candidate)
            published = True
            previous.close()
        finally:
            unlock_index_projection_session(connection, tenant_id)
    finally:
        if candidate is not None and not published:
            candidate.close()
        temporary_path.unlink(missing_ok=True)
        temporary_path.with_suffix(temporary_path.suffix + ".lock").unlink(
            missing_ok=True
        )
        if not database_published:
            final_path.unlink(missing_ok=True)
            final_path.with_suffix(final_path.suffix + ".lock").unlink(missing_ok=True)
        fcntl.flock(pending_lock, fcntl.LOCK_UN)
        pending_lock.close()
        pending_path.unlink(missing_ok=True)
        _fsync_directory(directory)
    return candidate


def _cleanup_pending_generations(
    connection: Connection, directory: Path, tenant_id: UUID
) -> None:
    for pending_path in directory.glob(f"{tenant_id}-*.pending"):
        with pending_path.open("a+b") as pending:
            try:
                fcntl.flock(pending, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            with tenant_transaction(connection, tenant_id):
                published = connection.execute(
                    """SELECT generation_path FROM vector_index_state
                       WHERE tenant_id = %s AND profile = %s""",
                    (tenant_id, ACTIVE_PROFILE.name),
                ).fetchone()
            published_path = Path(published[0]).resolve() if published else None
            stem = pending_path.stem
            temporary_path = directory / f".{stem}.building"
            final_path = directory / f"{stem}.tvim"
            for path in (
                temporary_path,
                temporary_path.with_suffix(temporary_path.suffix + ".lock"),
            ):
                path.unlink(missing_ok=True)
            if published_path is None or final_path.resolve() != published_path:
                final_path.unlink(missing_ok=True)
                final_path.with_suffix(final_path.suffix + ".lock").unlink(
                    missing_ok=True
                )
            pending_path.unlink(missing_ok=True)
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ready_vectors(connection: Connection, tenant_id: UUID):
    return connection.execute(
        """SELECT c.vector_id, c.embedding, r.revision
           FROM chunks c
           JOIN documents d
             ON (d.tenant_id, d.id, d.current_revision_id) =
                (c.tenant_id, c.document_id, c.revision_id)
           JOIN document_revisions r
             ON (r.tenant_id, r.document_id, r.id) =
                (c.tenant_id, c.document_id, c.revision_id)
           WHERE c.tenant_id = %s
             AND d.deleted_at IS NULL
             AND (d.expires_at IS NULL OR d.expires_at > now())
             AND c.chunk_profile = %s AND c.embedding_state = 'ready'
             AND c.embedding_model = %s AND c.embedding_version = %s
             AND c.embedding_dimension = %s
           ORDER BY c.vector_id""",
        (
            tenant_id,
            ACTIVE_PROFILE.chunk_profile,
            ACTIVE_PROFILE.embedding_model,
            ACTIVE_PROFILE.embedding_version,
            ACTIVE_PROFILE.embedding_dimension,
        ),
    ).fetchall()


def _ready_vector(connection: Connection, tenant_id: UUID, vector_id: int):
    return connection.execute(
        """SELECT c.embedding
           FROM chunks c
           JOIN documents d
             ON (d.tenant_id, d.id, d.current_revision_id) =
                (c.tenant_id, c.document_id, c.revision_id)
           WHERE c.tenant_id = %s AND c.vector_id = %s
             AND d.deleted_at IS NULL
             AND (d.expires_at IS NULL OR d.expires_at > now())
             AND c.chunk_profile = %s AND c.embedding_state = 'ready'
             AND c.embedding_model = %s AND c.embedding_version = %s
             AND c.embedding_dimension = %s""",
        (
            tenant_id,
            vector_id,
            ACTIVE_PROFILE.chunk_profile,
            ACTIVE_PROFILE.embedding_model,
            ACTIVE_PROFILE.embedding_version,
            ACTIVE_PROFILE.embedding_dimension,
        ),
    ).fetchone()


def _complete_index_job(
    connection: Connection, tenant_id: UUID, job: IndexJob
) -> None:
    connection.execute(
        """UPDATE jobs SET state = 'complete', error = NULL, updated_at = now()
           WHERE tenant_id = %s AND id = %s AND state = 'running'
             AND claimed_at = %s""",
        (tenant_id, job.id, job.claimed_at),
    )


def _record_index_failure(
    connection: Connection, tenant_id: UUID, job: IndexJob, error: Exception
) -> None:
    with tenant_transaction(connection, tenant_id):
        connection.execute(
            """UPDATE jobs
               SET state = 'failed', attempts = attempts + 1, error = %s,
                   next_attempt_at = now() + make_interval(secs => %s),
                   updated_at = now()
               WHERE tenant_id = %s AND id = %s AND state = 'running'
                 AND claimed_at = %s""",
            (
                str(error)[:2000],
                min(300, 2 ** (job.attempts + 1)),
                tenant_id,
                job.id,
                job.claimed_at,
            ),
        )
