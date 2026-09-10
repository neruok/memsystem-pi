import base64
import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import numpy as np
import psycopg
import pytest
from starlette.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError

from memsystem.authorization import AuthorizationError, compartment_permissions, require_compartment_access
from memsystem.context_token import ContextTokenClaims, ContextTokenCodec
from memsystem.database import tenant_transaction
from memsystem.documents import (
    DocumentNotFound,
    RevisionConflict,
    create_document,
    read_document,
    soft_delete_document,
    update_document,
)
from memsystem.idempotency import IdempotencyConflict
from memsystem.jobs import claim_embedding_job, process_embedding_job
from memsystem.links import (
    delete_document_link,
    read_backlinks,
    read_backlinks_page,
    set_document_link,
)
from memsystem.navigation import (
    read_collection_children,
    read_document_content,
    read_revision_history,
)
from memsystem.request_context import ContextResolutionError, resolve_context
from memsystem.retrieval import search_lexical, search_vector
from memsystem.vector_index import (
    IndexSlot,
    TurboVecIndex,
    claim_index_job,
    load_or_rebuild_index_slot,
    process_index_job,
    rebuild_index,
)
from memsystem import server as server_module
from memsystem.server import mcp


@pytest.fixture
def database_url() -> str:
    return os.getenv(
        "MEMSYSTEM_TEST_DATABASE_URL",
        "postgresql://memsystem_service:memsystem@127.0.0.1:5432/memsystem",
    )


@pytest.fixture
def admin_database_url() -> str:
    return os.getenv(
        "MEMSYSTEM_TEST_ADMIN_DATABASE_URL",
        "postgresql://postgres:postgres@127.0.0.1:5432/memsystem",
    )


@pytest.fixture
def context_records(database_url: str, admin_database_url: str):
    records = {
        name: uuid4()
        for name in (
            "tenant", "user", "agent", "run", "workspace", "project", "denied_project",
            "root_compartment", "global_compartment", "user_compartment", "agent_compartment",
            "workspace_compartment", "project_compartment",
        )
    }
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, records["tenant"]):
            connection.execute(
                "INSERT INTO tenants (id, name) VALUES (%s, 'tenant')",
                (records["tenant"],),
            )
            connection.execute(
                "INSERT INTO tenant_memberships (tenant_id, user_id) VALUES (%s, %s)",
                (records["tenant"], records["user"]),
            )
            connection.execute(
                "INSERT INTO agents (tenant_id, id, key, name) VALUES (%s, %s, 'pi', 'Pi')",
                (records["tenant"], records["agent"]),
            )
            connection.execute(
                """INSERT INTO agent_delegations
                   (tenant_id, user_id, agent_id, can_write, expires_at)
                   VALUES (%s, %s, %s, true, now() + interval '1 hour')""",
                (records["tenant"], records["user"], records["agent"]),
            )
            connection.execute(
                """INSERT INTO agent_runs
                   (tenant_id, id, agent_id, session_key, expires_at)
                   VALUES (%s, %s, %s, 'session', now() + interval '1 hour')""",
                (records["tenant"], records["run"], records["agent"]),
            )
            connection.execute(
                "INSERT INTO workspaces (tenant_id, id, key, name) VALUES (%s, %s, 'work', 'Work')",
                (records["tenant"], records["workspace"]),
            )
            connection.execute(
                """INSERT INTO workspace_memberships
                   (tenant_id, workspace_id, user_id, can_read)
                   VALUES (%s, %s, %s, false)""",
                (records["tenant"], records["workspace"], records["user"]),
            )
            connection.execute(
                """INSERT INTO projects (tenant_id, id, workspace_id, key, name)
                   VALUES (%s, %s, %s, 'project', 'Project'),
                          (%s, %s, %s, 'denied', 'Denied')""",
                (
                    records["tenant"], records["project"], records["workspace"],
                    records["tenant"], records["denied_project"], records["workspace"],
                ),
            )
            connection.execute(
                """INSERT INTO project_memberships
                   (tenant_id, project_id, user_id, can_write)
                   VALUES (%s, %s, %s, true)""",
                (records["tenant"], records["project"], records["user"]),
            )
            connection.execute(
                """INSERT INTO compartments (tenant_id, id, path, scope_type, name)
                   VALUES (%s, %s, 'tenant', 'tenant', 'Tenant')""",
                (records["tenant"], records["root_compartment"]),
            )
            connection.execute(
                """INSERT INTO compartments
                   (tenant_id, id, parent_id, path, scope_type, scope_id, name)
                   VALUES (%(tenant)s, %(global)s, %(root)s, 'tenant.global', 'global', NULL, 'Global'),
                          (%(tenant)s, %(user_comp)s, %(root)s, 'tenant.users', 'user', %(user)s, 'User'),
                          (%(tenant)s, %(agent_comp)s, %(root)s, 'tenant.agents', 'agent', %(agent)s, 'Agent'),
                          (%(tenant)s, %(workspace_comp)s, %(root)s, 'tenant.work', 'workspace', %(workspace)s, 'Work')""",
                {
                    "tenant": records["tenant"],
                    "global": records["global_compartment"],
                    "root": records["root_compartment"],
                    "user_comp": records["user_compartment"],
                    "user": records["user"],
                    "agent_comp": records["agent_compartment"],
                    "agent": records["agent"],
                    "workspace_comp": records["workspace_compartment"],
                    "workspace": records["workspace"],
                },
            )
            connection.execute(
                """INSERT INTO compartments
                   (tenant_id, id, parent_id, path, scope_type, scope_id, name)
                   VALUES (%s, %s, %s, 'tenant.work.project', 'project', %s, 'Project')""",
                (
                    records["tenant"], records["project_compartment"],
                    records["workspace_compartment"], records["project"],
                ),
            )
    try:
        yield records
    finally:
        with psycopg.connect(admin_database_url, autocommit=True) as connection:
            connection.execute("DELETE FROM documents WHERE tenant_id = %s", (records["tenant"],))
            connection.execute("DELETE FROM tenants WHERE id = %s", (records["tenant"],))


def test_resolve_context_checks_each_requested_scope(database_url: str, context_records):
    with psycopg.connect(database_url, autocommit=True) as connection:
        context = resolve_context(
            connection,
            context_records["user"],
            context_records["tenant"],
            agent_key="pi",
            session_key="session",
            workspace_key="work",
            project_key="project",
        )

    assert context.agent_id == context_records["agent"]
    assert context.agent_run_id == context_records["run"]
    assert context.agent_can_write
    assert context.workspace_id == context_records["workspace"]
    assert not context.workspace_can_read
    assert context.project_id == context_records["project"]
    assert context.project_can_write


def test_project_access_does_not_follow_from_workspace(database_url: str, context_records):
    with psycopg.connect(database_url, autocommit=True) as connection:
        with pytest.raises(ContextResolutionError):
            resolve_context(
                connection,
                context_records["user"],
                context_records["tenant"],
                workspace_key="work",
                project_key="denied",
            )


def test_workspace_context_requires_workspace_read_access(database_url: str, context_records):
    with psycopg.connect(database_url, autocommit=True) as connection:
        with pytest.raises(ContextResolutionError):
            resolve_context(
                connection,
                context_records["user"],
                context_records["tenant"],
                workspace_key="work",
            )


def test_inactive_tenant_membership_fails_closed(database_url: str, context_records):
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, context_records["tenant"]):
            connection.execute(
                "UPDATE tenant_memberships SET state = 'suspended' WHERE user_id = %s",
                (context_records["user"],),
            )

        with pytest.raises(ContextResolutionError):
            resolve_context(connection, context_records["user"], context_records["tenant"])


def test_agent_hint_requires_session_key(database_url: str, context_records):
    with psycopg.connect(database_url, autocommit=True) as connection:
        with pytest.raises(ValueError, match="supplied together"):
            resolve_context(
                connection,
                context_records["user"],
                context_records["tenant"],
                agent_key="pi",
            )


def test_context_endpoint_issues_user_bound_token(
    monkeypatch: pytest.MonkeyPatch, database_url: str, context_records
):
    monkeypatch.setenv("MEMSYSTEM_DATABASE_URL", database_url)
    monkeypatch.setenv("MEMSYSTEM_API_TOKEN", "secret")
    monkeypatch.setenv("MEMSYSTEM_API_TOKEN_USER_ID", str(context_records["user"]))
    monkeypatch.setenv("MEMSYSTEM_API_TOKEN_EXPIRES_AT", str(int(time.time()) + 60))
    monkeypatch.setenv("MEMSYSTEM_CONTEXT_TOKEN_KEY", "x" * 32)

    with TestClient(mcp.streamable_http_app()) as client:
        response = client.post(
            "/context",
            headers={"Authorization": "Bearer secret"},
            json={"tenantId": str(context_records["tenant"])},
        )

    assert response.status_code == 200
    claims = ContextTokenCodec("x" * 32).verify(
        response.json()["contextToken"], context_records["user"]
    )
    assert claims.tenant_id == context_records["tenant"]
    assert claims.user_id == context_records["user"]


def test_mcp_recall_and_read_use_signed_request_context(
    monkeypatch: pytest.MonkeyPatch, database_url: str, context_records
):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            documents = [
                create_document(
                    connection, claims, context_records["user_compartment"],
                    kind="page", slug=f"adapter-{number}", title="Adapter",
                    markdown="rare_identifier_X9Q7",
                    idempotency_key=f"create-adapter-{number}",
                )
                for number in range(11)
            ]
        resolved = resolve_context(connection, claims.user_id, claims.tenant_id)

    monkeypatch.setenv("MEMSYSTEM_DATABASE_URL", database_url)
    monkeypatch.setenv("MEMSYSTEM_CONTEXT_TOKEN_KEY", "x" * 32)
    monkeypatch.setattr(server_module, "authenticated_user_id", lambda: claims.user_id)
    token = ContextTokenCodec("x" * 32).issue(resolved)
    request = cast(object, SimpleNamespace(headers={"x-memsystem-context": token}))

    first = server_module.memory_recall(request, "rare_identifier_X9Q7", limit=5)
    second = server_module.memory_recall(
        request, "rare_identifier_X9Q7", limit=10, cursor=first["nextCursor"]
    )
    content = server_module.memory_read(request, str(documents[0].document_id))
    monkeypatch.setattr(server_module, "authenticated_user_id", uuid4)
    with pytest.raises(ToolError, match="context is invalid"):
        server_module.memory_read(request, str(documents[0].document_id))

    results = first["results"] + second["results"]
    assert {item["document"] for item in results} == {
        str(document.document_id) for document in documents
    }
    assert first["nextCursor"] is not None
    assert second["nextCursor"] is None
    assert first["contentTrust"] == content["contentTrust"] == "untrusted"
    assert content["markdown"] == "rare_identifier_X9Q7"
    for item in results:
        assert item["sourceEnd"] - item["sourceStart"] == len(item["excerpt"])


def test_mcp_recall_uses_vector_provider_and_falls_back_to_lexical(
    monkeypatch: pytest.MonkeyPatch, database_url: str, context_records, tmp_path
):
    claims = _claims(context_records)

    class DocumentProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            return ([1.0, 0.0] if text == "vector alpha" else [0.0, 1.0]) + [0.0] * 1534

    class QueryProvider:
        def embed_query(self, query: str):
            return [1.0] + [0.0] * 1535

    owner = TurboVecIndex(claims.tenant_id, tmp_path / "mcp-vector.tvim")
    slot = IndexSlot(owner)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            for index, text in enumerate((
                "vector alpha", "lexical target one", "lexical target two", "lexical target three"
            )):
                create_document(
                    connection, claims, context_records["user_compartment"],
                    kind="page", slug=f"hybrid-{index}", title=text, markdown=text,
                    idempotency_key=f"create-hybrid-{index}",
                )
        while process_embedding_job(connection, claims.tenant_id, DocumentProvider()):
            pass
        while process_index_job(connection, claims.tenant_id, slot):
            pass
        resolved = resolve_context(connection, claims.user_id, claims.tenant_id)

    monkeypatch.setenv("MEMSYSTEM_DATABASE_URL", database_url)
    monkeypatch.setenv("MEMSYSTEM_CONTEXT_TOKEN_KEY", "x" * 32)
    monkeypatch.setenv("MEMSYSTEM_VECTOR_RECALL", "true")
    monkeypatch.setattr(server_module, "authenticated_user_id", lambda: claims.user_id)
    monkeypatch.setattr(server_module, "_embedding_provider", lambda: QueryProvider())
    server_module._index_slots[claims.tenant_id] = slot
    request = cast(object, SimpleNamespace(headers={
        "x-memsystem-context": ContextTokenCodec("x" * 32).issue(resolved)
    }))

    hybrid = server_module.memory_recall(request, "target", limit=2)
    assert hybrid["mode"] == "hybrid"
    assert len(hybrid["results"]) == 2
    assert all("vectorScore" in item and "fusedScore" in item for item in hybrid["results"])
    assert hybrid["nextCursor"] is None

    monkeypatch.setattr(server_module, "search_vector", lambda *args, **kwargs: [])
    empty_vector = server_module.memory_recall(request, "target", limit=2)
    assert empty_vector["mode"] == "lexical"
    assert empty_vector["nextCursor"] is not None

    class FailingQueryProvider:
        def embed_query(self, query: str):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(server_module, "_embedding_provider", lambda: FailingQueryProvider())
    fallback = server_module.memory_recall(request, "target", limit=2)
    assert fallback["mode"] == "lexical"
    assert fallback["nextCursor"] is not None
    assert claims.tenant_id in server_module._index_slots
    server_module._drop_index_slot(claims.tenant_id)


def test_direct_user_gets_private_write_and_global_read(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            permissions = compartment_permissions(connection, claims)
            with pytest.raises(AuthorizationError):
                require_compartment_access(
                    connection, claims, context_records["global_compartment"], "write"
                )

    assert permissions[context_records["global_compartment"]].can_read
    assert not permissions[context_records["global_compartment"]].can_write
    assert permissions[context_records["user_compartment"]].can_write


def test_agent_context_rechecks_delegation_and_exact_run(database_url: str, context_records):
    claims = _claims(context_records, agent=True)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            permissions = compartment_permissions(connection, claims)
            assert permissions[context_records["agent_compartment"]].can_write
            assert permissions[context_records["user_compartment"]].can_write

            connection.execute(
                "UPDATE agent_runs SET state = 'revoked' WHERE id = %s",
                (context_records["run"],),
            )
            permissions = compartment_permissions(connection, claims)

    assert permissions == {}


def test_revoked_delegation_removes_agent_and_user_scopes(database_url: str, context_records):
    claims = _claims(context_records, agent=True)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            connection.execute(
                "UPDATE agent_delegations SET revoked_at = now() WHERE agent_id = %s",
                (context_records["agent"],),
            )
            permissions = compartment_permissions(connection, claims)

    assert permissions == {}


def test_project_permission_does_not_require_workspace_permission(
    database_url: str, context_records
):
    claims = _claims(context_records, workspace=True, project=True)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            permissions = compartment_permissions(connection, claims)

    assert context_records["workspace_compartment"] not in permissions
    assert permissions[context_records["project_compartment"]].can_write


def test_project_permission_ends_when_workspace_is_archived(
    database_url: str, context_records
):
    claims = _claims(context_records, workspace=True, project=True)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            connection.execute(
                "UPDATE workspaces SET archived_at = now() WHERE id = %s",
                (context_records["workspace"],),
            )
            permissions = compartment_permissions(connection, claims)

    assert context_records["project_compartment"] not in permissions


def test_stale_context_loses_access_after_membership_suspension(
    database_url: str, context_records
):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            connection.execute(
                "UPDATE tenant_memberships SET state = 'suspended' WHERE user_id = %s",
                (context_records["user"],),
            )
            assert compartment_permissions(connection, claims) == {}


def test_document_create_read_update_and_history(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            created = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="guide", title="Guide", markdown="First",
                idempotency_key="create-guide",
            )
            updated = update_document(
                connection, claims, created.document_id,
                expected_revision=1, title="Guide", markdown="Second",
                idempotency_key="update-guide",
            )
            current = read_document(connection, claims, created.document_id)
            original = read_document(connection, claims, created.document_id, revision=1)
            chunks = connection.execute(
                """SELECT r.revision, c.text, c.chunk_profile,
                          c.search_document @@ plainto_tsquery('simple', r.title)
                   FROM chunks c
                   JOIN document_revisions r
                     ON (r.tenant_id, r.document_id, r.id) =
                        (c.tenant_id, c.document_id, c.revision_id)
                   WHERE c.document_id = %s ORDER BY r.revision, c.position""",
                (created.document_id,),
            ).fetchall()

    assert created.revision == 1
    assert updated.revision == 2
    assert current.markdown == "Second"
    assert original.markdown == "First"
    assert chunks == [
        (1, "First", "markdown-whitespace-v1-512-64", True),
        (2, "Second", "markdown-whitespace-v1-512-64", True),
    ]


def test_embedding_job_stores_normalized_vector(database_url: str, context_records):
    claims = _claims(context_records)

    class FakeProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            assert text == "Embed me"
            return [1.0] * self.dimension

    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="embedding", title="Embedding", markdown="Embed me",
                idempotency_key="create-embedding",
            )
            embed_job = connection.execute(
                """SELECT j.chunk_id, j.vector_id, j.profile, j.state
                   FROM jobs j
                   JOIN chunks c ON (c.tenant_id, c.id) = (j.tenant_id, j.chunk_id)
                   WHERE j.document_id = %s AND j.kind = 'embed'""",
                (document.document_id,),
            ).fetchone()

        assert embed_job[2:] == ("qwen3-4b-1536-v1", "pending")
        abandoned = claim_embedding_job(connection, claims.tenant_id)
        assert abandoned is not None
        with tenant_transaction(connection, claims.tenant_id):
            connection.execute(
                "UPDATE jobs SET claimed_at = now() - interval '6 minutes' WHERE id = %s",
                (abandoned.id,),
            )
        assert process_embedding_job(connection, claims.tenant_id, FakeProvider()) is True

        with tenant_transaction(connection, claims.tenant_id):
            chunk = connection.execute(
                """SELECT embedding, embedding_model, embedding_version,
                          embedding_dimension, embedding_state
                   FROM chunks WHERE id = %s""",
                (embed_job[0],),
            ).fetchone()
            index_jobs = connection.execute(
                """SELECT vector_id, profile, state FROM jobs
                   WHERE document_id = %s AND kind = 'index_add'""",
                (document.document_id,),
            ).fetchall()
            source_job = connection.execute(
                "SELECT state, attempts FROM jobs WHERE id = %s",
                (abandoned.id,),
            ).fetchone()

    assert np.linalg.norm(chunk[0].to_numpy()) == pytest.approx(1.0)
    assert chunk[1:] == ("Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b", 1536, "ready")
    assert index_jobs == [(embed_job[1], "qwen3-4b-1536-v1", "pending")]
    assert source_job == ("complete", 1)


def test_reclaimed_embedding_job_fences_stale_worker(database_url: str, context_records):
    claims = _claims(context_records)
    started, release = Event(), Event()

    class SlowProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            started.set()
            assert release.wait(5)
            return [0.0, 1.0] + [0.0] * 1534

    class FastProvider(SlowProvider):
        def embed(self, text: str):
            return [1.0] + [0.0] * 1535

    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="lease-fence", title="Lease fence",
                markdown="Fence me", idempotency_key="create-lease-fence",
            )

        def run_slow_worker():
            with psycopg.connect(database_url, autocommit=True) as worker_connection:
                return process_embedding_job(
                    worker_connection, claims.tenant_id, SlowProvider()
                )

        with ThreadPoolExecutor(max_workers=1) as executor:
            stale_worker = executor.submit(run_slow_worker)
            assert started.wait(5)
            with tenant_transaction(connection, claims.tenant_id):
                connection.execute(
                    """UPDATE jobs SET claimed_at = now() - interval '6 minutes'
                       WHERE document_id = %s AND kind = 'embed'""",
                    (document.document_id,),
                )
            assert process_embedding_job(connection, claims.tenant_id, FastProvider()) is True
            release.set()
            assert stale_worker.result() is True

        with tenant_transaction(connection, claims.tenant_id):
            result = connection.execute(
                """SELECT c.embedding,
                          (SELECT count(*) FROM jobs index_job
                           WHERE index_job.tenant_id = c.tenant_id
                             AND index_job.chunk_id = c.id
                             AND index_job.kind = 'index_add'),
                          embed_job.attempts
                   FROM chunks c
                   JOIN jobs embed_job
                     ON (embed_job.tenant_id, embed_job.chunk_id) = (c.tenant_id, c.id)
                    AND embed_job.kind = 'embed'
                   WHERE c.document_id = %s""",
                (document.document_id,),
            ).fetchone()

    assert result[0].to_list()[:2] == [1.0, 0.0]
    assert result[1:] == (1, 1)


def test_embedding_job_skips_deleted_document(database_url: str, context_records):
    claims = _claims(context_records)

    class RejectingProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            raise AssertionError("deleted content reached the provider")

    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="deleted-embedding", title="Deleted",
                markdown="Do not embed", idempotency_key="create-deleted-embedding",
            )
            soft_delete_document(
                connection, claims, document.document_id, expected_revision=1,
                idempotency_key="delete-before-embedding",
            )

        assert process_embedding_job(connection, claims.tenant_id, RejectingProvider()) is True

        with tenant_transaction(connection, claims.tenant_id):
            states = connection.execute(
                """SELECT j.kind, j.state, c.embedding
                   FROM jobs j
                   LEFT JOIN chunks c ON (c.tenant_id, c.id) = (j.tenant_id, j.chunk_id)
                   WHERE j.document_id = %s ORDER BY j.kind""",
                (document.document_id,),
            ).fetchall()

    assert states == [("embed", "complete", None)]


def test_embedding_job_stops_after_bounded_failures(database_url: str, context_records):
    claims = _claims(context_records)

    class FailingProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            raise RuntimeError("provider unavailable")

    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="failed-embedding", title="Failed",
                markdown="Retry me", idempotency_key="create-failed-embedding",
            )

        for _ in range(5):
            assert process_embedding_job(connection, claims.tenant_id, FailingProvider()) is True
            with tenant_transaction(connection, claims.tenant_id):
                connection.execute(
                    "UPDATE jobs SET next_attempt_at = now() WHERE document_id = %s",
                    (document.document_id,),
                )
        assert process_embedding_job(connection, claims.tenant_id, FailingProvider()) is False

        with tenant_transaction(connection, claims.tenant_id):
            state = connection.execute(
                """SELECT j.state, j.attempts, j.error, c.embedding_state
                   FROM jobs j
                   JOIN chunks c ON (c.tenant_id, c.id) = (j.tenant_id, j.chunk_id)
                   WHERE j.document_id = %s AND j.kind = 'embed'""",
                (document.document_id,),
            ).fetchone()

    assert state == ("failed", 5, "provider unavailable", "failed")


def test_index_worker_adds_and_removes_current_vectors(
    database_url: str, context_records, tmp_path
):
    claims = _claims(context_records)

    class FakeProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            return [1.0] + [0.0] * 1535

    owner = TurboVecIndex(claims.tenant_id, tmp_path / "generation-1.tvim")
    slot = IndexSlot(owner)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="indexed", title="Indexed", markdown="First",
                idempotency_key="create-indexed",
            )
            first_vector_id = connection.execute(
                "SELECT vector_id FROM chunks WHERE document_id = %s",
                (document.document_id,),
            ).fetchone()[0]

        assert process_embedding_job(connection, claims.tenant_id, FakeProvider()) is True
        assert process_index_job(connection, claims.tenant_id, slot) is True
        assert owner.index.contains(first_vector_id)

        with tenant_transaction(connection, claims.tenant_id):
            update_document(
                connection, claims, document.document_id, expected_revision=1,
                title="Indexed", markdown="Second", idempotency_key="update-indexed",
            )
            second_vector_id = connection.execute(
                """SELECT c.vector_id
                   FROM chunks c
                   JOIN documents d
                     ON (d.tenant_id, d.id, d.current_revision_id) =
                        (c.tenant_id, c.document_id, c.revision_id)
                   WHERE c.document_id = %s""",
                (document.document_id,),
            ).fetchone()[0]

        assert process_embedding_job(connection, claims.tenant_id, FakeProvider()) is True
        while process_index_job(connection, claims.tenant_id, slot):
            pass
        assert not owner.index.contains(first_vector_id)
        assert owner.index.contains(second_vector_id)

        with tenant_transaction(connection, claims.tenant_id):
            soft_delete_document(
                connection, claims, document.document_id, expected_revision=2,
                idempotency_key="delete-indexed",
            )
        assert process_index_job(connection, claims.tenant_id, slot) is True

        with tenant_transaction(connection, claims.tenant_id):
            state = connection.execute(
                """SELECT c.indexed_revision, c.indexed_generation,
                          s.generation, s.generation_path, length(s.checksum)
                   FROM chunks c
                   JOIN vector_index_state s ON s.tenant_id = c.tenant_id
                   WHERE c.vector_id = %s AND s.profile = 'qwen3-4b-1536-v1'""",
                (second_vector_id,),
            ).fetchone()
            jobs = connection.execute(
                """SELECT kind::text, state::text FROM jobs
                   WHERE profile = 'qwen3-4b-1536-v1'
                     AND kind IN ('index_add', 'index_remove')
                   ORDER BY created_at, id"""
            ).fetchall()

    assert not owner.index.contains(second_vector_id)
    assert state == (None, None, 1, str(owner.path), 64)
    assert jobs == [
        ("index_add", "complete"),
        ("index_remove", "complete"),
        ("index_add", "complete"),
        ("index_remove", "complete"),
    ]
    owner.close()


def test_rebuild_replays_commits_and_swaps_generation(
    database_url: str, context_records, tmp_path, monkeypatch
):
    claims = _claims(context_records)

    class FakeProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            return [1.0] + [0.0] * 1535

    owner = TurboVecIndex(claims.tenant_id, tmp_path / "generation-1.tvim")
    slot = IndexSlot(owner)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="rebuild", title="Rebuild", markdown="First",
                idempotency_key="create-rebuild",
            )
            first_vector_id = connection.execute(
                "SELECT vector_id FROM chunks WHERE document_id = %s",
                (document.document_id,),
            ).fetchone()[0]
        assert process_embedding_job(connection, claims.tenant_id, FakeProvider()) is True
        assert process_index_job(connection, claims.tenant_id, slot) is True

        original_sync = TurboVecIndex.sync
        build_started, continue_build = Event(), Event()

        def pause_first_build(candidate):
            if candidate.path.suffix == ".building" and not build_started.is_set():
                build_started.set()
                assert continue_build.wait(5)
            return original_sync(candidate)

        monkeypatch.setattr(TurboVecIndex, "sync", pause_first_build)

        def run_rebuild():
            with psycopg.connect(database_url, autocommit=True) as rebuild_connection:
                return rebuild_index(
                    rebuild_connection, claims.tenant_id, slot, tmp_path / "generations"
                )

        with ThreadPoolExecutor(max_workers=1) as executor:
            rebuilding = executor.submit(run_rebuild)
            assert build_started.wait(5)
            with tenant_transaction(connection, claims.tenant_id):
                update_document(
                    connection, claims, document.document_id, expected_revision=1,
                    title="Rebuild", markdown="Second", idempotency_key="update-rebuild",
                )
                second_vector_id = connection.execute(
                    """SELECT c.vector_id
                       FROM chunks c
                       JOIN documents d
                         ON (d.tenant_id, d.id, d.current_revision_id) =
                            (c.tenant_id, c.document_id, c.revision_id)
                       WHERE c.document_id = %s""",
                    (document.document_id,),
                ).fetchone()[0]
            assert process_embedding_job(
                connection, claims.tenant_id, FakeProvider()
            ) is True
            continue_build.set()
            rebuilt = rebuilding.result()

        with tenant_transaction(connection, claims.tenant_id):
            state = connection.execute(
                """SELECT generation, generation_path, job_high_water_mark,
                          (SELECT max(enqueue_seq) FROM jobs WHERE tenant_id = %s)
                   FROM vector_index_state
                   WHERE tenant_id = %s AND profile = 'qwen3-4b-1536-v1'""",
                (claims.tenant_id, claims.tenant_id),
            ).fetchone()
            markers = connection.execute(
                """SELECT vector_id, indexed_revision, indexed_generation
                   FROM chunks WHERE document_id = %s ORDER BY vector_id""",
                (document.document_id,),
            ).fetchall()
            results = search_vector(
                connection, claims, slot, [1.0] + [0.0] * 1535
            )

    assert slot.current(claims.tenant_id) is rebuilt
    assert not owner.healthy
    assert rebuilt.generation == 2
    assert not rebuilt.index.contains(first_vector_id)
    assert rebuilt.index.contains(second_vector_id)
    assert state == (2, str(rebuilt.path), state[3], state[3])
    assert markers == [
        (first_vector_id, None, None),
        (second_vector_id, 2, 2),
    ]
    assert [item.document_id for item in results] == [document.document_id]
    orphan_stem = f"{claims.tenant_id}-qwen3-4b-1536-v1-g99-orphan"
    published_pending = rebuilt.path.with_suffix(".pending")
    published_pending.write_bytes(b"published")
    orphan_pending = rebuilt.path.parent / f"{orphan_stem}.pending"
    orphan_final = rebuilt.path.parent / f"{orphan_stem}.tvim"
    orphan_temporary = rebuilt.path.parent / f".{orphan_stem}.building"
    active_stem = f"{claims.tenant_id}-qwen3-4b-1536-v1-g100-active"
    active_pending = rebuilt.path.parent / f"{active_stem}.pending"
    active_final = rebuilt.path.parent / f"{active_stem}.tvim"
    active_temporary = rebuilt.path.parent / f".{active_stem}.building"
    for path in (
        orphan_pending, orphan_final, orphan_temporary,
        active_pending, active_final, active_temporary,
    ):
        path.write_bytes(b"orphan")
    active_lock = active_pending.open("r+b")
    fcntl.flock(active_lock, fcntl.LOCK_EX)
    rebuilt.close()
    with psycopg.connect(database_url, autocommit=True) as connection:
        restarted = load_or_rebuild_index_slot(
            connection, claims.tenant_id, rebuilt.path.parent
        )
    assert restarted.current(claims.tenant_id).index.contains(second_vector_id)
    assert rebuilt.path.exists()
    assert rebuilt.path.with_suffix(rebuilt.path.suffix + ".lock").exists()
    assert not published_pending.exists()
    assert not orphan_pending.exists()
    assert not orphan_final.exists()
    assert not orphan_temporary.exists()
    assert active_pending.exists() and active_final.exists() and active_temporary.exists()
    restarted.current(claims.tenant_id).close()
    fcntl.flock(active_lock, fcntl.LOCK_UN)
    active_lock.close()
    with psycopg.connect(database_url, autocommit=True) as connection:
        restarted = load_or_rebuild_index_slot(
            connection, claims.tenant_id, rebuilt.path.parent
        )
    assert not active_pending.exists()
    assert not active_final.exists()
    assert not active_temporary.exists()
    restarted.current(claims.tenant_id).close()


def test_vector_search_uses_authorized_allowlist_and_handles_stale_ids(
    database_url: str, context_records, tmp_path
):
    claims = _claims(context_records)
    project_claims = _claims(context_records, workspace=True, project=True)

    class FakeProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            axis = 0 if text == "User vector" else 1
            return [0.0] * axis + [1.0] + [0.0] * (1535 - axis)

    owner = TurboVecIndex(claims.tenant_id, tmp_path / "search.tvim")
    slot = IndexSlot(owner)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            user_document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="user-vector", title="User vector",
                markdown="User vector", idempotency_key="create-user-vector",
            )
            project_document = create_document(
                connection, project_claims, context_records["project_compartment"],
                kind="page", slug="project-vector", title="Project vector",
                markdown="Project vector", idempotency_key="create-project-vector",
            )

        while process_embedding_job(connection, claims.tenant_id, FakeProvider()):
            pass
        while process_index_job(connection, claims.tenant_id, slot):
            pass

        with pytest.raises(RuntimeError, match="active tenant transaction"):
            search_vector(connection, claims, slot, [1.0] + [0.0] * 1535)

        with tenant_transaction(connection, claims.tenant_id):
            user_results = search_vector(
                connection, claims, slot, [0.0, 1.0] + [0.0] * 1534
            )
            project_results = search_vector(
                connection, project_claims, slot,
                [0.0, 1.0] + [0.0] * 1534, project_only=True,
            )
            user_vector_id = connection.execute(
                "SELECT vector_id FROM chunks WHERE document_id = %s",
                (user_document.document_id,),
            ).fetchone()[0]

        assert [item.document_id for item in user_results] == [user_document.document_id]
        assert [item.document_id for item in project_results] == [project_document.document_id]

        with tenant_transaction(connection, claims.tenant_id):
            connection.execute(
                "UPDATE tenant_memberships SET state = 'suspended' WHERE user_id = %s",
                (claims.user_id,),
            )
            assert search_vector(
                connection, claims, slot, [1.0] + [0.0] * 1535
            ) == []
            connection.execute(
                "UPDATE tenant_memberships SET state = 'active' WHERE user_id = %s",
                (claims.user_id,),
            )

        owner.remove(user_vector_id)
        with tenant_transaction(connection, claims.tenant_id):
            assert search_vector(
                connection, claims, slot, [1.0] + [0.0] * 1535
            ) == []
            connection.execute(
                "UPDATE chunks SET indexed_generation = NULL WHERE vector_id = %s",
                (user_vector_id,),
            )
            assert search_vector(
                connection, claims, slot, [1.0] + [0.0] * 1535
            ) == []

    owner.close()


@pytest.mark.parametrize(
    "change", [
        "none", "deleted", "expired", "superseded", "revoked", "unindexed", "wrong-profile",
        "oversized", "duplicate", "unauthorized",
    ]
)
def test_vector_search_reranks_revalidated_candidates(
    database_url: str, context_records, tmp_path, monkeypatch, change
):
    claims = _claims(context_records)
    project_claims = _claims(context_records, workspace=True, project=True)

    class FakeProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            return ([0.6, 0.8] if text == "low" else [1.0, 0.0]) + [0.0] * 1534

    owner = TurboVecIndex(claims.tenant_id, tmp_path / "rerank.tvim")
    slot = IndexSlot(owner)
    try:
        with psycopg.connect(database_url, autocommit=True) as connection:
            documents = {}
            with tenant_transaction(connection, claims.tenant_id):
                for name in ("low", "high", "tie", "denied"):
                    documents[name] = create_document(
                        connection, project_claims if name == "denied" else claims,
                        context_records["project_compartment" if name == "denied" else "user_compartment"],
                        kind="page", slug=name, title=name, markdown=name,
                        idempotency_key=f"create-{name}",
                    )
            while process_embedding_job(connection, claims.tenant_id, FakeProvider()):
                pass
            while process_index_job(connection, claims.tenant_id, slot):
                pass

            with tenant_transaction(connection, claims.tenant_id):
                ids = dict(connection.execute(
                    "SELECT text, vector_id FROM chunks WHERE tenant_id = %s",
                    (claims.tenant_id,),
                ).fetchall())

                def candidates(query, limit, allowlist):
                    assert set(allowlist) == {ids[name] for name in ("low", "high", "tie")}
                    assert limit == 3
                    if change == "deleted":
                        soft_delete_document(
                            connection, claims, documents["high"].document_id,
                            expected_revision=1, idempotency_key="delete-high",
                        )
                    elif change == "expired":
                        connection.execute(
                            "UPDATE documents SET expires_at = now() WHERE id = %s",
                            (documents["high"].document_id,),
                        )
                    elif change == "superseded":
                        update_document(
                            connection, claims, documents["high"].document_id,
                            expected_revision=1, title="high", markdown="new",
                            idempotency_key="update-high",
                        )
                    elif change == "revoked":
                        connection.execute(
                            "UPDATE tenant_memberships SET state = 'suspended' WHERE user_id = %s",
                            (claims.user_id,),
                        )
                    elif change == "unindexed":
                        connection.execute(
                            "UPDATE chunks SET indexed_generation = NULL WHERE vector_id = %s",
                            (ids["high"],),
                        )
                    elif change == "wrong-profile":
                        connection.execute(
                            "UPDATE chunks SET embedding_version = 'other' WHERE vector_id = %s",
                            (ids["high"],),
                        )
                    names = {
                        "oversized": ("low", "tie", "high", "denied"),
                        "duplicate": ("low", "tie", "tie"),
                        "unauthorized": ("denied", "low", "tie"),
                    }.get(change, ("low", "tie", "high"))
                    return (
                        np.array([[99.0 - index for index in range(len(names))]]),
                        np.array([[ids[name] for name in names]]),
                    )

                monkeypatch.setattr(owner, "search", candidates)
                if change == "oversized":
                    with pytest.raises(RuntimeError, match="too many candidates"):
                        search_vector(connection, claims, slot, [2.0] + [0.0] * 1535, limit=2)
                    return
                results = search_vector(connection, claims, slot, [2.0] + [0.0] * 1535, limit=2)
                if change == "revoked":
                    assert results == []
                else:
                    expected = {"high", "tie"} if change == "none" else {"tie", "low"}
                    assert {item.document_id for item in results} == {
                        documents[name].document_id for name in expected
                    }
                    assert [item.score for item in results] == pytest.approx(
                        [1.0, 1.0] if change == "none" else [1.0, 0.6]
                    )
                    assert results == sorted(results, key=lambda item: (-item.score, item.chunk_id))
    finally:
        owner.close()


def test_index_worker_rejects_missing_active_generation(
    database_url: str, context_records, tmp_path
):
    claims = _claims(context_records)

    class FakeProvider:
        provider = "transformers"
        model = "Qwen/Qwen3-Embedding-4B"
        version = "5cf2132abc99cad020ac570b19d031efec650f2b"
        dimension = 1536

        def embed(self, text: str):
            return [1.0] + [0.0] * 1535

    owner = TurboVecIndex(claims.tenant_id, tmp_path / "missing.tvim")
    slot = IndexSlot(owner)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            first = _create_page(
                connection, claims, context_records["user_compartment"], "first-index"
            )
        assert process_embedding_job(connection, claims.tenant_id, FakeProvider()) is True
        assert process_index_job(connection, claims.tenant_id, slot) is True
        owner.path.write_bytes(b"broken")

        with tenant_transaction(connection, claims.tenant_id):
            second = _create_page(
                connection, claims, context_records["user_compartment"], "second-index"
            )
        assert process_embedding_job(connection, claims.tenant_id, FakeProvider()) is True
        assert process_index_job(connection, claims.tenant_id, slot) is True

        with tenant_transaction(connection, claims.tenant_id):
            state = connection.execute(
                """SELECT state, attempts, error FROM jobs
                   WHERE document_id = %s AND kind = 'index_add'""",
                (second.document_id,),
            ).fetchone()
            indexed = connection.execute(
                """SELECT indexed_generation FROM chunks
                   WHERE document_id IN (%s, %s) ORDER BY document_id""",
                (first.document_id, second.document_id),
            ).fetchall()

        owner.close()
        recovered_slot = load_or_rebuild_index_slot(
            connection, claims.tenant_id, tmp_path / "recovered"
        )
        rebuilt = recovered_slot.current(claims.tenant_id)
        with tenant_transaction(connection, claims.tenant_id):
            recovered = connection.execute(
                """SELECT count(*) FROM chunks
                   WHERE document_id IN (%s, %s) AND indexed_generation = %s""",
                (first.document_id, second.document_id, rebuilt.generation),
            ).fetchone()[0]

    assert state == ("failed", 1, "active index checksum does not match PostgreSQL")
    assert sorted(row[0] for row in indexed if row[0] is not None) == [1]
    assert recovered == 2
    assert len(rebuilt.index) == 2
    rebuilt.close()


def test_abandoned_index_job_stops_after_maximum_attempts(
    database_url: str, context_records
):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            job_id = connection.execute(
                """INSERT INTO jobs (tenant_id, vector_id, profile, kind)
                   VALUES (%s, 9000000000000000, 'qwen3-4b-1536-v1', 'index_remove')
                   RETURNING id""",
                (claims.tenant_id,),
            ).fetchone()[0]

        job = claim_index_job(connection, claims.tenant_id)
        assert job is not None
        for attempts in range(1, 6):
            with tenant_transaction(connection, claims.tenant_id):
                connection.execute(
                    "UPDATE jobs SET claimed_at = now() - interval '6 minutes' WHERE id = %s",
                    (job_id,),
                )
            job = claim_index_job(connection, claims.tenant_id)
            assert job is not None and job.attempts == attempts

        with tenant_transaction(connection, claims.tenant_id):
            connection.execute(
                "UPDATE jobs SET claimed_at = now() - interval '6 minutes' WHERE id = %s",
                (job_id,),
            )
        assert claim_index_job(connection, claims.tenant_id) is None

        with tenant_transaction(connection, claims.tenant_id):
            state = connection.execute(
                "SELECT state, attempts, error FROM jobs WHERE id = %s", (job_id,)
            ).fetchone()

    assert state == ("failed", 5, "claim lease expired after maximum attempts")


def test_document_content_uses_byte_bounded_revision_cursors(database_url: str, context_records):
    claims = _claims(context_records)
    markdown = "é" * 80
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="paged", title="Paged", markdown=markdown,
                idempotency_key="create-paged",
            )
            with pytest.raises(ValueError, match="max_bytes"):
                read_document_content(
                    connection, claims, document.document_id, max_bytes=1
                )
            first = read_document_content(
                connection, claims, document.document_id, max_bytes=100
            )
            second = read_document_content(
                connection, claims, document.document_id,
                cursor=first.next_cursor, max_bytes=100,
            )
            update_document(
                connection, claims, document.document_id, expected_revision=1,
                title="Paged", markdown="new", idempotency_key="update-paged",
            )
            with pytest.raises(ValueError, match="invalid cursor"):
                read_document_content(
                    connection, claims, document.document_id,
                    cursor=first.next_cursor, max_bytes=100,
                )

    assert len(first.markdown.encode()) <= 100
    assert first.markdown + second.markdown == markdown
    assert second.next_cursor is None


def test_large_heading_can_be_stored_as_bounded_chunks(database_url: str, context_records):
    claims = _claims(context_records)
    markdown = "# " + "x" * 848_000
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="large-heading", title="Large heading", markdown=markdown,
                idempotency_key="create-large-heading",
            )
            chunk_count = connection.execute(
                "SELECT count(*) FROM chunks WHERE document_id = %s",
                (document.document_id,),
            ).fetchone()[0]

    assert chunk_count > 1


def test_stale_document_update_preserves_current_revision(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="conflict", title="Conflict", markdown="One",
                idempotency_key="create-conflict",
            )
            update_document(
                connection, claims, document.document_id,
                expected_revision=1, title="Conflict", markdown="Two",
                idempotency_key="update-conflict",
            )
            with pytest.raises(RevisionConflict) as error:
                update_document(
                    connection, claims, document.document_id,
                    expected_revision=1, title="Conflict", markdown="Stale",
                    idempotency_key="stale-update",
                )
            revision_count = connection.execute(
                "SELECT count(*) FROM document_revisions WHERE document_id = %s",
                (document.document_id,),
            ).fetchone()[0]
            stale_key_count = connection.execute(
                "SELECT count(*) FROM mutation_idempotency WHERE key = 'stale-update'"
            ).fetchone()[0]

    assert error.value.current_revision == 2
    assert revision_count == 2
    assert stale_key_count == 0


def test_soft_deleted_document_is_unreadable(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="deleted", title="Deleted", markdown="Content",
                idempotency_key="create-deleted",
            )
            soft_delete_document(
                connection, claims, document.document_id,
                expected_revision=1, idempotency_key="delete-document",
            )
            with pytest.raises(DocumentNotFound):
                read_document(connection, claims, document.document_id)
            assert connection.execute(
                "SELECT deleted_at IS NOT NULL FROM documents WHERE id = %s",
                (document.document_id,),
            ).fetchone() == (True,)


def test_document_read_does_not_disclose_unauthorized_scope(database_url: str, context_records):
    project_claims = _claims(context_records, workspace=True, project=True)
    user_claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, user_claims.tenant_id):
            document = create_document(
                connection, project_claims, context_records["project_compartment"],
                kind="page", slug="project-only", title="Project", markdown="Private",
                idempotency_key="create-project",
            )
            with pytest.raises(DocumentNotFound):
                read_document(connection, user_claims, document.document_id)


def test_expired_document_is_unreadable(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="expired", title="Expired", markdown="Content",
                idempotency_key="create-expired",
            )
            connection.execute(
                "UPDATE documents SET expires_at = now() - interval '1 second' WHERE id = %s",
                (document.document_id,),
            )
            with pytest.raises(DocumentNotFound):
                read_document(connection, claims, document.document_id)


def test_agent_revision_records_agent_author(database_url: str, context_records):
    claims = _claims(context_records, agent=True)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = create_document(
                connection, claims, context_records["agent_compartment"],
                kind="page", slug="agent-note", title="Agent note", markdown="Content",
                idempotency_key="create-agent-note",
            )
            author = connection.execute(
                """SELECT created_by_type::text, created_by_id
                   FROM document_revisions WHERE document_id = %s""",
                (document.document_id,),
            ).fetchone()

    assert author == ("agent", context_records["agent"])


def test_repeated_create_returns_one_result(database_url: str, context_records):
    claims = _claims(context_records)
    arguments = {
        "kind": "page", "slug": "once", "title": "Once", "markdown": "Content",
        "idempotency_key": "same-create",
    }
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            first = create_document(
                connection, claims, context_records["user_compartment"], **arguments
            )
            second = create_document(
                connection, claims, context_records["user_compartment"], **arguments
            )
            document_count = connection.execute(
                "SELECT count(*) FROM documents WHERE compartment_id = %s",
                (context_records["user_compartment"],),
            ).fetchone()[0]
            with pytest.raises(IdempotencyConflict):
                create_document(
                    connection, claims, context_records["user_compartment"],
                    **{**arguments, "title": "Different"},
                )

    assert first == second
    assert document_count == 1


def test_update_and_delete_replay_without_new_changes(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            created = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="replay", title="Replay", markdown="One",
                idempotency_key="create-replay",
            )
            update_args = {
                "expected_revision": 1, "title": "Replay", "markdown": "Two",
                "idempotency_key": "update-replay",
            }
            first_update = update_document(
                connection, claims, created.document_id, **update_args
            )
            second_update = update_document(
                connection, claims, created.document_id, **update_args
            )
            first_delete = soft_delete_document(
                connection, claims, created.document_id,
                expected_revision=2, idempotency_key="delete-replay",
            )
            second_delete = soft_delete_document(
                connection, claims, created.document_id,
                expected_revision=2, idempotency_key="delete-replay",
            )
            revision_count = connection.execute(
                "SELECT count(*) FROM document_revisions WHERE document_id = %s",
                (created.document_id,),
            ).fetchone()[0]

    assert first_update == second_update
    assert first_delete == second_delete
    assert revision_count == 2


def test_concurrent_create_with_one_key_creates_one_document(database_url: str, context_records):
    claims = _claims(context_records)

    def create_once():
        with psycopg.connect(database_url, autocommit=True) as connection:
            with tenant_transaction(connection, claims.tenant_id):
                return create_document(
                    connection, claims, context_records["user_compartment"],
                    kind="page", slug="concurrent", title="Concurrent", markdown="Content",
                    idempotency_key="concurrent-create",
                )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: create_once(), range(2)))

    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document_count = connection.execute(
                "SELECT count(*) FROM documents WHERE compartment_id = %s",
                (context_records["user_compartment"],),
            ).fetchone()[0]

    assert results[0] == results[1]
    assert document_count == 1


def test_concurrent_updates_cannot_replace_newer_revision(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = _create_page(
                connection, claims, context_records["user_compartment"], "update-race"
            )

    def attempt_update(number):
        try:
            with psycopg.connect(database_url, autocommit=True) as connection:
                with tenant_transaction(connection, claims.tenant_id):
                    update_document(
                        connection, claims, document.document_id,
                        expected_revision=1, title="Race", markdown=str(number),
                        idempotency_key=f"race-update-{number}",
                    )
            return True
        except RevisionConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt_update, range(2)))

    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            current = read_document(connection, claims, document.document_id)
            revision_count = connection.execute(
                "SELECT count(*) FROM document_revisions WHERE document_id = %s",
                (document.document_id,),
            ).fetchone()[0]

    assert results.count(True) == 1
    assert current.revision == revision_count == 2


def test_collection_children_use_bounded_keyset_pages(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            collection = create_document(
                connection, claims, context_records["user_compartment"],
                kind="collection", slug="collection", title="Collection", markdown="",
                idempotency_key="create-collection",
            )
            for slug in ("c", "a", "b"):
                create_document(
                    connection, claims, context_records["user_compartment"],
                    kind="page", slug=slug, title=slug, markdown="Content",
                    parent_id=collection.document_id, idempotency_key=f"create-child-{slug}",
                )
            first = read_collection_children(
                connection, claims, collection.document_id, limit=2
            )
            second = read_collection_children(
                connection, claims, collection.document_id,
                limit=2, cursor=first.next_cursor,
            )
            with pytest.raises(ValueError, match="invalid cursor"):
                read_collection_children(
                    connection, claims, collection.document_id, cursor="not-base64!"
                )

    assert [item.slug for item in first.items] == ["a", "b"]
    assert first.next_cursor is not None
    assert [item.slug for item in second.items] == ["c"]
    assert second.next_cursor is None


def test_revision_history_uses_newest_first_pages(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            document = _create_page(
                connection, claims, context_records["user_compartment"], "history"
            )
            for revision in (2, 3):
                update_document(
                    connection, claims, document.document_id,
                    expected_revision=revision - 1,
                    title=f"Revision {revision}", markdown=f"Content {revision}",
                    idempotency_key=f"update-history-{revision}",
                )
            first = read_revision_history(
                connection, claims, document.document_id, limit=2
            )
            second = read_revision_history(
                connection, claims, document.document_id,
                limit=2, cursor=first.next_cursor,
            )

    assert [item.revision for item in first.items] == [3, 2]
    assert first.next_cursor is not None
    assert [item.revision for item in second.items] == [1]
    assert second.next_cursor is None


def test_lexical_search_ranks_fields_and_filters_current_authorized_chunks(
    database_url: str, context_records
):
    claims = _claims(context_records)
    project_claims = _claims(context_records, workspace=True, project=True)
    filler = "word " * 513
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            title = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="rank-title", title="Needle", markdown=filler,
                idempotency_key="create-rank-title",
            )
            heading = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="rank-heading", title="Other", markdown="# Needle\n" + filler,
                idempotency_key="create-rank-heading",
            )
            body = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="rank-body", title="Other", markdown="# Section\nneedle " + filler,
                idempotency_key="create-rank-body",
            )
            stale = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="rank-stale", title="Other", markdown="Needle",
                idempotency_key="create-rank-stale",
            )
            update_document(
                connection, claims, stale.document_id, expected_revision=1,
                title="Other", markdown="removed", idempotency_key="update-rank-stale",
            )
            hidden = create_document(
                connection, project_claims, context_records["project_compartment"],
                kind="page", slug="rank-hidden", title="Needle", markdown="Needle",
                idempotency_key="create-rank-hidden",
            )

            results = search_lexical(connection, claims, "Needle", limit=100).items

    scores = {
        document_id: max(item.score for item in results if item.document_id == document_id)
        for document_id in (title.document_id, heading.document_id, body.document_id)
    }
    assert scores[title.document_id] > scores[heading.document_id] > scores[body.document_id]
    assert stale.document_id not in {item.document_id for item in results}
    assert hidden.document_id not in {item.document_id for item in results}
    assert all(item.revision == 1 for item in results)


def test_lexical_search_uses_stable_query_bound_cursors(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            for number in range(3):
                create_document(
                    connection, claims, context_records["user_compartment"],
                    kind="page", slug=f"page-{number}", title="Same", markdown="Same",
                    idempotency_key=f"create-page-{number}",
                )

            found = []
            cursor = None
            while True:
                page = search_lexical(connection, claims, "Same", limit=1, cursor=cursor)
                found.extend(page.items)
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            with pytest.raises(ValueError, match="invalid cursor"):
                search_lexical(connection, claims, "Different", cursor=cursor)
            encoded = cast(str, cursor).encode()
            payload = json.loads(base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4)))
            payload["c"] = {}
            malformed = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
            for invalid in (malformed, "☃", cast(str, {})):
                with pytest.raises(ValueError, match="invalid cursor"):
                    search_lexical(connection, claims, "Same", cursor=invalid)
            foreign_claims = replace(claims, user_id=uuid4())
            with pytest.raises(ValueError, match="invalid cursor"):
                search_lexical(connection, foreign_claims, "Same", cursor=cursor)
            with pytest.raises(ValueError, match="invalid cursor"):
                search_lexical(connection, foreign_claims, "Same", cursor="☃")
            for invalid_keys in (["same"] * 51, [cast(str, [])]):
                with pytest.raises(ValueError, match="subsystem_keys"):
                    search_lexical(
                        connection, claims, "Same", subsystem_keys=invalid_keys
                    )
            with pytest.raises(ValueError, match="query"):
                search_lexical(connection, claims, cast(str, None))

    assert len(found) == 3
    assert [item.chunk_id for item in found] == sorted(item.chunk_id for item in found)


def test_lexical_search_filters_project_and_subsystems(database_url: str, context_records):
    claims = _claims(context_records, workspace=True, project=True)
    alpha, beta = uuid4(), uuid4()
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            connection.execute(
                """INSERT INTO subsystems
                   (tenant_id, workspace_id, id, key, name, description)
                   VALUES (%s, %s, %s, 'alpha', 'Alpha', 'Alpha'),
                          (%s, %s, %s, 'beta', 'Beta', 'Beta')""",
                (
                    claims.tenant_id, claims.workspace_id, alpha,
                    claims.tenant_id, claims.workspace_id, beta,
                ),
            )
            user_alpha = create_document(
                connection, claims, context_records["user_compartment"],
                kind="page", slug="user-alpha", title="Filter", markdown="Filter",
                idempotency_key="create-user-alpha",
            )
            project_alpha = create_document(
                connection, claims, context_records["project_compartment"],
                kind="page", slug="project-alpha", title="Filter", markdown="Filter",
                idempotency_key="create-project-alpha",
            )
            project_beta = create_document(
                connection, claims, context_records["project_compartment"],
                kind="page", slug="project-beta", title="Filter", markdown="Filter",
                idempotency_key="create-project-beta",
            )
            connection.execute(
                """INSERT INTO document_subsystems (tenant_id, document_id, subsystem_id)
                   VALUES (%s, %s, %s), (%s, %s, %s), (%s, %s, %s)""",
                (
                    claims.tenant_id, user_alpha.document_id, alpha,
                    claims.tenant_id, project_alpha.document_id, alpha,
                    claims.tenant_id, project_beta.document_id, beta,
                ),
            )

            all_results = search_lexical(connection, claims, "Filter").items
            project_results = search_lexical(
                connection, claims, "Filter", project_only=True
            ).items
            alpha_results = search_lexical(
                connection, claims, "Filter", subsystem_keys=["alpha"]
            ).items
            intersection = search_lexical(
                connection, claims, "Filter", project_only=True, subsystem_keys=["alpha"]
            ).items
            first = search_lexical(
                connection, claims, "Filter", limit=1, subsystem_keys=["alpha"]
            )
            with pytest.raises(ValueError, match="invalid cursor"):
                search_lexical(
                    connection, claims, "Filter", cursor=first.next_cursor,
                    subsystem_keys=["beta"],
                )
            with pytest.raises(ValueError, match="workspace context"):
                search_lexical(
                    connection, _claims(context_records), "Filter", subsystem_keys=["alpha"]
                )

    assert {item.document_id for item in all_results} == {
        user_alpha.document_id, project_alpha.document_id, project_beta.document_id
    }
    assert {item.document_id for item in project_results} == {
        project_alpha.document_id, project_beta.document_id
    }
    assert {item.document_id for item in alpha_results} == {
        user_alpha.document_id, project_alpha.document_id
    }
    assert {item.document_id for item in intersection} == {project_alpha.document_id}


def test_backlinks_filter_unreadable_sources(database_url: str, context_records):
    user_claims = _claims(context_records)
    project_claims = _claims(context_records, workspace=True, project=True)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, user_claims.tenant_id):
            target = _create_page(
                connection, user_claims, context_records["user_compartment"], "target"
            )
            source = _create_page(
                connection, project_claims, context_records["project_compartment"], "source"
            )
            link = set_document_link(
                connection, project_claims, source.document_id, target.document_id, "supports",
                idempotency_key="set-project-link", weight=0.75,
            )
            hidden = read_backlinks(connection, user_claims, target.document_id)
            visible = read_backlinks(connection, project_claims, target.document_id)

    assert hidden == []
    assert visible == [link]


def test_backlinks_use_bounded_keyset_pages(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            target = _create_page(
                connection, claims, context_records["user_compartment"], "paged-target"
            )
            for slug in ("paged-source-a", "paged-source-b"):
                source = _create_page(
                    connection, claims, context_records["user_compartment"], slug
                )
                set_document_link(
                    connection, claims, source.document_id, target.document_id, "related",
                    idempotency_key=f"link-{slug}",
                )
            first = read_backlinks_page(
                connection, claims, target.document_id, limit=1
            )
            second = read_backlinks_page(
                connection, claims, target.document_id, limit=1,
                cursor=first.next_cursor,
            )

    assert len(first.items) == len(second.items) == 1
    assert first.next_cursor is not None
    assert second.next_cursor is None
    assert first.items != second.items


def test_link_set_and_delete_are_idempotent(database_url: str, context_records):
    claims = _claims(context_records)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, claims.tenant_id):
            source = _create_page(
                connection, claims, context_records["user_compartment"], "link-source"
            )
            target = _create_page(
                connection, claims, context_records["user_compartment"], "link-target"
            )
            first = set_document_link(
                connection, claims, source.document_id, target.document_id, "related",
                idempotency_key="set-link",
            )
            second = set_document_link(
                connection, claims, source.document_id, target.document_id, "related",
                idempotency_key="set-link",
            )
            first_delete = delete_document_link(
                connection, claims, source.document_id, target.document_id, "related",
                idempotency_key="delete-link",
            )
            second_delete = delete_document_link(
                connection, claims, source.document_id, target.document_id, "related",
                idempotency_key="delete-link",
            )

    assert first == second
    assert first_delete is second_delete is True


def test_link_target_must_be_readable(database_url: str, context_records):
    user_claims = _claims(context_records)
    project_claims = _claims(context_records, workspace=True, project=True)
    with psycopg.connect(database_url, autocommit=True) as connection:
        with tenant_transaction(connection, user_claims.tenant_id):
            source = _create_page(
                connection, user_claims, context_records["user_compartment"], "source-user"
            )
            target = _create_page(
                connection, project_claims, context_records["project_compartment"], "target-project"
            )
            with pytest.raises(DocumentNotFound):
                set_document_link(
                    connection, user_claims, source.document_id, target.document_id, "references",
                    idempotency_key="unauthorized-link",
                )


def _create_page(connection, claims, compartment_id, slug):
    return create_document(
        connection, claims, compartment_id,
        kind="page", slug=slug, title=slug, markdown="Content",
        idempotency_key=f"create-{slug}",
    )


def _claims(records, *, agent=False, workspace=False, project=False):
    return ContextTokenClaims(
        tenant_id=records["tenant"],
        user_id=records["user"],
        agent_id=records["agent"] if agent else None,
        agent_run_id=records["run"] if agent else None,
        workspace_id=records["workspace"] if workspace or project else None,
        project_id=records["project"] if project else None,
        issued_at=1,
        expires_at=2,
    )
