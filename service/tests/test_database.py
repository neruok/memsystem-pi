import os
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg.errors import (
    CheckViolation,
    ForeignKeyViolation,
    InsufficientPrivilege,
    RaiseException,
)

from memsystem.database import tenant_transaction


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


def delete_tenants(admin_database_url: str, *tenant_ids: UUID) -> None:
    with psycopg.connect(admin_database_url, autocommit=True) as connection:
        connection.execute("DELETE FROM tenants WHERE id = ANY(%s)", (list(tenant_ids),))


def test_tenant_context_is_local_and_fail_closed(
    database_url: str, admin_database_url: str
):
    first, second = uuid4(), uuid4()

    with psycopg.connect(database_url, autocommit=True) as connection:
        try:
            with tenant_transaction(connection, first):
                connection.execute("INSERT INTO tenants (id, name) VALUES (%s, 'first')", (first,))
                assert connection.execute("SELECT id FROM tenants").fetchone() == (first,)

            assert connection.execute("SELECT id FROM tenants").fetchall() == []

            with tenant_transaction(connection, second):
                connection.execute("INSERT INTO tenants (id, name) VALUES (%s, 'second')", (second,))
                assert connection.execute("SELECT id FROM tenants").fetchone() == (second,)

            assert connection.execute("SELECT id FROM tenants").fetchall() == []
        finally:
            delete_tenants(admin_database_url, first, second)


def test_tenant_transaction_rejects_an_active_transaction(database_url: str):
    tenant_id = uuid4()

    with psycopg.connect(database_url) as connection, connection.transaction():
        with pytest.raises(RuntimeError, match="idle connection"):
            with tenant_transaction(connection, tenant_id):
                pass


def test_cross_tenant_foreign_key_fails(database_url: str, admin_database_url: str):
    first, second, workspace = uuid4(), uuid4(), uuid4()

    with psycopg.connect(database_url, autocommit=True) as connection:
        try:
            with tenant_transaction(connection, first):
                connection.execute("INSERT INTO tenants (id, name) VALUES (%s, 'first')", (first,))
                connection.execute(
                    "INSERT INTO workspaces (tenant_id, id, key, name) VALUES (%s, %s, 'first', 'first')",
                    (first, workspace),
                )

            with tenant_transaction(connection, second):
                connection.execute("INSERT INTO tenants (id, name) VALUES (%s, 'second')", (second,))

            with pytest.raises(ForeignKeyViolation), tenant_transaction(connection, second):
                connection.execute(
                    "INSERT INTO projects (tenant_id, workspace_id, key, name) VALUES (%s, %s, 'bad', 'bad')",
                    (second, workspace),
                )
        finally:
            delete_tenants(admin_database_url, first, second)


def test_project_compartment_must_match_parent_workspace(
    database_url: str, admin_database_url: str
):
    tenant, first_workspace, second_workspace, project = uuid4(), uuid4(), uuid4(), uuid4()
    root, first_compartment, second_compartment = uuid4(), uuid4(), uuid4()

    with psycopg.connect(database_url, autocommit=True) as connection:
        try:
            with tenant_transaction(connection, tenant):
                connection.execute("INSERT INTO tenants (id, name) VALUES (%s, 'tenant')", (tenant,))
                connection.execute(
                    "INSERT INTO workspaces (tenant_id, id, key, name) VALUES (%s, %s, 'one', 'one'), (%s, %s, 'two', 'two')",
                    (tenant, first_workspace, tenant, second_workspace),
                )
                connection.execute(
                    "INSERT INTO projects (tenant_id, id, workspace_id, key, name) VALUES (%s, %s, %s, 'project', 'project')",
                    (tenant, project, first_workspace),
                )
                connection.execute(
                    "INSERT INTO compartments (tenant_id, id, path, scope_type, name) VALUES (%s, %s, 'tenant', 'tenant', 'tenant')",
                    (tenant, root),
                )
                connection.execute(
                    """INSERT INTO compartments
                       (tenant_id, id, parent_id, path, scope_type, scope_id, name)
                       VALUES (%s, %s, %s, 'tenant.one', 'workspace', %s, 'one'),
                              (%s, %s, %s, 'tenant.two', 'workspace', %s, 'two')""",
                    (
                        tenant, first_compartment, root, first_workspace,
                        tenant, second_compartment, root, second_workspace,
                    ),
                )

            with pytest.raises(RaiseException, match="does not belong"), tenant_transaction(connection, tenant):
                connection.execute(
                    """INSERT INTO compartments
                       (tenant_id, parent_id, path, scope_type, scope_id, name)
                       VALUES (%s, %s, 'tenant.two.project', 'project', %s, 'bad')""",
                    (tenant, second_compartment, project),
                )
        finally:
            delete_tenants(admin_database_url, tenant)


def test_service_role_cannot_shadow_compartment_targets(database_url: str):
    with psycopg.connect(database_url, autocommit=True) as connection:
        with pytest.raises(InsufficientPrivilege):
            connection.execute("CREATE TEMP TABLE workspaces (tenant_id uuid, id uuid)")


def test_multi_row_document_cycle_fails(database_url: str, admin_database_url: str):
    tenant, root, compartment, first, second = (uuid4() for _ in range(5))

    with psycopg.connect(database_url, autocommit=True) as connection:
        try:
            with tenant_transaction(connection, tenant):
                connection.execute("INSERT INTO tenants (id, name) VALUES (%s, 'tenant')", (tenant,))
                connection.execute(
                    "INSERT INTO compartments (tenant_id, id, path, scope_type, name) VALUES (%s, %s, 'tenant', 'tenant', 'tenant')",
                    (tenant, root),
                )
                connection.execute(
                    """INSERT INTO compartments
                       (tenant_id, id, parent_id, path, scope_type, name)
                       VALUES (%s, %s, %s, 'tenant.global', 'global', 'global')""",
                    (tenant, compartment, root),
                )

            with pytest.raises(RaiseException, match="document cycle"), tenant_transaction(
                connection, tenant
            ):
                connection.execute(
                    """INSERT INTO documents
                       (tenant_id, id, compartment_id, parent_id, kind, slug)
                       VALUES (%s, %s, %s, %s, 'page', 'first'),
                              (%s, %s, %s, %s, 'page', 'second')""",
                    (tenant, first, compartment, second, tenant, second, compartment, first),
                )
        finally:
            delete_tenants(admin_database_url, tenant)


def test_removal_job_keeps_a_standalone_vector_tombstone(
    database_url: str, admin_database_url: str
):
    tenant = uuid4()

    with psycopg.connect(database_url, autocommit=True) as connection:
        try:
            with tenant_transaction(connection, tenant):
                connection.execute("INSERT INTO tenants (id, name) VALUES (%s, 'tenant')", (tenant,))
                rows = connection.execute(
                    """INSERT INTO jobs (tenant_id, profile, kind, vector_id)
                       VALUES (%s, 'default', 'index_remove', 42),
                              (%s, 'default', 'index_remove', 43)
                       RETURNING vector_id, enqueue_seq""",
                    (tenant, tenant),
                ).fetchall()
                assert [row[0] for row in rows] == [42, 43]
                assert rows[0][1] < rows[1][1]

            with pytest.raises((CheckViolation, RaiseException)), tenant_transaction(connection, tenant):
                connection.execute(
                    """INSERT INTO jobs (tenant_id, profile, kind, vector_id)
                       VALUES (%s, 'default', 'embed', 44)""",
                    (tenant,),
                )
        finally:
            delete_tenants(admin_database_url, tenant)


def test_project_subsystem_must_share_a_workspace(
    database_url: str, admin_database_url: str
):
    tenant, first_workspace, second_workspace = uuid4(), uuid4(), uuid4()
    project, subsystem = uuid4(), uuid4()

    with psycopg.connect(database_url, autocommit=True) as connection:
        try:
            with tenant_transaction(connection, tenant):
                connection.execute("INSERT INTO tenants (id, name) VALUES (%s, 'tenant')", (tenant,))
                connection.execute(
                    """INSERT INTO workspaces (tenant_id, id, key, name)
                       VALUES (%s, %s, 'one', 'one'), (%s, %s, 'two', 'two')""",
                    (tenant, first_workspace, tenant, second_workspace),
                )
                connection.execute(
                    """INSERT INTO projects (tenant_id, id, workspace_id, key, name)
                       VALUES (%s, %s, %s, 'project', 'project')""",
                    (tenant, project, first_workspace),
                )
                connection.execute(
                    """INSERT INTO subsystems
                       (tenant_id, id, workspace_id, key, name, description)
                       VALUES (%s, %s, %s, 'subsystem', 'subsystem', 'description')""",
                    (tenant, subsystem, second_workspace),
                )

            with pytest.raises(ForeignKeyViolation), tenant_transaction(connection, tenant):
                connection.execute(
                    """INSERT INTO project_subsystems
                       (tenant_id, workspace_id, project_id, subsystem_id)
                       VALUES (%s, %s, %s, %s)""",
                    (tenant, first_workspace, project, subsystem),
                )
        finally:
            delete_tenants(admin_database_url, tenant)


def test_service_role_cannot_rewrite_history_or_hierarchy(database_url: str):
    protected_targets = (
        "tenant_memberships", "agents", "workspaces", "projects",
        "compartments", "documents", "document_revisions",
    )

    with psycopg.connect(database_url, autocommit=True) as connection:
        for table in protected_targets:
            assert connection.execute(
                "SELECT has_table_privilege(current_user, %s, 'DELETE')", (table,)
            ).fetchone() == (False,)

        assert connection.execute(
            "SELECT has_table_privilege(current_user, 'document_revisions', 'UPDATE')"
        ).fetchone() == (False,)
        assert connection.execute(
            "SELECT has_column_privilege(current_user, 'documents', 'parent_id', 'UPDATE')"
        ).fetchone() == (False,)
        assert connection.execute(
            "SELECT has_column_privilege(current_user, 'compartments', 'path', 'UPDATE')"
        ).fetchone() == (False,)
        assert connection.execute(
            "SELECT has_column_privilege(current_user, 'workspaces', 'id', 'UPDATE')"
        ).fetchone() == (False,)
        assert connection.execute(
            "SELECT has_column_privilege(current_user, 'projects', 'workspace_id', 'UPDATE')"
        ).fetchone() == (False,)


def test_all_tenant_tables_force_rls_and_have_a_separate_owner(database_url: str):
    with psycopg.connect(database_url, autocommit=True) as connection:
        rows = connection.execute(
            """SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,
                      pg_get_userbyid(c.relowner) = current_user AS service_owned
               FROM pg_class c
               JOIN pg_namespace n ON n.oid = c.relnamespace
               WHERE n.nspname = 'public' AND c.relkind = 'r'
               ORDER BY c.relname"""
        ).fetchall()

        assert len(rows) == 20
        assert all(rls and forced and not service_owned for _, rls, forced, service_owned in rows)
        assert connection.execute(
            "SELECT rolsuper, rolinherit, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone() == (False, False, False)
