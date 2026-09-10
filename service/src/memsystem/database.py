"""Small PostgreSQL boundary for tenant-scoped work."""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from psycopg import Connection
from psycopg.pq import TransactionStatus


@contextmanager
def tenant_transaction(connection: Connection, tenant_id: UUID | str) -> Iterator[Connection]:
    """Open one transaction with fail-closed, transaction-local tenant context."""
    if connection.info.transaction_status is not TransactionStatus.IDLE:
        raise RuntimeError("tenant_transaction requires an idle connection")

    tenant_id = UUID(str(tenant_id))
    with connection.transaction():
        connection.execute(
            "SELECT set_config('memsystem.tenant_id', %s, true)",
            (str(tenant_id),),
        )
        yield connection
