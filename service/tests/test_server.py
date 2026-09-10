import pytest
from mcp import Client
from starlette.testclient import TestClient

from memsystem.server import mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(mcp, raise_exceptions=True) as connected:
        yield connected


@pytest.mark.anyio
async def test_planned_tools_are_registered(client: Client):
    result = await client.list_tools()

    assert {tool.name for tool in result.tools} == {
        "memory_manage",
        "memory_read",
        "memory_recall",
        "memory_remember",
        "memory_scope_manage",
    }


@pytest.mark.anyio
async def test_resource_templates_are_registered(client: Client):
    result = await client.list_resource_templates()

    assert {template.uri_template for template in result.resource_templates} == {
        "memory://compartments/{compartment_id}",
        "memory://compartments/{compartment_id}/children",
        "memory://documents/{document_id}",
        "memory://documents/{document_id}/revisions/{revision}",
    }


@pytest.mark.anyio
async def test_recall_reports_unconfigured_storage(client: Client):
    result = await client.call_tool("memory_recall", {"query": "authentication"})

    assert result.is_error
    assert "Storage is not configured" in result.content[0].text


@pytest.mark.anyio
async def test_recall_rejects_large_limit(client: Client):
    result = await client.call_tool("memory_recall", {"query": "x", "limit": 11})

    assert result.is_error


@pytest.mark.anyio
async def test_manage_rejects_unknown_link_type(client: Client):
    result = await client.call_tool(
        "memory_manage",
        {"action": "link", "document": "a", "target": "b", "link_type": "unknown"},
    )

    assert result.is_error
    assert "Storage is not configured" not in result.content[0].text


def test_http_transport_requires_authentication():
    with TestClient(mcp.streamable_http_app()) as client:
        response = client.get("/mcp")
        context_response = client.post("/context", json={"tenantId": "unused"})

    assert response.status_code == 401
    assert response.json() == {
        "error": "invalid_token",
        "error_description": "Authentication required",
    }
    assert context_response.status_code == 401
