# memsystem

Agent memory system with an MCP service and a companion Pi extension.

## Layout

- `service/`: Python MCP service
- `pi-extension/`: installable Pi extension
- `docs/mcp-service-architecture.md`: service design
- `docs/pi-extension-architecture.md`: extension design
- `ROADMAP.md`: implementation status and remaining phases

## Service

Start the local PostgreSQL service. The first start loads the resettable schema and the pgvector extension.

```bash
docker compose up -d --wait
```

Set the connection URL and run the checks:

```bash
export MEMSYSTEM_DATABASE_URL=postgresql://memsystem_service:memsystem@127.0.0.1:5432/memsystem
export MEMSYSTEM_TEST_DATABASE_URL="$MEMSYSTEM_DATABASE_URL"
uv sync --project service --group qwen
uv run --project service pytest
uv run --project service mcp dev service/src/memsystem/server.py
```

Reset all pre-release data after a schema change:

```bash
docker compose down -v
docker compose up -d --wait
```

Configure a bearer credential before you run Streamable HTTP. The user ID must match an active tenant membership.

```bash
export MEMSYSTEM_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MEMSYSTEM_API_TOKEN_USER_ID=<user-uuid>
export MEMSYSTEM_API_TOKEN_EXPIRES_AT="$(python -c 'import time; print(int(time.time()) + 3600)')"
export MEMSYSTEM_CONTEXT_TOKEN_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export MEMSYSTEM_VECTOR_RECALL=true
export MEMSYSTEM_EMBEDDING_DEVICE=auto
uv run --project service --group qwen mcp run service/src/memsystem/server.py --transport streamable-http
```

The HTTP transport rejects missing, unknown, malformed, and expired credentials. Rotate the environment values to revoke a credential.

Phase 2 includes authenticated identity, database-backed context resolution, signed context tokens, and fixed compartment authorization. Each authorization check reads current memberships, delegations, and agent runs in the document transaction.

An authenticated client can send trusted hints to `POST /context`. The response includes canonical scope details and a user-bound context token.

Context tokens expire after five minutes by default. Set `MEMSYSTEM_CONTEXT_TOKEN_TTL` from 1 through 900 seconds to change this period.

The document storage layer supports authorized create, read, update, and soft-delete operations. Updates lock the document row and require the current revision.

Create and update transactions split Markdown at headings. They store source offsets and weighted PostgreSQL full-text vectors for each chunk.

The storage layer can rank authorized current revisions with PostgreSQL full-text search. Search pages use bounded, filter-specific keyset cursors. Trusted context can restrict results to the active project or selected subsystem keys.

Every document mutation requires an idempotency key. Concurrent retries return one stored result, and conflicting requests fail.

Typed link mutations require source write access and target read access. Backlink reads omit unauthorized, deleted, and expired sources.

Collection children and revision history use bounded keyset pages. Cursors are bound to one document and operation.

The `memory_recall` and `memory_read` MCP tools use authenticated storage. Each HTTP call requires an `X-Memsystem-Context` header with a signed context token.

`memory_recall` uses lexical search by default. It uses hybrid search when vector recall is enabled and a healthy index exists. Library-path filtering and link expansion are not available yet.

A planned external fallback will add `external: "never" | "fallback"`. It will search bounded provider snippets only after an internal miss. A trusted user opt-in and a deployment policy must both allow provider use. The service will preserve the current internal response fields, add separate external results, and never persist them automatically.

Phase 4 includes embedding jobs, TurboVec index jobs, hybrid search, and generation rebuilds. Reciprocal rank fusion combines lexical and vector results. It returns lexical results when the provider or index is unavailable. Rebuilds use ordered job replay and a short commit fence. They publish the database state before they swap the in-memory owner. The first vector recall loads the published path and verifies its checksum.

The active profile uses pinned Qwen3-Embedding-4B vectors with 1,536 dimensions and TurboVec 4-bit compression. Set `MEMSYSTEM_VECTOR_RECALL=true` to enable vector results in `memory_recall`. Hybrid results do not provide a continuation cursor.

Run the Phase 5 retrieval evaluations:

```bash
uv run --project service python -m memsystem.evaluation \
  --seeds 7 17 29 \
  --output docs/retrieval-benchmark.json
uv run --project service python -m memsystem.relevance_evaluation \
  --output docs/local-embedding-evaluation.json
uv run --project service python -m memsystem.wikipedia_evaluation \
  --items 25000 --queries 50 \
  --output docs/wikipedia-e2e-evaluation.json
uv run --project service python -m memsystem.concurrency_evaluation \
  --items 25000 --requests 64 \
  --output docs/concurrency-evaluation.json
uv run --project service --group qwen python -m memsystem.mcp_concurrency_evaluation \
  --items 25000 --requests 64 \
  --output docs/mcp-concurrency-evaluation.json
service/benchmarks/download-longmemeval.sh
env -u MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT \
  uv run --project service python -m memsystem.longmemeval_scale_evaluation \
  --sizes 50000 100000 200000 \
  --output docs/longmemeval-scale-evaluation.json
```

See [retrieval evaluation](docs/retrieval-evaluation.md) for results and limits.

`memory_read` supports bounded content pages, collection children, backlinks, and revision history. Read and recall results label stored content as untrusted data. Phase 6 will make the Pi extension resolve and attach context tokens.

## Pi extension

```bash
npm install
npm run check
MEMSYSTEM_URL=http://127.0.0.1:8000/mcp pi -e ./pi-extension/index.ts
```

Use `/memory status` to show configured endpoint and connection state.

After publishing the repository, install the extension directly:

```bash
pi install git:github.com/OWNER/REPOSITORY
```

The root `package.json` points Pi to `pi-extension/index.ts`.

The extension does not attach credentials or context tokens yet. Phase 6 will add this client integration.

External fallback work starts only after Phase 6. The service, not the extension, will own provider calls.

Mutation adapters stay disabled until authorization and confirmation policies exist.
