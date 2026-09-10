# memsystem roadmap

## Status key

- `[x]` complete
- `[ ]` planned
- `[?]` requires measured evidence or a user decision

## Phase 0: Project skeleton (complete)

Completed on 2026-09-03. All exit checks passed.

- [x] Document the MCP service architecture.
- [x] Document the Pi extension architecture.
- [x] Create the Python MCP service package.
- [x] Register five placeholder MCP tools.
- [x] Register document and compartment resources.
- [x] Create the installable Pi extension package.
- [x] Add bounded untrusted-memory framing.
- [x] Add service and extension smoke tests.

Exit check:

```bash
npm run check
npm test
pi -e . --list-models
```

## Phase 1: PostgreSQL foundation (complete)

Completed on 2026-09-03. Pre-release builds use one resettable bootstrap schema.

- [x] Add local PostgreSQL and pgvector development configuration.
- [x] Add a resettable SQL bootstrap schema.
- [x] Create tenants, memberships, agents, and delegations.
- [x] Create compartments with tenant-aware foreign keys.
- [x] Create workspaces, projects, and subsystem facets.
- [x] Create documents, revisions, chunks, links, and jobs.
- [x] Add uniqueness, cycle, state, and size constraints.
- [x] Enable forced row-level security on every tenant table.
- [x] Add a non-owner service database role without `BYPASSRLS`.
- [x] Add a transaction-local tenant context helper.
- [x] Restrict the service role to safe column updates and soft deletion.
- [x] Reject cross-workspace scope targets and multi-row document cycles.
- [x] Protect trigger lookups from temporary-table shadowing.

Exit checks passed:

- Cross-tenant foreign keys fail.
- Missing tenant context returns no tenant rows.
- Reused connections do not retain previous tenant context.
- A clean database loads the complete schema.
- The service role cannot delete scope targets, documents, or revisions.
- All 15 Python tests and two TypeScript tests pass.
- An independent security review found no remaining issue in the final fixes.

## Phase 2: Secure document storage (complete)

Completed on 2026-09-03. MCP read and mutation adapters remain scheduled for Phases 3 and 8.

- [x] Add authenticated request identity.
- [x] Resolve tenant, user, agent, workspace, and project context.
- [x] Issue short-lived signed context tokens.
- [x] Implement fixed compartment authorization rules.
- [x] Implement document create, read, update, and soft delete.
- [x] Lock document rows during revision allocation.
- [x] Require expected revisions for updates.
- [x] Add idempotency keys for every mutation.
- [x] Implement typed links and authorized backlinks.
- [x] Implement collection navigation and revision history.

Exit criteria:

- Concurrent updates cannot regress the current revision.
- Unauthorized document reads and writes fail.
- Revoked or expired identities fail every read and write.
- Unauthorized link targets remain undisclosed.
- Repeated mutation requests produce one result.
- Soft-deleted and expired documents are unreadable.

## Phase 3: Full-text retrieval (complete)

Completed on 2026-09-03. All exit checks passed.

- [x] Split Markdown into heading-aware chunks.
- [x] Store chunk profile and source offsets.
- [x] Build `tsvector` values in the document transaction.
- [x] Implement title, heading, and chunk ranking.
- [x] Add deterministic tie-breaking and bounded pagination.
- [x] Implement project and subsystem filters.
- [x] Replace placeholder `memory_recall` and `memory_read` behavior.

Exit criteria:

- Exact identifiers remain searchable.
- Search returns only current authorized revisions.
- Search results contain stable provenance.
- Large pages return bounded results and continuation cursors.

## Phase 4: Embeddings and TurboVec (complete)

Completed on 2026-09-04. Phase 4 adds the `provisional-v1` profile, embedding jobs, a tenant-scoped TurboVec projection, and internal hybrid search. MCP vector recall remains disabled.

- [x] Define one provisional retrieval profile for evaluation.
- [x] Add the embedding job worker.
- [x] Add the index job worker.
- [x] Store normalized full-precision embeddings in PostgreSQL.
- [x] Create a TurboVec `IdMapIndex` with stable database IDs.
- [x] Implement incremental add, remove, prepare, and sync operations.
- [x] Add superseded and deleted vector tombstones.
- [x] Implement authorized allowlist search.
- [x] Handle empty and stale allowlists.
- [x] Implement generation rebuild, replay, fencing, and handle swap.
- [x] Add external index checksums.
- [x] Fuse lexical and vector ranks with reciprocal rank fusion.

Exit criteria:

- PostgreSQL can rebuild the complete TurboVec index.
- Stale index entries never reach callers.
- Interrupted indexing resumes without data loss.
- Hybrid search degrades to lexical search during vector failure.

## Phase 5: Retrieval evaluation (complete)

Completed on 2026-09-10. The evaluation separates semantic quality from storage and retrieval scale.

- [x] Create a local-model query and relevance corpus from project documentation.
  - [x] Add a deterministic clustered-vector corpus for engine measurements.
  - [x] Add realistic operational memory fixtures and labeled queries.
  - [x] Use authored fixtures instead of requiring unavailable production memories.
- [x] Compare exact pgvector, HNSW, TurboVec 4-bit, and TurboVec 2-bit.
- [x] Measure filtered recall, serial throughput, latency, index size, and TurboVec load time.
- [x] Measure comparable peak memory for PostgreSQL and TurboVec.
  - [x] Record engine-specific backend memory and process RSS signals.
- [x] Measure broad and scattered allowlists.
- [x] Repeat synthetic engine measurements across three deterministic seeds.
- [x] Measure scaling at 25,000 vectors across three seeds. Unchanged 4-bit settings miss full and broad recall targets.
- [x] Measure MTEB LongMemEval-shaped storage at 50,000, 100,000, and 200,000 vectors.
- [x] Replace the fixed allowlist ceiling with an optional runtime limit.
- [x] Evaluate exact reranking of TurboVec candidates before larger-scale selection.
- [x] Implement runtime exact candidate reranking with authorization-path regression tests.
- [x] Measure the full authorized reranking path with 25,000 Wikipedia chunks.
- [x] Activate Qwen3-Embedding-4B, 1,536 dimensions, and TurboVec 4-bit.
- [x] Enable MCP hybrid recall with lexical fallback.
- [x] Measure 1, 4, 8, 16, and 32 concurrent authorized retrieval clients.
- [x] Verify recall while an index rebuild runs.
- [x] Measure authenticated MCP concurrency with the Qwen provider.

Qwen3-Embedding-4B meets the semantic targets on the labeled project and operational corpora. Exact reranking makes TurboVec 4-bit meet the scaling recall target. The direct retrieval path stays below the one-second p95 target at 32 clients. The MCP path peaks at 5.810 queries per second and misses the p95 target. See [retrieval evaluation](docs/retrieval-evaluation.md).

Exit criteria:

- Recall and latency targets are documented.
- TurboVec configuration has benchmark evidence.
- Scaling thresholds are explicit.

## Phase 6: Pi context integration

- [ ] Load `~/.pi/agent/memsystem.json` safely.
- [ ] Load trusted workspace configuration.
- [ ] Load project and subsystem descriptors from `memsystem.d`.
- [ ] Validate configuration schemas and duplicate roots.
- [ ] Resolve Git and working-directory hints.
- [ ] Compute and refresh trusted-context fingerprints.
- [ ] Attach context tokens outside model-visible arguments.
- [ ] Implement `/memory status`, `/memory scope`, and `/memory target`.
- [ ] Add retry, cancellation, reconnect, and shutdown tests.

Exit criteria:

- Project files cannot override endpoint, credentials, tenant, or agent.
- The model cannot see authorization identifiers or tokens.
- Context changes force server resolution.
- Missing or conflicting context fails closed.

## Phase 7: External web fallback

- [ ] Define one external search provider, credential configuration, privacy terms, data region, and cost ceiling.
- [ ] Define an additive response contract that preserves `results`, `nextCursor`, trust, provenance, and score fields.
- [ ] Add a separate `externalResults` group.
- [ ] Keep `mode` limited to the internal `lexical` and `hybrid` modes.
- [ ] Define constrained `externalReason` and `externalStatus` values.
- [ ] Define status behavior for skipped, successful, failed, unavailable, cancelled, and truncated requests.
- [ ] Require trusted user opt-in and a deployment policy before provider use.
- [ ] Keep both policy controls outside model-visible arguments.
- [ ] Apply cancellation and strict query, result, byte, cost, and execution limits before provider activation.
- [ ] Define byte accounting for the service response and the Pi envelope.
- [ ] Keep provider credentials and authorization context outside model-visible data.
- [ ] Do not augment provider queries with authorization identifiers or stored content.
- [ ] Apply the untrusted-data envelope to all external fields.
- [ ] Add `external: "never" | "fallback"` to `memory_recall`.
- [ ] Keep `never` as the default.
- [ ] Search externally only when `fallback` is selected and internal recall returns no results.
- [ ] Use bounded provider snippets without direct page fetching.
- [ ] Preserve URL, title, excerpt, provider, and retrieval time for each external result.
- [ ] Return internal results when the provider fails.
- [ ] Test memory hits, memory misses, empty external results, provider failures, cancellation, truncation, and hostile snippets.
- [ ] Measure latency, cost, usefulness, and duplicate-query frequency.

Exit criteria:

- `external: "never"` preserves the existing response contract and internal recall behavior.
- `external: "fallback"` searches externally only after an internal miss.
- Empty results and provider failures have different status values.
- Internal and external rankings remain separate.
- External content never becomes a memory document automatically.
- Provider failures do not fail internal recall.
- The service does not augment provider queries with tenant identifiers, compartment identifiers, or stored content.
- The user sees the provider query privacy and cost boundary before opt-in.

## Phase 8: Safe mutation and scope management

- [ ] Enable `memory_remember` with strict action contracts.
- [ ] Enable `memory_manage` with confirmation policies.
- [ ] Enable `memory_scope_manage` with compare-and-swap checks.
- [ ] Add guarded compartment move operations.
- [ ] Implement atomic workspace and project descriptor writes.
- [ ] Implement `/memory init`, `/memory validate`, and `/memory sync`.
- [ ] Recover local bindings after partial server changes.
- [ ] Support archive and restore without model hard deletion.

Exit criteria:

- Shared-scope changes require user confirmation.
- Headless mutation requires an explicit operator policy.
- Failed local writes remain recoverable.
- Invalid or ambiguous actions make no change.

## Phase 9: Automatic recall and capture

- [ ] Add approved-page recall through `before_agent_start`.
- [ ] Enforce recall deadlines, result limits, and deduplication.
- [ ] Keep automatic recall disabled until user consent.
- [ ] Add private journal capture through `agent_settled`.
- [ ] Exclude thinking, images, environment values, and raw tool output.
- [ ] Add credential redaction and strict event limits.
- [ ] Track acknowledged session entries for idempotent retries.
- [ ] Add compaction summary checkpoints.
- [ ] Implement recall approval and capture consent commands.

Exit criteria:

- Captured journals remain user-private.
- Automatic recall uses approved curated pages only.
- Stored content cannot escape the untrusted-data envelope.
- Capture retries do not duplicate journal entries.

## Phase 10: Production hardening

- [ ] Add structured logs, metrics, and audit records.
- [ ] Add backup and restore procedures.
- [ ] Add index corruption drills and full rebuild tests.
- [ ] Add retention and hard-deletion jobs.
- [ ] Add embedding provider privacy controls.
- [ ] Add rate, query, result, and execution limits.
- [ ] Add deployment health and readiness checks.
- [ ] Add versioned SQL migrations.
- [ ] Add release packaging and upgrade documentation.

Exit criteria:

- Recovery drills meet documented targets.
- Security and race test suites pass.
- Metrics detect stale indexing and worker backlog.
- A clean environment can install and operate the release.

## Deferred work

Add these features only after measured need:

- [?] automatic knowledge consolidation
- [?] recursive graph traversal
- [?] custom deny policies
- [?] extracted fact or claim graphs
- [?] distributed vector index shards
- [?] LLM query rewriting or reranking
- [?] persistent external evidence cache
- [?] stale-memory revalidation policy
- [?] direct external page fetching
- [?] automatic promotion of external results
