# Operational memory samples

## Production deployment

Deploy the API from the `main` branch after all checks pass. Build one image and record its digest in the release ticket.

Apply database migrations before application tasks start. Keep the previous image digest available until health checks pass for ten minutes.

If error rate exceeds two percent, stop the rollout. Restore the previous image, then inspect migration compatibility before another attempt.

Never retry a failed destructive migration automatically. Ask the release operator to review database state and the migration log.

## Authentication incident

An incident on 2026-08-12 caused valid sessions to fail after key rotation. Workers cached the old verification key for thirty minutes.

The repair reduced the key cache lifetime to five minutes. A refresh failure now keeps the last valid key for one additional interval.

During another incident, compare token key IDs with the active key set. Do not disable signature verification to restore service.

## Billing idempotency

Every charge request must include a client-generated idempotency key. The key is unique for one account and one request body.

A repeated key with the same request body returns the stored charge result. A repeated key with different content returns a conflict.

Keep idempotency records for seven days. Support can search them by account, request key, and charge identifier.

## Database connection limit

Production permits 120 application database connections. Web processes can use 80, workers can use 30, and maintenance reserves 10.

Set each web process pool to a maximum of 10 connections. Increase process count only if the combined limit remains below 80.

Connection wait p95 must remain below 100 milliseconds. Reduce idle pool size before requesting a database limit increase.

## Search indexing recovery

PostgreSQL remains the source of truth for search content. Delete a damaged external vector index and rebuild it from current chunk embeddings.

Do not copy vectors from another tenant or generation. Verify the published checksum before the service loads a rebuilt generation.

Lexical search remains available during vector recovery. Alert when the oldest index job is more than five minutes old.

## API pagination contract

List endpoints use opaque keyset cursors. A cursor is valid only for the operation, filters, subject, and sort order that created it.

Reject changed filters instead of restarting pagination silently. Return no cursor when the page contains the final result.

The default page size is 20. The maximum page size is 100, even for internal clients.

## Customer export retention

Customer export archives expire after 24 hours. Store archives in the tenant region and encrypt each archive with a short-lived data key.

A download URL expires after 15 minutes. Creating another URL does not extend the archive retention period.

Delete an archive immediately after an account deletion completes. Audit records keep metadata but never archive content or download URLs.

## Repository test commands

Run `npm run check` for TypeScript validation. Run `npm test` for extension and service tests.

Start PostgreSQL with `docker compose up -d --wait` before database tests. Do not reset the volume unless schema changes require a clean load.

Use the service test database URL only for tests. Never point test cleanup code at a production database.

## Logging privacy

Structured logs can contain tenant-safe object identifiers, operation names, durations, and error classes. They must not contain document content.

Remove bearer tokens, context tokens, cookies, authorization headers, embeddings, and raw user prompts before log emission.

An audit event can record who changed access. It must not record the secret or memory text involved in that change.

## Worker retry policy

Workers claim jobs with a lease. Another worker can reclaim a job after the lease expires.

Use bounded exponential delay after failures. Stop automatic attempts after the configured maximum and expose the final error to operators.

A retry must verify that the document revision is still current. A stale worker must not overwrite newer content.

## Project conventions

Python service code lives under `service/src/memsystem`. TypeScript extension code lives under `pi-extension`.

Use PostgreSQL for authoritative tenant state. Use TurboVec only as a rebuildable projection.

Keep authorization checks inside the document transaction. Do not trust compartment identifiers supplied by model tool arguments.

## Emergency feature disable

Disable automatic recall before disabling explicit memory reads. Explicit reads give users a controlled recovery path.

Disable memory writes if context resolution becomes ambiguous. Keep authorized reads available when their context remains valid.

Record the feature change, operator, reason, and restoration condition. Do not store credentials in the incident note.
