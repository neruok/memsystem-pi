# Retrieval evaluation

## Status

This evaluation measures vector engine behavior, semantic relevance, authorized retrieval, and concurrency.

The active profile uses pinned Qwen3-Embedding-4B embeddings and TurboVec 4-bit compression.

## Run the benchmark

Start PostgreSQL, then run this command:

```bash
uv run --project service python -m memsystem.evaluation \
  --seeds 7 17 29 \
  --output docs/retrieval-benchmark.json
```

Use `--smoke` for a fast harness check. Do not use smoke results for profile decisions.

## Method

Each seed creates 10,000 normalized vectors with 1,536 dimensions. Each run contains 100 clusters and 50 query vectors.

The stored benchmark uses seeds 7, 17, and 29. The report includes each run and an aggregate summary.

The aggregate summary reports mean and minimum recall. It also reports mean and maximum p95 latency and process RSS growth.

The benchmark uses exact pgvector results as relevance labels. It measures recall at 10 for each approximate engine.

The benchmark tests these filters:

- all 10,000 vector identifiers
- one contiguous allowlist with 5,000 identifiers
- one random allowlist with 1,000 identifiers

Each case runs one warmup query. Timed queries run serially, and throughput is the inverse of mean latency.

PostgreSQL timing includes client serialization and one server round trip. Baseline TurboVec timing includes only the in-process search call.

The unfiltered PostgreSQL query omits the identifier filter. TurboVec receives the full authorized identifier list, as the service requires.

The pgvector HNSW test uses default index options. It uses `ef_search=40` and strict ordered iterative scans.

The harness verifies each HNSW plan with `EXPLAIN`. Each plan must use the HNSW index.

The baseline TurboVec test requests 30 candidates and measures recall for the first 10. This matches the service overfetch behavior.

The optional reranking test requests 10, 20, 30, 50, and 100 candidates. It ranks each set again with full-precision inner products.

Reranking latency includes the TurboVec search, candidate conversion, and exact scoring. Candidate recall shows the maximum recall that reranking can recover.

The benchmark rotates the multiplier order for each query. This prevents one multiplier from always receiving a cold or warm cache position.

TurboVec runs in a separate child process for each bit width. This keeps the resident memory measurements independent.

The harness records Linux `VmHWM` growth during build, load, and search. Each PostgreSQL engine uses a fresh backend process.

A second PostgreSQL connection reads the benchmark backend status from `/proc`. The harness disables PostgreSQL parallel query and maintenance workers.

Each TurboVec engine uses a fresh child process. Each value uses the process peak RSS before engine setup as its baseline.

PostgreSQL build measurements include table loading. The HNSW build also includes index creation.

The TurboVec baseline includes the inherited full-precision corpus. Its growth value measures index build, load, and search work.

## Results

The stored result is [retrieval-benchmark.json](retrieval-benchmark.json).

The recall columns show the mean and minimum across three seeds. The latency columns show the mean and maximum p95.

| Engine | Recall at 10, all | p95, all | Recall, broad | p95, broad | Recall, scattered | p95, scattered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pgvector exact | 1.000 / 1.000 | 103.609 / 106.642 ms | 1.000 / 1.000 | 253.039 / 255.169 ms | 1.000 / 1.000 | 66.716 / 68.785 ms |
| pgvector HNSW | 0.439 / 0.434 | 10.473 / 11.362 ms | 0.418 / 0.402 | 29.741 / 31.793 ms | 0.281 / 0.258 | 43.800 / 49.488 ms |
| TurboVec 4-bit | 0.863 / 0.858 | 5.605 / 5.904 ms | 0.865 / 0.858 | 1.676 / 1.768 ms | 0.872 / 0.852 | 2.635 / 2.798 ms |
| TurboVec 2-bit | 0.605 / 0.602 | 3.298 / 3.350 ms | 0.586 / 0.566 | 1.001 / 1.037 ms | 0.621 / 0.616 | 1.472 / 1.537 ms |

The build and load columns show the mean across three seeds. The RSS column shows the maximum.

| Engine | Build | Load | Storage | Maximum process RSS growth |
| --- | ---: | ---: | ---: | ---: |
| pgvector exact | 0.663 s | not applicable | 83,509,248-byte table | 10,227,712 bytes |
| pgvector HNSW | 21.149 s | database managed | 83,509,248-byte table and 81,928,192-byte index | 74,674,176 bytes |
| TurboVec 4-bit | 0.339 s | 0.012 s | 9,454,035-byte index | 129,351,680 bytes |
| TurboVec 2-bit | 0.175 s | 0.006 s | 4,809,843-byte index | 116,346,880 bytes |

These values compare `VmHWM` growth from fresh process baselines. PostgreSQL RSS includes touched shared pages. It excludes other server processes and operating-system cache.

The JSON also records the PostgreSQL backend memory-context total. This value is diagnostic only and is not the peak RSS measurement.

These values apply only to the recorded machine and dependency versions. The JSON result contains the complete environment details.

## Scaling check: 25,000 vectors

The stored result is [retrieval-benchmark-25000.json](retrieval-benchmark-25000.json).

Run this check with:

```bash
uv run --project service python -m memsystem.evaluation \
  --items 25000 --seeds 7 17 29 \
  --output docs/retrieval-benchmark-25000.json
```

This run retains 1,536 dimensions, 100 clusters, 50 queries per seed, and the existing engine settings.

The broad allowlist contains 12,500 identifiers. The scattered allowlist contains 2,500 identifiers.

The table shows minimum recall and maximum p95 latency across three seeds.

| Engine | Recall at 10, all | p95, all | Recall, broad | p95, broad | Recall, scattered | p95, scattered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pgvector exact | 1.000 | 267.967 ms | 1.000 | 1,333.040 ms | 1.000 | 348.597 ms |
| pgvector HNSW | 0.250 | 64.158 ms | 0.220 | 108.276 ms | 0.176 | 93.248 ms |
| TurboVec 4-bit | 0.824 | 10.198 ms | 0.838 | 3.641 ms | 0.858 | 4.548 ms |
| TurboVec 2-bit | 0.554 | 7.559 ms | 0.506 | 2.488 ms | 0.586 | 3.609 ms |

TurboVec 4-bit uses a 21,160,275-byte index. Its maximum process RSS growth is 234,590,208 bytes, or 223.7 MiB.

Its build time ranges from 0.493 to 0.561 seconds. Its load time ranges from 0.022 to 0.027 seconds.

The unchanged 4-bit settings miss the 0.85 recall target for the full and broad allowlists. Full-allowlist p95 also exceeds 10 ms.

RSS growth exceeds the 128 MiB budget defined for 10,000 vectors. That budget is not a validated capacity model for larger corpora.

The HNSW builds take 317–351 seconds. An earlier attempt exceeded a 600-second command limit and produced no report.

The completed retry used unchanged settings and a 2,400-second command limit.

These results do not extend the provisional acceptance envelope beyond the measured 10,000-vector corpus.

## Exact candidate reranking at 25,000 vectors

The stored result is [retrieval-benchmark-reranked-25000.json](retrieval-benchmark-reranked-25000.json).

Run this check with:

```bash
uv run --project service python -m memsystem.evaluation \
  --items 25000 --seeds 7 17 29 \
  --candidate-multipliers 1 2 3 5 10 \
  --output docs/retrieval-benchmark-reranked-25000.json
```

The test ranks each TurboVec candidate set again with the original full-precision vectors. It uses exact normalized inner products.

Each query rotates the first tested multiplier. This balances cache and execution-order effects across the candidate sizes.

The table shows minimum recall and maximum p95 latency across three seeds. Each latency includes TurboVec search and exact reranking.

| Engine | Candidates | Recall, all | Recall, broad | Recall, scattered | Maximum p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboVec 4-bit | 10 | 0.824 | 0.838 | 0.858 | 7.815 ms |
| TurboVec 4-bit | 20 | 0.986 | 0.994 | 0.998 | 7.854 ms |
| TurboVec 4-bit | 30 | 0.998 | 0.998 | 1.000 | 7.840 ms |
| TurboVec 4-bit | 50 | 1.000 | 1.000 | 1.000 | 7.884 ms |
| TurboVec 4-bit | 100 | 1.000 | 1.000 | 1.000 | 8.060 ms |
| TurboVec 2-bit | 10 | 0.554 | 0.506 | 0.586 | 4.202 ms |
| TurboVec 2-bit | 20 | 0.724 | 0.730 | 0.788 | 4.212 ms |
| TurboVec 2-bit | 30 | 0.824 | 0.828 | 0.884 | 4.245 ms |
| TurboVec 2-bit | 50 | 0.914 | 0.918 | 0.958 | 4.302 ms |
| TurboVec 2-bit | 100 | 0.970 | 0.976 | 0.994 | 4.477 ms |

Candidate recall and reranked recall are equal in these runs. Exact reranking recovered every exact top-10 result present in each candidate set.

The 4-bit index meets the recall and latency targets with 20 candidates. Exact reranking of the current 30-candidate setting reaches at least 0.998 recall.

The 2-bit index needs 50 candidates to meet the recall target. This result does not justify a compression change.

The 4-bit run used at most 233,824,256 bytes of process RSS growth. The 25,000-vector corpus has no approved memory target.

This test kept the provisional profile unchanged. Internal vector search reranks TurboVec candidates with exact PostgreSQL inner products.

The service retains the three-times-result-limit candidate budget. PostgreSQL rechecks authorization, current revision, expiry, embedding profile, and indexed generation before it scores candidates.

Results use descending exact scores, with chunk identifiers as the tie-breaker. Duplicate candidate identifiers produce one result.

The service rejects index output that exceeds the requested candidate count before PostgreSQL reranking.

Regression tests cover incorrect approximate ranking, duplicate and unauthorized candidates, and changes after allowlist selection. These changes include revocation, deletion, expiry, supersession, and invalid profile or generation markers.

The synthetic reranking measurements use in-memory vectors. They do not include PostgreSQL authorization checks or round trips.

The Wikipedia check below adds authorization and database round trips before profile activation.

These synthetic results do not establish production relevance or capacity. Test sharding only if a later threshold fails.

## Local embedding relevance

The labeled corpus uses 76 chunks from project documentation. It contains 20 hand-labeled queries in [project-docs-corpus.json](../service/benchmarks/project-docs-corpus.json).

Run the local model evaluation with this command:

```bash
uv run --project service python -m memsystem.relevance_evaluation \
  --output docs/local-embedding-evaluation.json
```

The command downloads `BAAI/bge-small-en-v1.5` to `~/.cache/memsystem/fastembed` on its first run.

| Engine | Hit rate at 5 | Mean recall at 5 | Mean reciprocal rank at 5 |
| --- | ---: | ---: | ---: |
| Exact vectors | 0.850 | 0.850 | 0.733 |
| TurboVec 4-bit | 0.850 | 0.850 | 0.708 |
| TurboVec 2-bit | 0.900 | 0.900 | 0.735 |

Exact search missed three labeled chunks. The misses cover rebuild replay, trusted workspace overrides, and stable vector identifiers.

Use 0.90 hit rate at 5 and 0.75 mean reciprocal rank at 5 as provisional semantic targets. The exact model result misses both targets.

The 2-bit result comes from only 20 queries. It does not override the larger synthetic result.

Do not promote this model to the active profile. Its 384 dimensions also require a full profile migration from the current 1,536 dimensions.

## Operational memory baseline

The expanded corpus adds 12 realistic operational memory sections and 12 labeled queries. It contains 89 chunks and 32 queries in total.

The fixtures cover deployment, incidents, billing, retention, privacy, retries, recovery, and configuration. They do not contain production data.

Run this baseline with:

```bash
uv run --project service python -m memsystem.relevance_evaluation \
  --manifest service/benchmarks/project-operational-corpus.json \
  --output docs/operational-memory-evaluation.json
```

The stored result is [operational-memory-evaluation.json](operational-memory-evaluation.json).

| BGE-small engine | Hit rate at 5 | Mean recall at 5 | Mean reciprocal rank at 5 |
| --- | ---: | ---: | ---: |
| Exact vectors | 0.844 | 0.844 | 0.718 |
| TurboVec 4-bit | 0.844 | 0.844 | 0.703 |
| TurboVec 2-bit | 0.844 | 0.844 | 0.713 |

TurboVec 4-bit preserved the exact hit rate. BGE-small still misses the semantic targets.

The Qwen runs used the scheduled 125-watt GPU mode. Timing was disabled, but quality and PyTorch allocation measurements remain valid.

Stored results: [0.6B](qwen3-embedding-0.6b-operational-evaluation.json), [4B](qwen3-embedding-4b-operational-evaluation.json), and [8B](qwen3-embedding-8b-operational-evaluation.json).

| Qwen model | Engine | Hit rate at 5 | Mean reciprocal rank at 5 |
| --- | --- | ---: | ---: |
| 0.6B | Exact | 0.906 | 0.779 |
| 0.6B | TurboVec 4-bit | 0.906 | 0.779 |
| 0.6B | TurboVec 2-bit | 0.906 | 0.779 |
| 4B | Exact | 0.938 | 0.826 |
| 4B | TurboVec 4-bit | 0.938 | 0.846 |
| 4B | TurboVec 2-bit | 0.938 | 0.841 |
| 8B | Exact | 0.938 | 0.798 |
| 8B | TurboVec 4-bit | 0.938 | 0.799 |
| 8B | TurboVec 2-bit | 0.938 | 0.784 |

Aggregate results include 20 project queries and 12 operational queries. The operational queries are easier than the project queries.

| Qwen model with TurboVec 4-bit | Project hit / MRR at 5 | Operational hit / MRR at 5 |
| --- | ---: | ---: |
| 0.6B | 0.900 / 0.696 | 0.917 / 0.917 |
| 4B | 0.950 / 0.829 | 0.917 / 0.875 |
| 8B | 0.950 / 0.729 | 0.917 / 0.917 |

Qwen3 4B exact, 4-bit, and 2-bit results meet both semantic targets in both query groups.

Select 4-bit compression for further tests. It also meets the larger synthetic recall target, while 2-bit compression does not.

## Qwen3 embedding relevance

The evaluator also supports `Qwen3-Embedding-0.6B`, `Qwen3-Embedding-4B`, and `Qwen3-Embedding-8B` through Transformers.

Install the optional Qwen dependencies, then run the 0.6B model:

```bash
uv sync --project service --group qwen
uv run --project service --group qwen python -m memsystem.relevance_evaluation \
  --backend qwen \
  --model Qwen/Qwen3-Embedding-0.6B \
  --device cuda \
  --exact-only \
  --quality-only \
  --output docs/qwen3-embedding-0.6b-evaluation.json
```

The query uses the official web-search retrieval instruction. Documents do not use an instruction.

| Model and device | Dimensions | Hit rate at 5 | Mean reciprocal rank at 5 | Peak PyTorch allocation |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B, CPU | 1,024 | 0.900 | 0.696 | not measured |
| Qwen3 0.6B, RTX 3090 | 1,024 | 0.900 | 0.696 | 1,323,270,144 bytes |
| Qwen3 4B, RTX 3090 | 1,536 | 0.950 | 0.808 | 8,317,076,480 bytes |
| Qwen3 8B, RTX 3090 | 1,536 | 0.950 | 0.746 | 15,385,838,592 bytes |

The CPU and GPU 0.6B runs produced the same top-five lists. Different numeric precision produced different vector checksums.

GPU timing results were discarded because the scheduled power limit changed during the runs.

The 8B quality result comes from an earlier unpinned run. Its artifact records the resolved cache revision and this limitation.

Qwen3 0.6B meets the hit-rate target but misses the mean reciprocal rank target.

Qwen3 4B meets both targets and has the best relevance result. Qwen3 8B uses more memory without a relevance gain on this corpus.

The 4B model supports up to 2,560 dimensions. The 8B model supports up to 4,096 dimensions.

The larger-model runs used `--dimensions 1536` to match the current vector dimension. A model change still requires a full profile migration.

Keep 0.6B as the resource-saving candidate. Select Qwen3 4B with TurboVec 4-bit for provider integration.

## Authorized Wikipedia scaling check

The final scaling check uses 25,000 chunks from the Simple English Wikipedia dump. It reads 10,118 non-redirect articles without expanding the dump.

The test stores Wikipedia text in the service schema. It uses deterministic clustered vectors so results remain comparable to the engine benchmark.

Semantic quality is not part of this test. The labeled project and operational corpora supply the semantic evidence.

Run the check with this command:

```bash
uv run --project service python -m memsystem.wikipedia_evaluation \
  --items 25000 --queries 50 \
  --output docs/wikipedia-e2e-evaluation.json
```

The measured path includes these stages:

1. Start the database transaction.
2. Set the tenant context.
3. Build the authorized vector allowlist.
4. Search TurboVec for 30 candidates.
5. Rerank candidates in PostgreSQL.
6. Commit the transaction.

The check sets a one-second p95 target at 25,000 authorized vectors. This target applies to the complete measured path, not only the vector engine call.

| Metric | Result |
| --- | ---: |
| Recall at 10, mean and minimum | 1.000 / 1.000 |
| End-to-end latency, p50 | 385.144 ms |
| End-to-end latency, p95 | 443.349 ms |
| Serial throughput | 2.637 queries/s |
| Database load time | 221.219 s |
| Index rebuild time | 299.581 s |
| Index size | 21,160,275 bytes |
| Benchmark process peak RSS | 1,058,766,848 bytes |
| PostgreSQL backend peak RSS | 163,209,216 bytes |

The p95 result meets the one-second target. The index rebuild completed in 5.0 minutes.

The stored result is [wikipedia-e2e-evaluation.json](wikipedia-e2e-evaluation.json).

## MCP vector provider

The active profile uses Qwen3-Embedding-4B at the pinned revision. It truncates vectors to 1,536 dimensions.

Set `MEMSYSTEM_VECTOR_RECALL=true` to enable hybrid MCP results. Each process loads one provider and at most eight tenant indexes.

The service returns lexical results when the provider or vector index fails. Hybrid responses do not return a continuation cursor.

A GPU smoke check loaded the provider and embedded one document and one query. The warm query took 81 ms.

The document and query similarity was 0.884586. The stored result is [qwen3-4b-provider-smoke.json](qwen3-4b-provider-smoke.json).

The integration test covers authenticated MCP recall, vector search, exact reranking, rank fusion, and lexical fallback.

## Concurrent authorized retrieval

The concurrency test uses the same 25,000 Wikipedia chunks. Each request opens a database connection and runs the authorized reranking path.

The test uses precomputed query vectors. It excludes query embedding and MCP transport overhead.

| Clients | Throughput | p50 | p95 | p99 | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.123 queries/s | 310.913 ms | 372.831 ms | 382.601 ms | 0 |
| 4 | 10.123 queries/s | 381.985 ms | 448.151 ms | 512.823 ms | 0 |
| 8 | 18.194 queries/s | 404.650 ms | 511.724 ms | 527.723 ms | 0 |
| 16 | 27.682 queries/s | 457.376 ms | 684.537 ms | 707.878 ms | 0 |
| 32 | 43.274 queries/s | 570.549 ms | 889.959 ms | 922.117 ms | 0 |

Mean recall at 10 was 0.998437 for each client count. Minimum per-query recall was 0.9.

The rebuild check completed 21 searches while it rebuilt the index. It had no errors and retained 0.9 minimum recall.

Rebuild-search p95 was 1,024.967 ms. The 64-second rebuild caused a small increase above the one-second steady-state target.

The stored result is [concurrency-evaluation.json](concurrency-evaluation.json).

## Concurrent MCP hybrid recall

This test uses authenticated Streamable HTTP MCP calls against the 25,000-chunk corpus. Each call includes context verification and lexical search.

The call also includes serialized Qwen query embedding, vector search, exact reranking, rank fusion, and response framing.

| Clients | Throughput | p50 | p95 | p99 | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.232 queries/s | 772.657 ms | 1,056.685 ms | 1,154.696 ms | 0 |
| 4 | 3.830 queries/s | 1,009.002 ms | 1,383.245 ms | 1,533.292 ms | 0 |
| 8 | 5.810 queries/s | 1,255.293 ms | 1,894.704 ms | 2,268.235 ms | 0 |
| 16 | 5.203 queries/s | 2,594.409 ms | 4,208.842 ms | 4,691.517 ms | 0 |
| 32 | 4.372 queries/s | 5,318.225 ms | 9,055.250 ms | 9,528.383 ms | 0 |

All 320 measured calls returned complete hybrid results with trust framing and no cursor. The warmup loaded the model and index in 10.628 seconds.

Throughput peaks at 5.810 queries per second with eight clients. Higher concurrency increases queue time because one process serializes Qwen inference.

The integrated MCP path misses the one-second p95 target. Provider batching or more provider processes require a new measurement before automatic recall.

The stored result is [mcp-concurrency-evaluation.json](mcp-concurrency-evaluation.json).

## Larger semantic dataset search

The recommended source is [LongMemEval Cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). Its MIT-licensed Cleaned-S split has 500 questions and session evidence identifiers.

[ReMe LongMemEval Cleaned-S](https://huggingface.co/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2) corrects 58 evidence-label records. Its dataset card does not declare a license.

The MTEB [LongMemEval](https://huggingface.co/datasets/mteb/LongMemEval) conversion uses an MIT license and standard corpus, query, and relevance-label files. Its corpus has 237,655 rows.

ConvoMem supplies 75,336 questions with message evidence and distractors. LoCoMo supplies 5,882 passages and 1,964 queries. Both use CC BY-NC 4.0.

Use the MTEB LongMemEval conversion. It gives the evaluator standard corpus, query, and relevance-label files.

## LongMemEval scale check

The scale check uses the MTEB LongMemEval corpus at revision `9dc1a8fdcf9b5676f87c2cdccac021988f6ff5af`. The corpus contains 237,655 rows.

This check uses deterministic clustered vectors. It measures storage and retrieval scale but does not measure LongMemEval semantic relevance.

| Vectors | p50 | p95 | Throughput | Minimum recall at 10 | Index size | Rebuild |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 | 697.657 ms | 794.822 ms | 1.558 queries/s | 0.900 | 40,654,035 bytes | 90.141 s |
| 100,000 | 1,341.283 ms | 1,514.299 ms | 0.729 queries/s | 1.000 | 79,666,515 bytes | 174.815 s |
| 200,000 | 3,717.212 ms | 4,573.362 ms | 0.259 queries/s | 1.000 | 157,666,515 bytes | 427.159 s |

The test used no allowlist limit. The 200,000-vector run completed without fallback or errors.

The index size scales linearly at about 790 bytes per vector. Whole-path retrieval latency increases sharply with the authorized vector count.

The scale run reached 6,839,037,952 bytes of peak RSS. This value includes full-precision source vectors, benchmark data, and a buffered database result.

A later fix changed rebuild reads to a server-side cursor and removed old generation files after publication. A 10,000-vector post-fix check reached 671,875,072 bytes and passed all retrieval checks.

Do not use the old 200,000-vector RSS value as a current capacity estimate. Run a new large memory check only when a capacity decision requires it.

Stored results: [scale run](longmemeval-scale-evaluation.json) and [post-fix rebuild check](longmemeval-rebuild-smoke.json).

## Decision

Select Qwen3-Embedding-4B with 1,536 dimensions and TurboVec 4-bit for provider integration. Exact reranking uses 30 candidates for a result limit of 10.

The maximum measured RSS growth is 123.4 MiB across three seeds. This result meets the provisional 128 MiB target.

Do not promote TurboVec 2-bit. Its measured recall is below the minimum target for all filters.

Use these provisional acceptance targets for the 10,000-vector corpus:

- recall at 10 must be at least 0.85 for each filter
- p95 vector search latency must not exceed 10 ms
- TurboVec process RSS growth must not exceed 128 MiB

The service has no default allowlist ceiling. An unbounded query materializes every authorized vector identifier and can use substantial memory and time.

Set `MEMSYSTEM_VECTOR_ALLOWLIST_LIMIT` to a positive runtime-specific ceiling for deployed services. On this host, 50,000 vectors is the largest measured size below the one-second p95 target.

Recheck the profile when p95 latency exceeds 10 ms or sampled recall falls below 0.85. Test sharding only after one threshold fails.

## Limits

The labeled corpora contain project documentation and authored operational fixtures. They cannot predict every future memory workload.

Add representative production samples to later regression checks when those samples exist. Phase 5 does not require production history from an unbuilt system.

The serial throughput result does not predict concurrent service throughput. PostgreSQL process RSS does not measure total server memory or operating-system cache. Parallel or concurrent workloads can use more memory.

The three runs use 150 total queries. This sample remains too small to predict all production workloads.
