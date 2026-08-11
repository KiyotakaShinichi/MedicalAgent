# NLCare Runtime Performance V-Next

Generated from repository artifacts at `2026-08-11T04:48:02.105828+00:00`.

> NLCare remains synthetic-only, non-diagnostic, not clinically validated, and not production healthcare ready. Internal tests are engineering evidence, not evidence of patient benefit or medical effectiveness.

## Measurement scopes

- Planner load matrix: process-local route and authorization concurrency only.
- Agent latency probe: local sparse RAG with in-memory SQLite.
- Mixed query stress: internally generated research, garbage, and dangerous prompts.

## Cold and warm evidence

- Planner prewarm: `224.796` ms across `8` route families.
- Agent warmup: `NOT_RUN` ms.
- Mixed-query prewarm: `NOT_RUN` ms.

## Concurrency matrix

| Concurrency | Requests | Throughput rps | Error rate | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 153.677 | 0.0 | 6.137 | 8.843 | 11.493 |
| 10 | 100 | 173.876 | 0.0 | 5.604 | 63.415 | 162.593 |
| 25 | 100 | 185.186 | 0.0 | 5.392 | 6.251 | 6.576 |
| 50 | 100 | 164.824 | 0.0 | 6.105 | 7.704 | 12.026 |
| 100 | 200 | 139.252 | 0.0 | 6.844 | 12.706 | 59.229 |

## Decision

No production SLO is claimed. Provider-token accounting remains incomplete when provider usage metadata is absent, and staged serving is not declared production-ready. The cold/warm split is mandatory for future comparisons.
