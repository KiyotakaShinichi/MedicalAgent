# NLCare Runtime Performance V-Next

Generated from repository artifacts at `2026-08-11T06:52:48.685003+00:00`.

> NLCare remains synthetic-only, non-diagnostic, not clinically validated, and not production healthcare ready. Internal tests are engineering evidence, not evidence of patient benefit or medical effectiveness.

## Measurement scopes

- Planner load matrix: process-local route and authorization concurrency only.
- Agent latency probe: local sparse RAG with in-memory SQLite.
- Mixed query stress: internally generated research, garbage, and dangerous prompts.

## Cold and warm evidence

- Planner prewarm: `2988.334` ms across `8` route families.
- Agent warmup: `NOT_RUN` ms.
- Mixed-query prewarm: `NOT_RUN` ms.

## Concurrency matrix

| Concurrency | Requests | Throughput rps | Error rate | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 11.832 | 0.0 | 69.098 | 181.846 | 277.021 |
| 10 | 100 | 30.786 | 0.0 | 172.556 | 614.037 | 1017.797 |
| 25 | 100 | 44.604 | 0.0 | 41.717 | 244.601 | 416.096 |
| 50 | 100 | 48.104 | 0.0 | 25.712 | 234.067 | 320.4 |
| 100 | 200 | 15.808 | 0.0 | 345.692 | 2639.216 | 4638.863 |

## Decision

No production SLO is claimed. Provider-token accounting remains incomplete when provider usage metadata is absent, and staged serving is not declared production-ready. The cold/warm split is mandatory for future comparisons.
