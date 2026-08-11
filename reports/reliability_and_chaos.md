# NLCare Reliability and Chaos Evidence

Generated from repository artifacts at `2026-08-11T04:48:02.120343+00:00`.

> NLCare remains synthetic-only, non-diagnostic, not clinically validated, and not production healthcare ready. Internal tests are engineering evidence, not evidence of patient benefit or medical effectiveness.

## Executed drills

- Automation fault injection: `strong`, `8/8` passed.
- RAG degradation resilience: `strong_offline_drill`.
- Planner concurrency matrix: `acceptable_internal_stress`.
- Forbidden tool exposure: `0`.
- Exceptions under planner load: `0`.

## Covered failure conditions

- Duplicate enqueue and delivery, lease contention, stale-lease recovery, bounded retries, dead letter and audited requeue.
- Stable event IDs after crash-like replay, signature rotation, tamper rejection, and stale-event rejection.
- Dense-index degradation and local fallback through existing RAG resilience drills.

## Not yet proven

- Managed Redis/PostgreSQL outage behavior, point-in-time restore, multi-host worker termination, real provider timeouts, and network partition recovery remain BLOCKED_EXTERNAL or require managed staging.
- No external notification is treated as clinician acknowledgement or emergency coverage.
