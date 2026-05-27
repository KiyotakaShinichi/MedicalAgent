# Latency Phase 3

The phase-3 latency artifact keeps latency presentation honest and operational.

Run:

```bash
python scripts/run_latency_phase3.py
```

Output:

```text
Data/evals/ops/latest_latency_phase3_plan.json
```

It summarizes current route p50/p95/p99 values, route budgets, bottleneck stages, safe optimizations already preserved, and the next optimization backlog.

The artifact keeps `production_ready: false`. Local p95 budgets are engineering regression checks, not production SLOs or hospital-readiness evidence.
