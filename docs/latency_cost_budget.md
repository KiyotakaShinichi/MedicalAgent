# Latency and Cost Budget

NLCare tracks route-level latency as prototype engineering observability.
These are not production SLOs.

Routes:

- deterministic safety refusal
- cached educational answer
- dense/sparse RAG
- RAG plus reranker
- low-confidence retrieval safe-default
- hybrid prediction
- emotional distress support

Run:

```bash
python scripts/run_latency_profile.py
python scripts/run_route_latency_budget.py
```

Artifacts:

```text
Data/evals/ops/latest_latency_profile.json
Data/evals/ops/latest_latency_profile_phase2.json
Data/evals/ops/latest_route_latency_budget.json
Data/evals/ops/latest_cost_latency_report.json
```

If reranking increases latency without retrieval gain, the reranker should stay
optional and marked `needs_attention` or informational. High local p95 latency
must not be presented as production readiness.

The local load smoke uses `RAG_FORCE_SPARSE=true` and
`RAG_ENABLE_CROSS_ENCODER=false` by default so it measures app routing and cache
behavior rather than cold optional model downloads. Dense/cross-encoder latency
must be profiled in a separate explicit retrieval experiment.

Every route row includes `production_ready: false`; passing a local budget only
means the prototype did not regress under the current dev environment.

Percentiles require at least 30 observations per route. Routes with fewer
observations retain their raw p50/p95/p99 values for debugging but are labelled
`insufficient_samples`, never `ideal` or `acceptable`. This is a minimum
measurement-credibility rule, not evidence that 30 local samples establish a
production SLO.

Phase 2 also reports cold-start warm-up separately from steady local route
timing. That makes regression tracking cleaner, but it is still not a
production SLO. If warm-up is high, record it honestly instead of folding it
into a fake steady-state claim.
