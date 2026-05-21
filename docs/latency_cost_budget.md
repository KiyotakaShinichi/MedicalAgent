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
