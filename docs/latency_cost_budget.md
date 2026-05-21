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

Run:

```bash
python scripts/run_route_latency_budget.py
```

Artifacts:

```text
Data/evals/ops/latest_route_latency_budget.json
Data/evals/ops/latest_cost_latency_report.json
```

If reranking increases latency without retrieval gain, the reranker should stay
optional and marked `needs_attention` or informational. High local p95 latency
must not be presented as production readiness.
