# Observability

NLCare uses lightweight observability artifacts so AI-assisted edits can be
reviewed after the fact.

## Request-Level Fields

Trace and ops artifacts should prefer these fields when available:

- `request_id`
- `correlation_id`
- `route`
- `intent`
- `model`
- `tokens_in`
- `tokens_out`
- `estimated_cost`
- `retrieval_latency_ms`
- `reranker_latency_ms`
- `source_governance_latency_ms`
- `validator_latency_ms`
- `post_generation_validation_latency_ms`
- `cache_hit`
- `answerability_status`
- `retrieval_confidence`

## Quality Snapshots

Run:

```bash
python scripts/run_runtime_quality_sentinel.py
python scripts/load_test_agent.py
python scripts/run_dependency_security_scan.py
```

These outputs are engineering diagnostics. They do not prove clinical safety or
production healthcare readiness.
