# Cost and Latency Observability

NLCare tracks cost and latency as engineering telemetry only. These metrics help
compare route choices, cache behavior, and future local-SLM helper options. They
do not prove clinical safety, clinical usefulness, or real-world patient benefit.

## Artifact

Run:

```bash
python scripts/run_cost_latency_report.py
```

Output:

```text
Data/evals/ops/latest_cost_latency_report.json
```

The report includes:

- route and intent
- model/provider label when known
- estimated input/output tokens
- estimated provider cost
- p50/p95/p99 latency
- retrieval, reranker, validator, source-governance, and post-generation timing when captured
- cache hit/miss status
- source-tier correctness
- claim-validation pass status
- route-level cost comparison

Rows created before schema revision `0005_rag_cost_latency_fields` may not have
per-stage timing. The report keeps those fields null rather than inventing
precision.

## Route Comparison

The comparison table estimates:

- full API path
- validated cached path
- local SLM routing plus API answer
- local SLM query rewrite plus API answer
- deterministic-only refusal path

The current local/deterministic development path may log `$0` provider cost.
Comparison costs are explicit planning estimates, not audited billing.

## Safety Boundary

Cost optimization must never bypass:

- deterministic pre-generation safety gates
- source governance
- claim-level citation validation
- medical claim boundary checks
- post-generation safety validation
- release gate artifact checks

Fast or cheap is not allowed to mean less safe.
