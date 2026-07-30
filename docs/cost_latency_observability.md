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

Two narrower companion evaluations keep optimization claims separated from
billing and deployment claims:

```bash
python scripts/run_retrieval_runtime_cache_eval.py
python scripts/run_provider_usage_reconciliation.py
```

The first compares a fixed pre-cache baseline with a repeated local
forced-sparse regression probe. It does not measure dense unique-query,
network, cloud-load, or production latency. The second only computes estimate
error when the same request has both an estimate and provider-reported usage.
It remains `blocked_configuration` until at least 30 paired requests and 80%
provider-usage coverage exist; it never creates paid traffic to satisfy that
requirement.

The report includes:

- route and intent
- model/provider label when known
- provider-reported input/output tokens when returned by the provider
- separately labelled per-call and pipeline token estimates when usage metadata is unavailable
- estimated provider cost using configurable engineering pricing assumptions
- p50/p95/p99 latency
- safety-gate, intent-routing, retrieval, pre-generation governance, reranker,
  compression, generation, source-governance, and post-generation timing when captured
- cache hit/miss status
- source-tier correctness
- claim-validation pass status
- route-level cost comparison
- token-usage coverage and p95/p99 sample-size credibility labels

Rows created before schema revision `0005_rag_cost_latency_fields` may not have
per-stage timing. The report keeps those fields null rather than inventing
precision. Rows before `0011_llm_usage_telemetry` may not have structured
provider usage. The report never silently mixes those legacy estimates with
provider-reported totals.

The `local_probe_stage_latency` section is derived separately from the repeated
local route probe. It reports sample counts and p50/p95 values per stage, but it
also records that the probe uses local execution, forced sparse retrieval, and
in-memory SQLite. These numbers are useful for bottleneck attribution, not a
production SLO.

No prompt text, completion text, or private model reasoning is stored in the
token-usage envelope. Cost is a capacity-planning estimate, not reconciled
provider billing. Tail percentiles with fewer than 30 samples are labelled
`insufficient_n_for_tail_claim`.

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
