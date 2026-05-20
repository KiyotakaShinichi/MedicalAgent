# Runtime Quality Sentinel

The runtime quality sentinel aggregates existing RAG, safety, OOD, trace, and
cost artifacts into one operational snapshot.

It tracks:

- unsupported claim rate
- unsafe answer rate
- retrieval confidence distribution
- insufficient-evidence rate
- over-refusal rate
- post-generation validator trigger rate
- source-governance rejection rate
- cache hit rate
- p50/p95/p99 latency
- estimated route cost when available
- drift/OOD alert count

The output artifact is:

```bash
python scripts/run_runtime_quality_sentinel.py
```

```text
Data/evals/ops/latest_runtime_quality_sentinel.json
```

This is engineering observability only. It is not production SRE, clinical
safety monitoring, or proof of patient benefit.
