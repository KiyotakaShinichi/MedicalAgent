# RAG Reranking

NLCare uses dense + sparse retrieval with reciprocal-rank fusion first, then an optional local cross-encoder reranker over the fused candidate set.

This is retrieval precision engineering only. It does not guarantee medical correctness, clinical safety, or real-world answer quality.

## Behavior

- `RAG_ENABLE_CROSS_ENCODER=true` enables the optional reranker.
- `RAG_CROSS_ENCODER_MODEL` selects the local model, for example `BAAI/bge-reranker-base`.
- If the dependency or model is unavailable, the system falls back to the existing heuristic metadata/safety reranker.
- Source tier metadata, `allowed_use`, staleness, `parent_id`, and provenance fields are preserved.
- Source governance, citation validation, medical claim boundaries, and post-generation validation still run after reranking.

## Evaluation

Run:

```bash
python scripts/run_reranker_ablation.py
python scripts/run_retrieval_ablation_metrics.py
```

Artifact:

```text
Data/evals/rag/latest_reranker_ablation.json
```

The artifact compares dense-only, sparse-only, hybrid RRF, and hybrid RRF plus
cross-encoder/fallback. It reports MRR, NDCG@10, Recall@5/10, source-hit rate,
claim-support proxy, unsupported-answer proxy, p50/p95 retrieval latency, and
reranker latency.

Current policy: keep reranking as supporting evidence unless the ablation and
retrieval goldset both show better Recall@10/MRR without increasing unsupported
context or violating source-tier correctness. If the artifact says
`improvement_proven: false`, the dashboard must say the scaffold exists but
retrieval improvement has not been proven.

The local load smoke forces sparse retrieval and disables optional cross-encoder
loading by default so latency measurements do not include surprise model
downloads. Enable model-backed reranking explicitly when evaluating retrieval
quality, not during routine ship smoke.
