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
```

Artifact:

```text
Data/evals/rag/latest_reranker_ablation.json
```

The artifact reports retrieval hit proxies, source-tier correctness, unsupported-answer proxy, and reranker latency.
