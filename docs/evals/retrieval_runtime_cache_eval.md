# Retrieval runtime-cache evaluation

This evaluation compares the current RAG runtime with a fixed pre-change
latency artifact from Git revision
`ac5dd1ec3ecadd488bd2a45c691c441b440ca9bf`.

The optimization caches the deserialized index, BM25 runtime, FAISS runtime,
and bounded exact-query vectors. The cache is invalidated when the index file
size or modification time changes.

The comparison is deliberately narrow:

- local Windows process
- in-memory SQLite
- fast agent mode
- forced sparse retrieval
- 30 repeated-route samples

It is useful for catching a regression in repeated local retrieval. It is not
a cloud load test, a dense unique-query benchmark, a provider-network
measurement, a production SLO, or clinical evidence.

Run:

```powershell
python scripts/run_credible_route_latency_sample.py
python scripts/run_retrieval_runtime_cache_eval.py
```
