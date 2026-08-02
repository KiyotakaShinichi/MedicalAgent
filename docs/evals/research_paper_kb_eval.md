# Research-Paper KB Evaluation

## Purpose

This suite measures whether NLCare can retrieve and preserve provenance for the
research papers that are actually present in its local knowledge base. It is an
engineering regression test for source identity, paper discrimination, section
retrieval, source-tier enforcement, and unsafe-premise handling.

It does **not** evaluate clinical correctness, completeness of the medical
literature, real-patient safety, or clinical validation.

## Current corpus

The research-paper manifest contains nine open-access PubMed Central papers.
The corpus is concentrated in two topic groups:

- MRI/DCE-MRI response-monitoring research.
- Chemotherapy-associated neutropenia, hematologic toxicity, G-CSF, and
  supportive-care research.

This is not a broad breast-oncology literature collection. Genetics/VUS,
tumor-marker interpretation, treatment selection, supplements, prognosis, and
many other topics remain governed boundary or no-evidence cases rather than
paper-backed coverage claims.

## Case design

`Data/evals/rag/research_paper_grounding_cases.jsonl` contains 32 cases:

- 9 exact-title retrieval cases.
- 9 topic paraphrases, including Taglish.
- 9 section-anchor cases using details found inside the papers.
- 5 no-research-evidence boundary cases.

The cases were authored from the evaluated KB. They are marked
`was_used_for_tuning: false`, and this pass must not tune retrieval against
their failures. They are still **internal corpus-derived cases**, not an
independent holdout.

## Compared retrieval configurations

1. BM25 only.
2. FAISS dense only.
3. Dense + sparse hybrid with reciprocal-rank fusion.
4. Hybrid + query rewriting.
5. Hybrid + query rewriting + parent-child expansion.
6. Hybrid + query rewriting + parent-child expansion + source-tier filtering.

## Metrics

- Recall@5 and Recall@10 by expected PMCID.
- Mean reciprocal rank and nDCG@10.
- Top-1 paper identity accuracy.
- Expected-section hit rate.
- Taglish Recall@10.
- Provenance completeness for matched paper chunks.
- Source-tier correctness after governance.
- No-evidence false-paper-attribution rate.
- Pre-retrieval boundary-route correctness.
- p50/p95 retrieval latency.
- Paired Recall@10 delta, deterministic bootstrap interval, and exact sign test
  for the governed stack versus BM25.

## Artifacts

- `Data/evals/rag/latest_research_paper_kb_audit.json`
- `Data/evals/rag/latest_research_paper_retrieval_eval.json`
- `Data/evals/rag/latest_research_paper_retrieval_failures.json`

Run:

```powershell
python scripts/run_research_paper_kb_eval.py
```

## Latest internal result

The first untuned run is `needs_attention`:

| Configuration | Recall@10 | MRR | Top-1 paper | Section hit | No-evidence false paper attribution |
|---|---:|---:|---:|---:|---:|
| BM25 only | 1.000 | 0.9568 | 0.9259 | 0.9444 | 0.800 |
| FAISS dense only | 0.963 | 0.9012 | 0.8519 | 0.7778 | 0.400 |
| Hybrid RRF | 0.963 | 0.9630 | 0.9630 | 0.8889 | 0.400 |
| Full source-governed stack | 0.963 | 0.9630 | 0.9630 | 0.8889 | 1.000 |

All matched paper chunks preserved the required provenance fields, all nine
manifest PMCIDs were represented, and the false PMCID identity count is zero.
However, the full stack did not beat BM25 on Recall@10: paired delta `-0.037`,
95% bootstrap interval `[-0.111111, 0.0]`, exact sign-test `p=1.0`.
`paper_retrieval_improvement_proven_vs_bm25` therefore remains `false`.

The raw full-stack retriever returns a T2 paper for every unsupported-premise
case. Pre-retrieval boundary routing catches 3/5 of those cases. This is why the
overall result is `needs_attention` despite high positive-case recall. The two
misses are logged for a future generalized safety change; this pass does not
tune on the benchmark.

Latency values are evaluator-local measurements. BM25 is rebuilt per query in
the baseline implementation while hybrid candidates share the comparison
cache, so these timings are useful for regression detection but not a fair
production latency claim.

## Interpretation rule

A positive internal result only shows that a configuration retrieves this
narrow local corpus more reliably under these corpus-derived queries. A
negative result remains visible. Neither result establishes medical
authority, independent generalization, patient benefit, or production
healthcare readiness.
