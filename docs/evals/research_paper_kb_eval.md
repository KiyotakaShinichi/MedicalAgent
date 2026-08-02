# Research-Paper KB Evaluation

## Purpose

This suite measures whether NLCare can retrieve and preserve provenance for the
research papers that are actually present in its local knowledge base. It is an
engineering regression test for source identity, paper discrimination, section
retrieval, source-tier enforcement, and unsafe-premise handling.

It does **not** evaluate clinical correctness, completeness of the medical
literature, real-patient safety, or clinical validation.

## Current corpus

The research-paper manifest contains 21 PubMed Central papers selected through
the NCBI open-access path. The corpus now spans:

- MRI/DCE-MRI response-monitoring research.
- Chemotherapy-associated neutropenia, hematologic toxicity, G-CSF, and
  supportive-care research.
- Electronic patient-reported symptom monitoring and PRO-CTCAE measurement.
- Genetics/VUS standards and reclassification boundaries.
- Tumor-marker limitations, oncology distress, and herb-drug interactions.
- Breast-cancer digital monitoring and symptom self-management studies,
  including a negative primary-outcome trial.

This is not a broad breast-oncology literature collection. Genetics/VUS,
tumor-marker interpretation, treatment selection, supplements, prognosis, and
many other topics remain governed boundary or no-evidence cases rather than
paper-backed coverage claims.

## Case design

`Data/evals/rag/research_paper_grounding_cases_v2.jsonl` contains 44 cases:

- 21 exact-title retrieval cases.
- 15 topic, Taglish, and negative-result cases.
- 8 no-research-evidence boundary cases.

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
- `Data/evals/rag/latest_research_paper_query_telemetry.json`
- `Data/evals/rag/latest_research_paper_query_telemetry_failures.json`

Run:

```powershell
python scripts/download_research_papers.py
python scripts/ingest_knowledge_base.py
python scripts/run_kb_source_governance.py
python scripts/run_research_paper_kb_eval.py
python scripts/run_research_paper_query_telemetry.py
```

The downloaded article files and local retrieval index are reproducible local
inputs rather than committed binary/data payloads. A clean checkout must run
the download and ingestion steps before the evaluation commands.

## Latest internal result

The expanded untuned v2 run is `needs_attention`:

| Configuration | Recall@10 | MRR | Top-1 paper | Section hit | No-evidence false paper attribution |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.9722 | 0.9236 | 0.8889 | 0.2667 | 0.625 |
| Full source-governed stack | 1.0000 | 0.9792 | 0.9722 | 0.2667 | 1.000 |

All matched paper chunks preserved the required provenance fields, all 21
manifest PMCIDs were represented, and the false PMCID identity count is zero.
The full stack's Recall@10 delta versus BM25 is `+0.0278`, but the deterministic
bootstrap interval is `[0.0, 0.083333]` and the exact sign-test is `p=1.0`.
`paper_retrieval_improvement_proven_vs_bm25` therefore remains `false`.

The raw full-stack retriever returns a T2 paper for every unsupported-premise
case. Pre-retrieval boundary routing catches 4/8 of those cases. This is why the
overall result is `needs_attention` despite high positive-case recall. The four
misses are logged for a future generalized safety change; this pass does not
tune on the benchmark.

The separate 30-query telemetry suite runs the real patient-agent pipeline and
records route, cache state, cited PMCIDs, stage and wall latency, estimated
tokens, provider-reported tokens when available, and usage coverage. It stores
no generated response text. Cold-start and warm-path latency are reported
separately so local encoder initialization is not hidden by aggregate p95.
Offline token estimates and local latency are not
provider billing data or production traffic evidence.

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
