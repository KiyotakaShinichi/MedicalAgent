# NLCare Retrieval Ablation Study

Generated from repository artifacts at `2026-08-11T04:48:01.993104+00:00`.

> NLCare remains synthetic-only, non-diagnostic, not clinically validated, and not production healthcare ready. Internal tests are engineering evidence, not evidence of patient benefit or medical effectiveness.

## Experiment

The repository compares BM25, dense FAISS, hybrid RRF, query rewriting, parent-child expansion, source-tier filtering, and section-aware variants. The section experiment uses corrected structural metadata but remains internal and tuning-used.

## Baseline decision

- Complex stack improvement over BM25 proven: `False`.
- Prior evidence showed governance value from source-tier filtering, but did not prove raw retrieval superiority over BM25.
- The citation pruner remains not promoted because it regressed citation precision.

## Section-aware result

- Known section misses evaluated: `319`.
- Recovered misses: `31`.
- Remaining misses: `288`.
- Regression cases: `0`.
- Section hit delta: `0.0`.
- Paper Recall@10 delta: `0.0`.
- Paper precision@5 delta: `0.0`.
- Promoted to live retrieval: `False`.

## Failure attribution

| Stage | Unique case count | Share |
|---|---:|---:|
| section_mismatch | 332 | 0.654832 |
| citation_alignment | 67 | 0.13215 |
| source_tier_filtering | 44 | 0.086785 |
| ranking_failure | 21 | 0.04142 |
| retrieval_miss | 18 | 0.035503 |
| context_assembly | 17 | 0.033531 |
| unknown_unclassified | 8 | 0.015779 |

## Dense serving decision

A fingerprint-matched local dense index is implemented and benchmarkable. Restricted synthetic staging keeps sparse fallback until held-out evidence justifies dense/hybrid quality and latency. This is an evidence-based non-promotion, not a missing feature.
