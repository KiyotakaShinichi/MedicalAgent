# RAG Baseline Comparison

This artifact compares NLCare's retrieval stack against simpler baselines on the
same frozen internal RAG retrieval goldset:

1. BM25-only retrieval
2. FAISS dense/vector-only retrieval
3. Dense + sparse hybrid retrieval with RRF-style fusion
4. Hybrid + query rewriting
5. Hybrid + query rewriting + parent-child context expansion
6. Hybrid + query rewriting + parent-child + source-tier filtering

Outputs:

- `Data/evals/rag/latest_rag_baseline_comparison.json`
- `Data/evals/rag/latest_rag_baseline_failures.json`

Run:

```bash
python scripts/run_rag_baseline_comparison.py
```

## What It Measures

- `Recall@5` / `Recall@10`: mean fraction of expected logical sources retrieved
  in the top-k contexts.
- `MRR`: first relevant source rank.
- `NDCG@10`: rank-sensitive source relevance.
- `citation_precision`: fraction of the top cited contexts that match an
  expected source.
- `claim_support_rate`: percentage of cases with at least one expected source
  retrieved.
- `unsupported_context_rate`: percentage of cases with no expected source in
  the top 10.
- `refusal_correctness`: policy/intent proxy only. Generated-answer refusal is
  evaluated by live-agent safety evals.
- `source_tier_correctness`: whether selected contexts satisfy the case's
  source-tier/allowed-use policy.
- `latency_p50_ms` / `latency_p95_ms`: local retrieval-stage latency. Model
  load and cache warmth can affect these numbers.
- `case-level failures`: concrete retrieval misses, partial recall, low citation
  precision, source-tier mismatches, or refusal-policy mismatches.

## Current Result

Latest internal run on the expanded 74-case frozen goldset:

- BM25 Recall@10: `0.7635`
- Best raw Recall@10: `0.8514` from `hybrid_rrf_query_rewrite`
- Full filtered stack Recall@10: `0.7703`
- Full filtered stack MRR: `0.6153`
- Full filtered stack source-tier correctness: `1.0`
- Full filtered stack unsupported-context rate: `0.1757`
- Full filtered stack p95 retrieval latency: `690.276 ms`
- `improvement_proven_vs_bm25`: `false`

Interpretation: hybrid retrieval plus query rewriting improves raw source
finding on this harder internal set. Source-tier filtering preserves governance
but drops some relevant contexts, so the full filtered pipeline shows only a
small Recall@10 lift over BM25 and is not marked as a proven broad improvement.
Do not claim the complex stack is universally better unless future frozen or
external-authored sets support that claim.

## Goldset Governance

The suite uses `Data/evals/rag/retrieval_goldset.jsonl`.

The current frozen set has 74 internally authored cases covering:

- easy education
- hard contradiction traps
- no-evidence / insufficient-evidence boundaries
- Taglish/code-switched queries
- genetics and VUS boundaries
- tumor-marker limitation boundaries
- supplement/pharmacist-review boundaries
- urgent symptom escalation
- source-tier and allowed-use filtering

Rules:

- Do not tune retrieval weights, rewrite rules, or source aliases on the frozen
  goldset.
- Keep `was_used_for_tuning=false` for this comparison artifact.
- Treat source ID aliases as logical source normalization only, not ranking
  changes.
- Add future externally authored cases as a separate eval set before claiming
  external credibility.

## Held-out / external-author goldset (PREPARED, NOT COMPLETED)

A separate held-out goldset has been **prepared** for external-author
evaluation under a no-read protocol. It has **not** been completed.

- Protocol: [`docs/evals/no_read_rag_goldset_protocol.md`](no_read_rag_goldset_protocol.md)
- Template: `Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl`
- Template README: `Data/evals/rag/retrieval_goldset_holdout_v2.README.md`
- Runner: `python scripts/run_rag_holdout_baseline_comparison.py`
- Result artifact: `Data/evals/rag/latest_rag_holdout_baseline_comparison.json`

Until a reviewer authors cases under the no-read protocol AND those
cases are placeholder-free, externally-authored, and untuned, the
runner emits `completed: false` with
`status: "ready_for_external_authoring"`. The runner refuses to
fabricate a completed external result by reusing the internal goldset.

## Citation-context pruner (eval-path experiment, current verdict: NOT promoted)

A new configuration ``hybrid_rrf_query_rewrite_parent_child_source_tier_pruned``
has been added to the comparison. It applies
[`backend/services/citation_context_pruner.py`](../../backend/services/citation_context_pruner.py)
**between** source-tier filtering and citation assembly.

The pruner is currently **eval-path only**. The live patient agent's
`apply_intent_aware_rag_layer` does NOT call the pruner yet. Promotion
depends on a clean win on citation_precision / unsupported_context
on a held-out goldset — neither of which has occurred.

### Measured trade on the internal goldset (74 cases)

| Metric | full stack | full stack + pruner | Δ |
|---|---:|---:|---:|
| Recall@5 | 0.7027 | **0.7770** | **+0.0743** ✅ |
| Recall@10 | 0.7838 | 0.7838 | 0 |
| MRR | 0.6892 | 0.6622 | −0.0270 ⚠ |
| nDCG@10 | 0.6453 | 0.6393 | −0.0060 |
| **citation_precision** | **0.5243** | **0.4275** | **−0.0968** ❌ |
| claim_support_rate | 0.8378 | 0.8378 | 0 |
| unsupported_context_rate | 0.1622 | 0.1622 | 0 |
| source_tier_correctness | 1.0 | 1.0 | 0 |
| refusal_correctness | 1.0 | 1.0 | 0 |
| latency p50 | 269ms | 275ms | +6ms |

**Verdict**: the pruner improves top-5 *recall* (more expected sources
make it into top-5) but **regresses citation_precision** (the cited
context now includes more non-expected chunks because the composite-
score reorder pushes some high-overlap chunks below new entrants).
The brief was explicit: "Do NOT optimize only for Recall@10 if citation
precision worsens." The pruner stays experimental, not promoted.

The pruner's existing test invariants (no goldset-specific input,
preserves metadata, refusal-source retention, clinician-only blocked)
remain green; the regression is on the goldset's citation_precision
distribution, not on the safety contract.

### Failure analysis behind the regression

`Data/evals/rag/latest_citation_precision_failure_analysis.json`
classifies the **full-stack** (pre-pruner) low_citation_precision
failures into 12 owner-tagged categories. The top categories are:

| Category | Owner | Count |
|---|---|---:|
| source_alias_or_metadata_mismatch | metadata | 27 |
| bm25_lexical_distractor | retrieval_ranking | 23 |
| query_rewrite_drift | retrieval_ranking | 21 |
| duplicated_near_equivalent_chunk | context_pruning | 19 |
| parent_child_expansion_too_broad | context_pruning | 18 |
| insufficient_top_k_pruning | context_pruning | 13 |
| expected_gold_source_too_narrow | goldset_design | 9 |
| low_value_safety_policy_chunk_over_selected | citation_assembly | 4 |
| dense_semantic_distractor | retrieval_ranking | 2 |

Owner mix: **context_pruning = 50**, **retrieval_ranking = 46**,
**metadata = 27**, **goldset_design = 9**, **citation_assembly = 4**.

The pruner addresses 50/146 ≈ 34% of root causes by category. The
remaining 66% sit in retrieval ranking, metadata, and goldset
design — none of which the pruner is allowed to touch under this
brief's constraints.

## Stage-wise retrieval oracle diagnostic

After the citation-context pruner's negative result, the next
correct move is *attribution*, not another fix. The stage-wise
diagnostic (`backend/services/rag_stage_oracle_diagnostic.py`)
re-runs every retrieval stage in isolation against the frozen
goldset and records whether an expected source survives that stage.

It is **read-only**: it does not change retrieval ranking, source
governance, the goldset, or any live-agent behaviour. It does **not**
improve retrieval by itself — it tells us *where* grounding is being
lost.

- Module: [`backend/services/rag_stage_oracle_diagnostic.py`](../../backend/services/rag_stage_oracle_diagnostic.py)
- Script: [`scripts/run_rag_stage_oracle_diagnostic.py`](../../scripts/run_rag_stage_oracle_diagnostic.py)
- Artifact: [`Data/evals/rag/latest_rag_stage_oracle_diagnostic.json`](../../Data/evals/rag/latest_rag_stage_oracle_diagnostic.json)
- Tests: [`tests/test_rag_stage_oracle_diagnostic.py`](../../tests/test_rag_stage_oracle_diagnostic.py)

### Current stage-wise attribution (74 cases)

| Stage | Retention | Note |
|---|---:|---|
| corpus_coverage_rate | 1.0000 | Every expected source exists in the KB. |
| bm25_candidate_recall@50 | 0.9865 | BM25 puts an expected source in top-50 for 73/74 cases. |
| dense_candidate_recall@50 | 0.9865 | Dense candidate pool matches BM25. |
| hybrid_candidate_recall@50 | 0.9865 | Hybrid RRF pool matches both. |
| source_filter_retention_rate | **0.8378** | Tier/allowed_use filter drops the expected source for 12 / 74 (16%). |
| citation_window_retention_rate | 0.8378 | The top-10 citation window does NOT lose anything past the source filter. |
| oracle_recall@10_upper_bound | 0.8378 | What an ideal reranker could achieve on the post-filter window. |
| actual_full_stack_recall@10 | 0.7838 | What the current ranker achieves. |
| **oracle_gap** | **0.0540** | Gap attributable to ranking, not governance. |

### Failure-stage attribution (74 cases)

| Final stage | Count | Owner |
|---|---:|---|
| no_failure | 60 | — |
| **source_filter_drop** | **9** | source governance (working as designed) |
| rrf_ranking_failure | 2 | ranking |
| dense_failure | 1 | dense retriever |
| sparse_failure | 1 | BM25 |
| citation_window_drop | 1 | window selection |

**Interpretation**:

- **The bottleneck is source-tier filtering, not retrieval.** Candidate
  recall is 98.65% across all three retrievers; the gold is *there*.
  Source governance drops it from 98.65% to 83.78%. That drop is
  **intentional** — the filter exists to keep clinician-only / stale
  / out-of-allowed-use chunks out of patient-facing citations.
- **The ranking gap is small.** Only 5.4pp separates the oracle upper
  bound (0.8378) from the actual full stack (0.7838). 5 of those
  5.4pp are RRF-rank / dense-only / sparse-only / citation-window
  problems.
- **Query rewrite drift, parent-child noise, and alias mismatch do
  not appear** as failure stages in the current run.
- The 9 source_filter_drop cases are *not* a bug to fix by weakening
  the filter. They are evidence that the goldset's
  `expected_source_ids` include sources the patient-facing filter is
  correctly designed to exclude. This is a goldset-design conversation,
  not a retrieval one.

## Honest interpretation of current numbers

- The full source-governed stack currently has Recall@10 ≈ 0.78 on
  the internal goldset. BM25 has Recall@10 ≈ 0.80.
  `complex_stack_improvement_over_bm25 ≈ −0.02`.
- `improvement_proven_vs_bm25` is **false**. The full stack is
  valuable for governance and safety routing; **raw retrieval
  superiority over BM25 is not proven** on this goldset.
- The 0.7838 number is in-sample (the internal goldset has shaped
  the alias map, the failure analysis, and several thresholds).
  The held-out v2 result is the closer-to-honest signal once it
  exists.

## Claim Boundary

This is engineering evidence for retrieval and grounding, not medical
correctness, clinical validation, clinician approval, or real-world safety
evidence. It does not prove that generated answers are clinically safe.
