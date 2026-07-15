# 0009 — Source-alias normalisation for the frozen retrieval goldset

**Status**: accepted
**Date**: 2026-05-27

## Context

`Data/evals/rag/retrieval_goldset.jsonl` (74 cases) labels each case
with `expected_source_ids` — sometimes human-readable canonicals
(`curated-fever-neutropenia`, `infection-safety`, `portal-help`),
sometimes the actual KB chunk parent_id hashes. The KB itself
(`Data/rag_knowledge_base_chunks.json`) stores 16-char hashed
`parent_id`s and human-readable `title` / `source_name` fields.

`Data/evals/rag/latest_retrieval_failure_analysis.json` flags
`gold_source_id_alias_or_metadata_normalization` as the dominant
failure category in the harder goldset (10 of 74 cases). The
retriever brings back the *right content*; the eval can't credit it
because the alias map in
`backend/services/rag_baseline_comparison.LOGICAL_SOURCE_ALIASES`
is incomplete.

Naively expanding the alias map case-by-case is brittle — it tunes
on the goldset.

## Decision

Two-layer system:

1. **Read-only diagnostic**:
   `backend/services/rag_source_alias_coverage.py` walks the
   goldset, walks the live KB, and emits
   `Data/evals/rag/latest_source_alias_coverage.json`.
   For each goldset `expected_source_id`, it lists:
   - the current alias set,
   - which KB `parent_id`s are already covered,
   - **proposed additions** found by content matching the alias key's
     tokens against KB chunk title/source_name/topic, requiring at
     least 2 token overlaps.

2. **Reviewer-gated promotion**: proposed additions are merged into
   `LOGICAL_SOURCE_ALIASES` only after the diagnostic flags them and a
   reviewer (or this ADR) confirms each addition has a clear
   content match. Every promoted parent_id carries a comment:

   ```
   # Discovered by content match (see latest_source_alias_coverage.json).
   "<parent_id>",  # <KB title>
   ```

The diagnostic itself **never changes retrieval ranking** and the alias
map only affects how the *eval* scores recall — not what the live
agent retrieves.

## Initial promotion (2026-05-27)

12 additions across 8 alias keys, all 2-token-overlap matches with
clearly-related KB titles. See
`Data/evals/rag/latest_source_alias_coverage.json` for the exact
artifact the additions were derived from.

| Alias key | parent_id | KB title |
|---|---|---|
| genetic-counseling | `664fb49bb1343408` | Family History Readiness Depth Reference |
| genetic-counseling | `ef3bcc511aad3c2c` | Genetic Counseling Readiness and Family History Intake |
| tumor-marker-context | `5598e2371d2713c4` | Breast Cancer Biomarkers and Tumor Marker Safety |
| curated-tumor-marker-limitations | `5598e2371d2713c4` | (same) |
| supplement-safety | `918edc260afd2d63` | Diagnosis, Treatment, and Supplement Safety Boundaries |
| infection-safety | `9a6347c207d53299` | Hematology, Bleeding, and Infection Review Reference |
| curated-mri-response-terms | `2524619e8115a75d` | DCE-MRI texture features … |
| curated-mri-response-terms | `2a9f2ed73f0b189c` | Early treatment response prediction using DCE-MRI … |
| treatment-side-effects | `24de6c8ad0379f43` | GI Symptoms, Mouth Sores, Neuropathy, and Fatigue |
| treatment-side-effects | `d50090fd5d38a39d` | Symptom Red Flags and Review Hints During Treatment |
| portal-help | `c35c9264029ff9c9` | NLCare Portal Help and Data Entry |
| portal-help | `479e2ce02e7d9e05` | Patient Portal Workflow Reference |

## Measured before/after (in-sample, frozen goldset, 74 cases)

Both numbers are produced by the same script with the same
configurations; only `LOGICAL_SOURCE_ALIASES` was extended.

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| best_recall@10 (hybrid_rrf_query_rewrite) | 0.8514 | 0.8851 | +0.0337 |
| bm25_recall@10 | 0.7635 | 0.8041 | **+0.0406** |
| full_stack_recall@10 | 0.7703 | 0.7838 | +0.0135 |
| complex_stack_improvement_over_bm25 | +0.0068 | **−0.0203** | turned negative |
| full_stack_MRR | 0.6153 | 0.6892 | +0.0739 |
| full_stack_nDCG@10 | 0.5794 | 0.6453 | +0.0659 |
| citation_precision | 0.4189 | 0.5243 | +0.1054 |
| claim_support_rate | 0.8243 | 0.8378 | +0.0135 |
| unsupported_context_rate | 0.1757 | 0.1622 | −0.0135 |
| **improvement_proven_vs_bm25** | false | **false** | unchanged |

**Two honest readings**:

1. **All metrics moved up** because the identifier match now credits
   chunks that were already being retrieved. This is bookkeeping
   correction, not retrieval improvement.
2. **BM25 improved the most** (+0.041) and now beats the full source-
   governed stack (full_stack 0.7838 < BM25 0.8041 →
   `complex_stack_improvement_over_bm25 = −0.0203`). The simpler
   baseline was the most undercredited by the previous aliasing gap.
   The full stack's safety/governance value is unchanged; its raw
   recall lead over BM25 is **not proven** on this fixed goldset.

`improvement_proven_vs_bm25` remains **false** and is now *more*
defensibly so.

## Consequences

- ✅ The 10 cases that previously failed on identifier bookkeeping
  now match against the right KB content. The eval becomes a more
  honest read on retrieval quality.
- ✅ The diagnostic artifact is reproducible and re-runnable. Future
  goldset extensions will surface their own coverage gaps without
  manual triage.
- ⚠ This is **bookkeeping correction, not retrieval improvement**.
  The next baseline comparison run reports raw vs alias-normalized
  recall side by side and `improvement_proven_vs_bm25` is recomputed
  against the in-sample fixed bank. Any raw-vs-alias gap is
  attributable to identifier hygiene, not model quality.
- ⚠ Contamination risk: the alias additions were chosen using the
  goldset's `expected_source_id` strings. Future authors writing new
  goldset cases must **not** read this ADR before authoring; they
  pick fresh canonicals that the diagnostic re-runs against the KB.

## Anti-overclaim rules

- Do not present any post-alias-correction recall number as
  "improvement over BM25" without re-asserting `improvement_proven_vs_bm25`
  on a held-out goldset.
- Do not collapse raw and alias-normalized recall into one figure in
  the README; both numbers must appear together.
- Do not auto-apply diagnostic-proposed aliases. Promotion is
  always reviewer-gated and tied to a specific diagnostic run.

## Reversal cost

Trivial. Revert this commit; the diagnostic stays as a read-only
artifact. The aliases are removed and the previous recall numbers
return verbatim.
