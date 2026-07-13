# Conflict-aware RAG adjudication eval

> **Eval-only.** Does not change live-agent behaviour. Synthetic
> in-sample only. Not clinical validation.

For each frozen-goldset case, the runner builds **two disjoint
candidate slices** from the post-source-filter top-20:

- **candidate_a** = top-5 chunks
- **candidate_b** = chunks 6–10

It computes the Jaccard overlap of their source IDs and flags
`candidates_conflict = jaccard < 0.34`. It does NOT force consensus.
The system's `answerability_status` is then read to check whether the
conflict was escalated to `conflicting_evidence` /
`clinician_review_required` / `insufficient_evidence` /
`refuse_due_to_safety`.

## Files

- Module: [`backend/services/conflict_aware_rag_adjudicator.py`](../backend/services/conflict_aware_rag_adjudicator.py)
- Script: [`scripts/run_conflict_aware_rag_eval.py`](../scripts/run_conflict_aware_rag_eval.py)
- Artifact: [`Data/evals/rag/latest_conflict_aware_rag_eval.json`](../Data/evals/rag/latest_conflict_aware_rag_eval.json)
- Tests: [`tests/test_frontier_engineering_layers.py`](../tests/test_frontier_engineering_layers.py)

## Current honest result

| Metric | Value |
|---|---:|
| conflict_detection_rate | 0.7973 |
| conflict_resolution_rate | 0.7966 |
| **unsafe_consensus_rate** | **0.1622** |
| escalation_correctness | 1.0 |
| source_tier_correctness | 1.0 |

**Honest finding**: ~80% of cases have meaningfully different
candidate slices and ~80% of those escalate correctly — but a
**16.2% unsafe-consensus rate** remains. That is, 12 of 74 cases
have conflicting candidates AND the system stays at
`answerable_with_*`. Recorded openly; not promoted.

## What this eval does NOT establish

- Does not establish real-world conflict-handling.
- Does not change `apply_intent_aware_rag_layer`.
- Does not promote any prompt or policy.
- Does not constitute clinical validation.

## Related

- [`docs/uncertainty_aware_retrieval.md`](uncertainty_aware_retrieval.md)
- [`docs/iterative_rag_sufficiency.md`](iterative_rag_sufficiency.md)
- [`docs/negative_results_gallery.md`](negative_results_gallery.md)
