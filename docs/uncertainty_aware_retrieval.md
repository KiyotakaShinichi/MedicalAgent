# Uncertainty-aware retrieval routing

OncoTrack classifies the assistant's confidence in its own retrieval +
citation evidence before producing an answer. This lets the answer
composer pick between fully-cited answer, hedged answer, refusal,
conflict surface, clinician deferral, or safety-driven refusal.

## Status values

The routing emits one of six values for `answerability_status`:

| Status | When |
|---|---|
| `answerable_with_citations` | retrieval/tier/support confidences all above floor |
| `answerable_with_limited_context` | at least one axis middling, none below floor |
| `insufficient_evidence` | retrieval confidence < 0.3 OR support confidence < 0.3 OR no T1/T2 chunks |
| `conflicting_evidence` | claim validator flags both supported and contradicted claims |
| `clinician_review_required` | record-explanation intent with low citation support |
| `refuse_due_to_safety` | safety scope is high-risk (dominates all other signals) |

## Confidence signals

- `retrieval_confidence` (float, 0–1) — top-score saturation against
  the 0.6 knee.
- `source_tier_confidence` (float, 0–1) — fraction of top-k chunks
  in T1/T2, with T3 counted half.
- `citation_support_confidence` (float, 0–1) — `(supported -
  contradicted) / total` from the claim validator, clamped to [0, 1].
- `evidence_conflict_flag` (bool) — true when both supported and
  contradicted claims exist in the same envelope.

## Precedence

`refuse_due_to_safety` always wins. After that, `conflicting_evidence`
> `clinician_review_required` > `insufficient_evidence` >
`answerable_with_limited_context` > `answerable_with_citations`.

## Files

- Module: [`backend/services/retrieval_confidence.py`](../backend/services/retrieval_confidence.py)
- Probe script: [`scripts/run_uncertainty_aware_retrieval_eval.py`](../scripts/run_uncertainty_aware_retrieval_eval.py)
- Eval JSON: [`Data/evals/rag/latest_uncertainty_aware_retrieval_eval.json`](../Data/evals/rag/latest_uncertainty_aware_retrieval_eval.json)
- Tests: [`tests/test_retrieval_confidence.py`](../tests/test_retrieval_confidence.py)

## Wiring into the pipeline

The module is pure: it does not call FAISS or any LLM. The caller
passes already-scored chunks and an already-validated claim envelope.
Wiring into `agent_rag.run_patient_agent_pipeline` is a follow-up
step (PART 6 will plumb the result into the trace surface).
