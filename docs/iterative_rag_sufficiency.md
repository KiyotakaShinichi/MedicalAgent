# Iterative evidence-sufficiency RAG eval scaffold

> **Eval-only.** Not wired into the live patient agent. The bounded
> loop runs at most ONE follow-up retrieval per case. Not clinical
> validation; not a retrieval-improvement claim.

The runner replays the frozen internal goldset twice:

1. **Pass 1** — original query through the full source-governed stack.
2. **Pass 2** — IF pass 1's `answerability_status` is in
   `{insufficient_evidence, conflicting_evidence,
   clinician_review_required}`, the runner appends a generic
   intent-keyed hint (e.g. "fever neutropenia infection safety
   reference") and re-retrieves once.

The follow-up hint table is **generic** — no case_ids, no goldset
strings, no per-row patches.

## Files

- Module: [`backend/services/iterative_rag_sufficiency.py`](../backend/services/iterative_rag_sufficiency.py)
- Script: [`scripts/run_iterative_rag_sufficiency_eval.py`](../scripts/run_iterative_rag_sufficiency_eval.py)
- Artifact: [`Data/evals/rag/latest_iterative_rag_sufficiency_eval.json`](../Data/evals/rag/latest_iterative_rag_sufficiency_eval.json)
- Tests: [`tests/test_frontier_engineering_layers.py`](../tests/test_frontier_engineering_layers.py)

## Current honest result

| Metric | Value |
|---|---:|
| initial_answerability_rate | 0.8649 |
| second_pass_answerability_rate | 0.8649 |
| **insufficiency_reduction_rate** | **0.0** |
| unsafe_answer_rate | 0.0 |
| latency_delta_ms | +124.9 |

**Honest finding**: the targeted follow-up hint does NOT improve
answerability on the in-sample 74-case goldset. The +125ms latency
buys zero new answers. This is a real engineering signal — not a
promotion candidate.

## What this scaffold does NOT do

- Does not call the live LLM.
- Does not change `run_patient_agent_pipeline`.
- Does not promote a multi-pass retrieval policy.
- Does not establish clinical validity, real-world recall, or
  generalisation.

## Related

- [`docs/evals/rag_baseline_comparison.md`](evals/rag_baseline_comparison.md)
- [`docs/uncertainty_aware_retrieval.md`](uncertainty_aware_retrieval.md)
- [`docs/negative_results_gallery.md`](negative_results_gallery.md)
