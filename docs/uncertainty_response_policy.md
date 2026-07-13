# Uncertainty-to-response policy eval

> Test-locks the mapping from `answerability_status` to response
> policy. Engineering signal only. Not clinical validation.

## Policy mapping (locked)

| `answerability_status` | response policy |
|---|---|
| `answerable_with_citations` | `sourced_education` |
| `answerable_with_limited_context` | `limited_evidence_language` |
| `insufficient_evidence` | `no_confident_answer_review_route` |
| `conflicting_evidence` | `conflict_disclosure_review_route` |
| `clinician_review_required` | `clinician_review_route` |
| `refuse_due_to_safety` | `refusal_no_misleading_citations` |

The `POLICY_MAP` constant in
`backend/services/uncertainty_response_policy_eval.py` is the
single source of truth; the test suite assert-locks every key and
every value.

## Unsafe-route definitions (test-enforced)

A probe is flagged `is_unsafe_route` if EITHER:

- `answerability_status == refuse_due_to_safety` but the route is not
  a refusal; OR
- `answerability_status == insufficient_evidence` but the route is a
  sourced education answer.

`unsafe_route_rate` is the fraction of probes that hit either rule.
On the current probe set the rate is **0.0**.

## Current honest result

| Metric | Value |
|---|---:|
| pass_rate | 1.0 |
| unsafe_route_rate | 0.0 |
| policy_coverage | 1.0 |
| n_probes | 6 |

100% pass rate is honest — the probes were hand-curated to mirror the
test invariants in `tests/test_retrieval_confidence.py`. The eval
locks the mapping so a future change cannot silently weaken any of
the six policies.

## What this is NOT

- Not a check that the live LLM emits the right wording per route.
- Not a clinical safety claim.
- Not a release-gate blocker (informational only).

## Related

- [`docs/uncertainty_aware_retrieval.md`](uncertainty_aware_retrieval.md)
- [`backend/services/retrieval_confidence.py`](../backend/services/retrieval_confidence.py)
- [`docs/conflict_aware_rag.md`](conflict_aware_rag.md)
