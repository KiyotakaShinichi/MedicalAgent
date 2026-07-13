# Claim-source alignment hardening

> **Engineering signal only.** This ledger enriches the existing
> claim-source alignment artifact with stronger per-row classification.
> It is **not** clinical-grade entailment, **not** clinical
> validation, and **not** claim-level fact-checking against
> real-world evidence.

The hardening layer never promotes the heuristic validator past its
documented capability. The artifact records the validator method
(`heuristic` by default, `embedding` or `optional_nli` when
`ONCOTRACK_RAG_CLAIM_VALIDATOR` is flipped) so a reviewer can
distinguish "support_status came from token overlap" from
"support_status came from NLI".

- Module: [`backend/services/claim_source_alignment_hardening.py`](../backend/services/claim_source_alignment_hardening.py)
- Script: [`scripts/run_claim_source_alignment_hardening.py`](../scripts/run_claim_source_alignment_hardening.py)
- Artifact: [`Data/evals/rag/latest_claim_source_alignment_hardening.json`](../Data/evals/rag/latest_claim_source_alignment_hardening.json)
- Tests: [`tests/test_claim_source_alignment_hardening.py`](../tests/test_claim_source_alignment_hardening.py)

## Per-row fields

| Field | Meaning |
|---|---|
| `claim_text` | The claim sentence under review. |
| `expected_source_ids` | Gold canonical source IDs. |
| `source_tier` | Concatenated required source tiers (e.g. `T1\|T2\|T3`). |
| `allowed_use` | Coarse `general_patient_education` / `medical_claim_boundary_or_insufficient_evidence` / `portal_help_only` / `unspecified_engineering_default`. |
| `patient_facing_allowed` | False iff `required_source_tiers ⊆ {T4, T5}`. |
| `support_status` | `supported` / `partially_supported` / `unsupported` / `contradicted` / `insufficient_evidence`. |
| `contradiction_category` | One of `tumor_marker_overclaim`, `genetic_vus_overclaim`, `treatment_recommendation`, `dosage_instruction`, `diagnosis_claim`, `prognosis_estimate`, `false_reassurance`, `supplement_replacement`, or `trap_present_pattern_unmatched`. |
| `validator_method` | `heuristic` / `embedding` / `optional_nli`. |
| `alignment_action`, `blocked_rule`, `underlying_passed` | Forwarded from the source artifact. |

## How `support_status` is derived

| Source-artifact condition | Hardened `support_status` |
|---|---|
| `alignment_action` contains "block" or "refuse" AND claim_type is a contradiction trap | `contradicted` |
| `alignment_action` contains "block" or "refuse" otherwise | `unsupported` |
| `alignment_action` or `claim_type` contains "insufficient" | `insufficient_evidence` |
| `source_id_present` AND `alignment_action` contains "keep" | `supported` |
| `source_id_present` otherwise | `partially_supported` |
| else | `insufficient_evidence` |

The classifier is deterministic and side-effect-free.

## Honest reporting

The hardening run is allowed to flag a `trap_present_pattern_unmatched`
contradiction when a row's `claim_type` says contradiction but the
text doesn't trigger any of the 8 pattern families. That is the
honest report — the heuristic is not exhaustive.

## What this hardening does NOT claim

- Not clinical-grade entailment.
- Not factual-correctness certification.
- Not a substitute for clinician review.
- Not a release-gate blocker (it's `status: informational`).

## Related

- [`Data/evals/rag/latest_claim_source_alignment_eval.json`](../Data/evals/rag/latest_claim_source_alignment_eval.json) — source artifact.
- [`docs/evals/eval_contamination_harmonization.md`](evals/eval_contamination_harmonization.md)
- [`docs/ten_out_of_ten_under_constraints.md`](ten_out_of_ten_under_constraints.md)
