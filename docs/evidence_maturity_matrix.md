# Evidence Maturity Matrix

No aggregate score is emitted. Evidence volume cannot compensate for a blocking domain or missing independent review.

| Dimension | Tier | Proven | Not proven |
|---|---:|---|---|
| AIE/RAG | 3 (frozen_internal_with_contamination_controls) | Frozen internal case-level comparisons, paired bootstrap intervals, multiple-comparison correction, source governance, and negative results. | Full governed-stack raw Recall@10 superiority over BM25 is not proven; current headline=False. |
| AIE/adversarial safety | 3 (frozen_internal_with_contamination_controls) | Frozen internal adversarial and safe-negative controls. | Unsafe leakage remains 0.07272727272727275; external authorship is absent. |
| LLM token/latency observability | 2 (internal_self_test) | Request totals, local route samples, stage timings, token estimates, and provider usage fields. | Provider usage coverage is 0.0; cost is not billing reconciliation. |
| MLE/statistics | 2 (internal_self_test) | Patient-level temporal splits, leakage/shortcut audits, bootstrap uncertainty, paired tests, calibration and synthetic perturbation sensitivity. | All outcomes and uncertainty remain simulator-bounded; transportability is unproven. |
| XAI | 2 (internal_self_test) | Mechanical additivity, bootstrap set stability, retraining stability, and fail-closed display policy. | Patient display mode is grouped_factors_without_rank_claim; exact feature rank stability is not established. |
| Fine-tuning | 1 (scaffold_or_contract_only) | Behavior-only synthetic dataset, immutable revision contracts, promotion tripwires, and semantic-similarity screening. | Promotion=HOLD; blockers=6. |
| SWE/release discipline | 2 (internal_self_test) | Integrated ship manifest status=passed with repeatable tests and release gates. | Evidence is owner-run; architecture has large modules and no independent clean-clone reproduction. |
| Automation | 2 (internal_self_test) | Local redacted outbox/retry/dead-letter/idempotency contracts status=strong. | External delivery is disabled by default and no live clinician acknowledgement workflow is proven. |
| Infrastructure/deployment | 1 (scaffold_or_contract_only) | Compile-checked reference cloud architecture status=compiled_reference_architecture. | Deployment readiness=needs_attention; no authenticated what-if, deployment, restore, or cloud load evidence. |
| Data engineering | 2 (internal_self_test) | Non-patient bronze/silver/gold lineage pipeline status=strong. | No governed real-patient pipeline, managed deletion proof, or healthcare interoperability evidence. |
| Medical/human factors | 1 (scaffold_or_contract_only) | Deterministic boundaries, evidence policies, escalation language, overtrust warnings, and review packets. | No clinician, nurse, pharmacist, genetic counselor, or real-user review has been completed. |

## Architecture Budget

- Status: `needs_attention`
- Oversized files: `11`
- Critical files: `0`
- Backend service files: `364`

Do not add a new service or artifact solely to increase feature count. New modules should close a measured gap, replace an older surface, or come with a deletion/consolidation plan.

## Boundary

This matrix rates evidence provenance, not product quality or clinical readiness. Internal engineering evidence cannot be averaged into clinical validation, real-world safety, patient benefit, or production healthcare readiness.
