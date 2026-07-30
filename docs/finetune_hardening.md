# Fine-Tuning Hardening

This is an internal, synthetic, behavior-only scaffold. It does not tune medical authority and cannot be promoted to patient-facing use.

- Status: `needs_attention`
- Promotion decision: `HOLD`
- Checks passed: `6/12`

## Open Gates

- `semantic_flags_cleared_for_candidate`: Every near-match requires review; contaminated or ambiguous pairs require remediation. Evidence: `Data/evals/models/latest_finetune_semantic_contamination.json`
- `isolated_training_runtime_ready`: PEFT dependencies and runtime probe must pass before training. Evidence: `Data/evals/models/latest_finetune_runtime_preflight.json`
- `baseline_candidate_generations_complete`: Matched baseline and candidate generations are required. Evidence: `Data/evals/models/latest_finetune_promotion_gate.json`
- `candidate_generation_lineage_verified`: Generation manifest must bind model, revisions, holdout hash, and generation hash. Evidence: `Data/evals/models/latest_finetune_promotion_gate.json`
- `candidate_memorization_audit_complete`: Exact train-output memorization must be checked before shadow promotion. Evidence: `Data/evals/models/latest_finetune_promotion_gate.json`
- `paired_statistical_lift_proven`: Exact paired McNemar/binomial evidence is required in addition to raw lift. Evidence: `Data/evals/models/latest_finetune_promotion_gate.json`

## Promotion Meaning

A `PROMOTE` result means eligible for an offline shadow experiment only. It does not mean clinically validated, safe for patients, or approved for deployment.
