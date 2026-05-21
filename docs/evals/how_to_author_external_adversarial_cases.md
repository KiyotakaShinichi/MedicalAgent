# How To Author External Adversarial Cases

Do not read prompts, code internals, safety-rule lists, or existing adversarial
banks before authoring. The purpose is to reduce contamination.

Each case should include:

- `case_id`
- `category`
- `language`
- `query`
- `attack_layer`
- `surface` (`patient-facing`, `clinician-facing`, or `admin-facing`)
- `expected_safety_level`
- `expected_scope` or `expected_security_blocked`
- `expected_answerability_status`
- `expected_refusal_or_escalation`
- `acceptable_source_tiers` if the case expects an educational answer
- `reviewer_role`
- `authored_by`
- `authored_date`
- `was_used_for_tuning: false`
- `contamination_notes`

Useful categories:

- privacy / PII
- prompt injection
- cross-patient data exfiltration
- treatment change
- dosage
- prognosis
- genetic-risk interpretation
- VUS misinterpretation
- tumor-marker conclusion
- supplement replacement
- safe educational negative controls

Do not include real patient data. This is an engineering evaluation workflow,
not clinical review.
