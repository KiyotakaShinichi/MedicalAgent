# How To Author External RAG Cases

Do not read system prompts, safety rules, implementation files, or existing
gold cases before authoring cases. The goal is independent challenge data.

For each case, include:

- `case_id`
- `user_query`
- `expected_intent`
- `expected_answerability_status`
- `expected_refusal_or_escalation`
- `acceptable_source_tiers`
- `expected_allowed_use`
- `gold_source_ids` if known
- `contradiction_traps`
- `surface` (`patient-facing`, `clinician-facing`, or `admin-facing`)
- `reviewer_role`
- `authored_by`
- `authored_date`
- `contamination_notes`

Keep cases patient-safe. Do not include real patient identifiers.

This process prepares for external-author evaluation; it is not clinician
approval or clinical validation.
