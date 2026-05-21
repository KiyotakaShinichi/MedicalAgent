# External-Author Eval Packet

Status: unreviewed and prepared for future review.

This packet is for external authors who can write RAG or adversarial eval cases without first reading prompts, safety rules, code internals, or existing gold cases. It is not clinical validation and does not imply clinician approval.

## Reviewer Instructions

- Do not inspect prompts, guardrail lists, source code, or existing gold cases before authoring.
- Write cases from the perspective of a patient, clinician, or reviewer.
- Mark whether each case is patient-facing, clinician-facing, or admin-facing.
- Include expected route, expected answerability status, expected refusal/escalation behavior, acceptable source-tier expectations, and contamination notes.
- Label your role and date.

## Review Output

Use:

- `Data/evals/review_templates/external_author_rag_cases_template.jsonl`
- `Data/evals/review_templates/external_author_adversarial_cases_template.jsonl`

Review purpose: evaluation credibility and safety-boundary testing only. This is not clinical approval.
