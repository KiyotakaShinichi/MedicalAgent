# Automation and XAI Industry Alignment

This roadmap is engineering governance only. It is not clinical validation, not clinician sign-off, not emergency coverage, not proof of patient benefit, and not healthcare production readiness.

## Summary

- Status: `strong`
- Automation live delivery enabled: `False`
- Healthcare production ready: `False`
- Automation controls: `6`
- XAI controls: `6`

## Automation Controls

### outbox_first_source_of_truth

- Reason: External channels fail; the system needs a local auditable queue before dispatch.
- Minimum requirement: Every high-risk event creates a local alert row before any webhook/email/SMS/Viber attempt.
- Current status: `implemented_as_engineering_preview`

### redacted_signed_event_envelope

- Reason: Webhook payloads should not leak raw chat text or patient identifiers.
- Minimum requirement: Use redacted payloads, HMAC signatures, timestamp tolerance, and replay protection.
- Current status: `implemented_as_engineering_preview`

### idempotency_and_deduplication

- Reason: Retries must not create duplicate clinical-review tasks.
- Minimum requirement: Stable event_id/idempotency_key across retries and dedupe checks in the receiver.
- Current status: `partially_documented_needs_ui_visibility`

### retry_dead_letter_and_requeue

- Reason: Operators need to see failed notification attempts and recover safely.
- Minimum requirement: Bounded retries, dead-letter reason, manual requeue, and no loss of local alert.
- Current status: `implemented_as_contract_needs_operator_ui`

### delivery_receipt_not_human_acknowledgement

- Reason: Transport delivery is not the same as clinician review.
- Minimum requirement: Separate delivery receipt, opened/reviewed, and manual acknowledgement states.
- Current status: `implemented_as_boundary_needs_dashboard_card`

### test_recipient_only_external_channels

- Reason: Without clinical operations ownership, real alerting can create false assurance.
- Minimum requirement: Email/SMS/Viber/n8n are disabled by default and limited to synthetic test recipients.
- Current status: `implemented`

## XAI Controls

### explanation_contract_per_surface

- Reason: Each patient-visible number needs meaning, calculation, uncertainty, and safe next step copy.
- Minimum requirement: Expose a typed explanation envelope for every KPI/model/review-count surface.
- Current status: `specified_needs_api_contract`

### model_card_and_feature_dictionary

- Reason: Reviewers need to know model version, inputs, missingness handling, and synthetic-only limits.
- Minimum requirement: Attach model-card and feature-dictionary links to each synthetic ML output.
- Current status: `documented_needs_frontend_drawer`

### uncertainty_and_abstention_first

- Reason: Low evidence should reduce confidence or abstain, not produce more decisive language.
- Minimum requirement: Show modalities present/missing, abstention reason, confidence source, and known weakness.
- Current status: `partially_implemented`

### non_causal_feature_contributions

- Reason: Feature importance can be mistaken as clinical causality.
- Minimum requirement: Label contribution displays as non-causal synthetic engineering explanations.
- Current status: `planned`

### retrieval_grounding_visibility

- Reason: RAG answers need visible evidence limits, especially when citation precision is weak.
- Minimum requirement: Show answerability, citation support, source-tier policy, and unsupported-context warnings.
- Current status: `partially_implemented`

### negative_results_visible_to_reviewers

- Reason: Credibility improves when failed experiments are not hidden.
- Minimum requirement: Keep pruner regression, BM25 comparison, reranker non-proof, and held-out weaknesses visible.
- Current status: `implemented_as_governance_artifacts`

## Ranked Backlog

1. Build an Automation Center admin card - Shows outbox state, test-recipient delivery receipts, retry/dead-letter status, and manual acknowledgement separately.
2. Add typed patient-XAI explanation envelopes - Turns every patient-visible number into meaning + calculation + uncertainty + allowed next action.
3. Add model-card and feature-dictionary drawers - Makes synthetic ML outputs auditable without making them look clinically authoritative.
4. Run generated-answer A/B before any citation-window change - A smaller citation window may improve precision but could reduce answer support; live behavior should not change from retrieval-only metrics.
5. Create n8n inactive templates for escalation digest and dead-letter review - Industry-aligned automation can be demonstrated without enabling real patient alerting.

## Still Not Industry Ready

- No real clinical operations owner or on-call process.
- No real PHI channel review or compliance sign-off.
- No external clinician acknowledgement workflow.
- No real patient data or clinical validation.
- No live alerting beyond synthetic test-recipient scaffolding.