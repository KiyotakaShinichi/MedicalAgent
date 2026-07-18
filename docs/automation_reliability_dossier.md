# Automation Reliability Dossier

Automation reliability evidence is engineering evidence only. It does not prove emergency coverage, clinician receipt, patient benefit, clinical validation, compliance, or healthcare production readiness. All external notification channels remain redacted, test-recipient-only unless an operator deliberately configures them under a separate security and clinical-review process.

## Summary

- Status: `strong`
- Checks: `8/8`
- External delivery enabled by default: `False`
- Real emergency coverage claim: `False`

## Required Invariants

- `PASS` `local_outbox_first`: High-risk conversation events are written to the local review outbox before external dispatch.
- `PASS` `redacted_signed_webhook`: n8n/webhook events use signed redacted envelopes and block PHI-like payload fields.
- `PASS` `inactive_import_templates`: n8n templates are inactive by default and are optional scaffolds, not live clinical automation.
- `PASS` `test_recipient_only_high_risk_delivery`: High-risk review alerts require synthetic test-recipient mode when external dispatch is enabled.
- `PASS` `delivery_receipt_not_acknowledgement`: Channel delivery receipts are explicitly not treated as clinician acknowledgement.
- `PASS` `retry_dead_letter_contract`: Failed delivery attempts are retried with bounded attempts and then dead-lettered without losing the local alert.
- `PASS` `preview_only_schedule_plan`: Scheduled automation is documented as ready for scheduler/n8n, but no host scheduler is installed automatically.
- `PASS` `dry_run_control_plane`: The automation control plane queues redacted engineering jobs in dry-run mode and sends no webhooks while building artifacts.

## Channel Matrix

- `email`: Notify an internal demo/reviewer inbox that a redacted review item exists. Status: `disabled_by_default`.
- `sms`: Optional short redacted engineering notification to a test maintainer number. Status: `disabled_by_default`.
- `viber_or_chatops`: Optional internal demo channel notification for reviewer workflow visibility. Status: `disabled_by_default`.
- `admin_dashboard`: Primary source of truth for queued/notified/acknowledged review items. Status: `local_demo_source_of_truth`.

## Automation Center Visibility Requirements

- `local_outbox_status`: Local outbox status - The dashboard remains the source of truth even when external channels are configured.
- `delivery_receipt_status`: Delivery receipt status - Delivery status is transport evidence only and must not be confused with clinician acknowledgement.
- `retry_dead_letter_visibility`: Retry/dead-letter visibility - Operators need failure visibility without losing the local review item.
- `redaction_and_signature_visibility`: Redaction/signature visibility - Webhook safety should be inspectable before any optional n8n/email/SMS/Viber path is used.
- `claim_boundary_visibility`: Claim-boundary visibility - Automation must not make the product look clinically monitored or production-ready.

## What This Does Not Prove

- No clinical validation.
- No clinician receipt or response guarantee.
- No emergency coverage.
- No compliance certification.
- No healthcare production readiness.
