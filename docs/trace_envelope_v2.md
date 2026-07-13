# Trace Envelope V2

Trace envelope v2 stores redacted decision metadata for engineering observability only. It is not clinical validation, not a medical record, not clinician review, and not a chain-of-thought log.

## Status

- Status: `strong`
- Clinical validation: `False`
- Validation pass rate: `1.0`
- Forbidden field catch rate: `1.0`

## Required Fields

- `schema_version`
- `generated_at`
- `request_id`
- `correlation_id`
- `patient_id_hash`
- `route`
- `intent`
- `safety_decision`
- `policy_decision`
- `retrieval_backend`
- `source_ids`
- `claim_validation`
- `post_generation_decision`
- `cache_status`
- `latency_ms`
- `estimated_cost`
- `final_policy_status`
- `clinical_validation`
- `claim_boundary`

## Forbidden Keys

- `chain_of_thought`
- `draft_response`
- `full_chat_transcript`
- `patient_id`
- `private_chain_of_thought`
- `raw_patient_identifier`
- `raw_patient_message`
- `raw_prompt_with_secrets`
- `reasoning_text`
- `scratchpad`
- `unredacted_phi`
