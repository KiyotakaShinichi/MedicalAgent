# External adversarial red-team quickstart

> **30-minute path** for an external reviewer to submit fresh
> adversarial safety cases under the no-read protocol. The reviewer
> writes the cases; the repo owner runs the validator. No clinician
> sign-off, no IRB, no clinical claim is implied.

## Who this is for

- External peer engineers / senior MLEs willing to author 5–15
  adversarial cases.
- People who have **not** read the existing adversarial bank, its
  failure analyses, or `backend/services/agent_safety.py`'s vocabulary
  tables.

If you have read any of those, your cases are filed as **internal**
(not external held-out). The repo owner cannot verify which files you
read — the no-read attestation is on your honour.

## What you'll commit (3 files)

1. **Cases JSONL** — your adversarial cases, one per line.
   Template: `Data/evals/external_review/adversarial_case_submission_template.jsonl`.
   Save your filled file as
   `Data/evals/external_review/adversarial_cases_<role>_<date>.jsonl`.
2. **Intake form** — `Data/evals/external_review/<role>_<date>_intake.md`,
   filled from `reviewer_intake_template.md`.
3. **Attestation** — `Data/evals/external_review/<role>_<date>_attestation.md`,
   filled from `reviewer_attestation_template.md`. `engagement_type`
   must be `adversarial_case_authoring`.

## Per-case fields

| Field | Notes |
|---|---|
| `case_id` | `EXT-ADV-NNN` — unique, three-digit. |
| `category` | One of the 13 categories listed in the template. |
| `language` | `en`, `tl`, `es`, or `mixed`. |
| `query` | One natural-language sentence the agent would receive. |
| `expected_safety_level` | `high_risk` or `low_risk`. |
| `expected_scope` | One of the agent_safety scope values when applicable. |
| `expected_security_blocked` | Required when `attack_layer == "security"`. |
| `attack_layer` | `safety_scope`, `security`, or `negative_control`. |
| `notes` | One-line description of the attack family. |
| `authored_date` | ISO `YYYY-MM-DD`. |
| `authored_by` | Role descriptor only; no real name. |
| `was_used_for_tuning` | `false`. **Must stay false.** |
| `case_source` | `external_author_red_team_v1`. |
| `clinical_validation` | `false`. |

## What the validator checks

Once your three files are committed, the repo owner runs:

```
python -c "from backend.services.external_red_team_readiness import build_readiness; \
  import json; print(json.dumps(build_readiness(), indent=2))"
```

The readiness builder counts your case-file rows as **external-author
adversarial cases** only if:

- the file path matches
  `Data/evals/external_review/adversarial_cases_*_*.jsonl`,
- every row has `case_source == "external_author_red_team_v1"`,
- every row has `was_used_for_tuning == false`,
- every row has `authored_by` populated AND not equal to
  `"engineering"` / `"oncotrack_team"` / `"oncotrack_team+claude_codex"`,
- a matching `<role>_<date>_attestation.md` file is committed.

If any rule fails, the readiness reports `completed_external_cases == 0`
and lists the offending rows in `disqualified_rows`.

## What we never do with your cases

- Tune the safety vocabulary against them. `was_used_for_tuning: false`
  is monitored across releases.
- Show them to anyone who has read the existing bank.
- Quote your case-level pass/fail without your attestation file
  alongside.
- Imply clinician sign-off or clinical validity from your work.

## Boundary acknowledgements

By submitting cases under this quickstart, the reviewer agrees their
work:

- **Is not** clinician sign-off, IRB clearance, or clinical validation.
- **Does not** authorise the project owner to claim regulatory or
  patient-safety status.
- May be acknowledged by role descriptor only. No real-name byline
  will be added.

## Related

- [`docs/review_packets/INDEX.md`](INDEX.md)
- [`docs/review_packets/reviewer_outreach_message_templates.md`](reviewer_outreach_message_templates.md)
- [`docs/review_packets/review_execution_checklist.md`](review_execution_checklist.md)
- [`docs/evals/no_read_rag_goldset_protocol.md`](../evals/no_read_rag_goldset_protocol.md)
