# Adversarial Holdout V2 Design

This is an internally authored frozen adversarial set for engineering regression only.

It is not external-authored, not clinician-reviewed, and not clinical validation. It was created to expose whether unsafe-intent routing generalizes beyond the older fixed bank and held-out v1 variants.

## Scope

- 110 cases across 11 unsafe families.
- 10 cases per family.
- Styles include formal, casual, Taglish, indirect, hypothetical, and emotionally phrased prompts.
- Every case has `authored_by`, `authored_date`, `was_used_for_tuning: false`, `expected_route`, `expected_refusal_or_escalation`, `safe_negative`, and a contamination note.

## Families

- privacy / PII request
- prompt injection / boundary override
- cross-patient data exfiltration
- genetic-risk interpretation
- VUS misinterpretation
- diagnosis confirmation
- tumor-marker conclusion
- treatment change
- dosage request
- prognosis / survival estimate
- supplement replacement / natural-cure claim

## Commands

```bash
python scripts/build_adversarial_holdout_v2.py
python scripts/run_adversarial_generalization_v2_eval.py
```

## Interpretation

Low scores should be treated as useful evidence of weak generalization, not as a reason to silently tune the set. If v2 is used for hardening, create v3 or external-author cases before claiming further generalization.

Current claim boundary: this is an engineering safety-regression artifact only. It does not prove clinical safety, real-world robustness, patient benefit, or clinician approval.
