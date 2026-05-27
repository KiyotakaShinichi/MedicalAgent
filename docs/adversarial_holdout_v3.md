# Adversarial Holdout V3

`adversarial_holdout_v3.jsonl` is a newly frozen internal adversarial baseline.

It covers privacy/PII, prompt injection, cross-patient exfiltration, genetics/VUS, diagnosis confirmation, tumor-marker overclaim, treatment changes, dosage requests, prognosis/survival, supplement replacement, and safe educational negative controls.

Run:

```bash
python scripts/build_adversarial_holdout_v3.py
python scripts/run_adversarial_holdout_v3_eval.py
```

Artifacts:

```text
Data/evals/safety/adversarial_holdout_v3.jsonl
Data/evals/safety/latest_adversarial_holdout_v3_baseline.json
```

Rules:

- Do not tune directly on v3 in the same pass that creates it.
- If v3 is later used for hardening, create v4 or collect external-author cases before claiming further generalization.
- Passing v3 is internal engineering evidence only, not clinical safety proof.
