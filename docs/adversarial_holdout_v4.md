# Adversarial Holdout V4

V4 is a fresh internal holdout created after v3-driven generalized hardening.

Run:

```bash
python scripts/build_adversarial_holdout_v4.py
python scripts/run_adversarial_holdout_v4_eval.py
python scripts/run_adversarial_hardening_report.py
```

Artifacts:

```text
Data/evals/safety/adversarial_holdout_v4.jsonl
Data/evals/safety/latest_adversarial_holdout_v4_baseline.json
Data/evals/safety/latest_adversarial_hardening_report.json
```

Rules:

- V3 can now be treated as a hardening/dev set because it was used for generalized fixes.
- V4 is the new untuned internal holdout baseline.
- Do not tune on V4 without creating V5 or collecting external-author cases.
- These are engineering safety-regression checks, not clinical safety proof.
