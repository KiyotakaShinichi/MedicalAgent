# Real-Time OOD / Drift / Data-Quality Gate

NLCare includes an inference-time OOD/data-quality gate before synthetic ML heads.

This is an engineering guardrail. It does not prove clinical safety, real-world robustness, or patient benefit.

## Checks

- Lab values outside broad physiological bounds.
- Unknown or mismatched units.
- Impossible or invalid dates.
- Missing imaging and low modality availability.
- Suspicious prompt-injection-like strings in structured fields.

## Behavior

- `none`: allow.
- `mild`: lower confidence and log a warning.
- `moderate`: lower confidence and record reasons.
- `severe`: abstain or route for clinician review.

The patient report bundle includes an `ood_gate` field so reviewers can see whether confidence was lowered or scoring was blocked.

## Evaluation

Run:

```bash
python scripts/run_realtime_ood_eval.py
```

Artifact:

```text
Data/evals/ops/latest_realtime_ood_eval.json
```
