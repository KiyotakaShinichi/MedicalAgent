# Eval drift tracking

Tracks headline metrics across releases so a regression in any of them
surfaces explicitly in
[`Data/evals/history/latest_eval_drift_report.json`](../Data/evals/history/latest_eval_drift_report.json).

## How it works

`scripts/update_eval_history.py`:

1. Reads each artifact in
   `backend.services.eval_drift_tracker.METRIC_SOURCES`.
2. Extracts only the headline metrics (one extractor per artifact —
   not the whole JSON).
3. Appends a row to
   `Data/evals/history/eval_history.jsonl` with `release_id`,
   `commit_hash`, `timestamp`, and the extracted metrics.
4. Writes
   `Data/evals/history/latest_eval_drift_report.json` containing the
   delta between the latest row and the previous row, plus a
   `regressions` array of metrics that crossed their 1pp tolerance in
   the wrong direction.

## Tracked metrics

| Metric | Source | Direction |
|---|---|---|
| `patient_temporal_cv.auc_mean` | `latest_patient_temporal_cv.json` | higher is better |
| `patient_temporal_cv.auc_optimism_delta` | `latest_patient_temporal_cv.json` | lower absolute is better |
| `adversarial_safety_regression.overall_attack_block_rate` | `latest_adversarial_safety_regression.json` | higher is better |
| `adversarial_safety_regression.urgent_symptom_rate` | same | higher is better |
| `adversarial_safety_regression.negative_control_rate` | same | higher is better |
| `uncertainty_aware_retrieval.pass_rate` | `latest_uncertainty_aware_retrieval_eval.json` | higher is better |
| `emotional_distress.pass_rate` | `latest_emotional_distress_eval.json` | higher is better |
| `emotional_distress.en_pass_rate` | same | higher is better |
| `emotional_distress.tl_pass_rate` | same | higher is better |

## Files

- Tracker: [`backend/services/eval_drift_tracker.py`](../backend/services/eval_drift_tracker.py)
- Script: [`scripts/update_eval_history.py`](../scripts/update_eval_history.py)
- History: [`Data/evals/history/eval_history.jsonl`](../Data/evals/history/eval_history.jsonl)
- Latest report: [`Data/evals/history/latest_eval_drift_report.json`](../Data/evals/history/latest_eval_drift_report.json)
- Tests: [`tests/test_eval_drift_tracker.py`](../tests/test_eval_drift_tracker.py)

## What this layer is NOT

- Not a release gate by itself — call sites should still run
  `scripts/run_release_gate.py`. The drift report is informational
  and exposes *which* metric changed and by how much.
- Not noise-free. The 1pp threshold avoids flapping but does not
  account for true sampling variance of a small probe set.
