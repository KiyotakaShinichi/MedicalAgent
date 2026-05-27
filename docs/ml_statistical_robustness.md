# ML Statistical Robustness

This artifact adds statistical discipline around the synthetic row-level prediction export.

Run:

```bash
python scripts/run_ml_statistical_robustness.py
```

Output:

```text
Data/evals/models/latest_ml_statistical_robustness.json
```

It reports:

- bootstrap intervals for classification accuracy, Brier score, and ECE
- bootstrap intervals for regression MAE, RMSE, and R2
- subgroup confidence intervals using Wilson intervals
- synthetic label-noise sensitivity
- stability flags for calibration, subgroup small-n, and label-noise brittleness

Boundary: all metrics are synthetic-only. They do not establish real-patient calibration, clinical validation, treatment utility, or patient benefit.
