# Synthetic Prediction Statistical Audit

This audit reads the existing row-level synthetic prediction export and adds:

- 1,000 patient-row bootstrap replicates for AUROC, Brier score, ECE, and
  regression MAE intervals;
- an exact two-sided McNemar comparison against logistic regression;
- selective-risk curves showing coverage, abstention, and covered accuracy;
- molecular-subtype slices;
- 30-seed sensitivity distributions for 10% outcome-label noise and 20%
  random outcome missingness.

Run:

```powershell
python scripts/run_synthetic_prediction_statistical_audit.py
```

The perturbation section does not recompute model predictions from noisy
features. It measures sensitivity of reported metrics to label and outcome
availability changes. All intervals are conditional on the simulator-built
dataset. They are not clinical confidence intervals, external validation, or
evidence of fairness or performance in real patients. Promotion is fixed to
`hold_synthetic_only`.
