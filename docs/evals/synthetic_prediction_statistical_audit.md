# Synthetic Prediction Statistical Audit

This audit reads the existing row-level synthetic prediction export and adds:

- 1,000 patient-row bootstrap replicates for AUROC, Brier score, ECE, and
  regression MAE intervals;
- an exact two-sided McNemar comparison against logistic regression, a
  4,000-replicate paired accuracy-delta interval, and a Wilson interval over
  discordant wins;
- selective-risk curves showing coverage, abstention, and covered accuracy;
- threshold sensitivity, prevalence reweighting, and molecular-subtype slices
  with small-n warnings;
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

The audit returns `needs_attention` when near-perfect simulator metrics coexist
with unproven paired superiority, small subgroup intervals, or outcome-only
perturbations. That status is an evidence conclusion, not a script failure.
