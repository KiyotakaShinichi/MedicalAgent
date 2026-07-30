# Synthetic Model Perturbation and Retraining Evaluation

This benchmark asks whether the controlled temporal ML pipeline survives
changes to its synthetic assumptions. It uses patient-grouped splits and
re-trains the classifier and regressor under:

- measurement noise,
- patient-level modality dropout,
- combined noise and dropout,
- five-percent patient-level label noise, and
- transfer between the default and realism-v2 synthetic generators.

It also compares the legacy full feature set with a guarded feature policy
that removes `mri_percent_change_from_baseline`. That field is definitionally
close to `response_score_percent`, so retaining it can make regression evidence
look stronger than it is.

The benchmark reports patient-level AUROC, Brier score, regression MAE, and
300-resample patient-level bootstrap intervals. Large degradation is retained
as a visible stress failure. The intervals describe sampling variation inside
the simulator; they do not quantify uncertainty about real patients. Every
result remains synthetic-only and promotion is held regardless of score.

Run:

```powershell
python scripts/run_synthetic_model_perturbation_retrain_eval.py
```
