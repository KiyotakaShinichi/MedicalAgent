# ML Statistical Evidence Dossier

The ML statistical evidence dossier adds uncertainty framing around existing synthetic-only model artifacts. It is meant to improve reporting discipline, not to claim clinical performance.

Artifact:

- `Data/evals/models/latest_ml_statistical_evidence.json`

## What It Adds

- Wilson confidence intervals around reliability-bin empirical rates.
- Approximate two-proportion intervals for scenario-level modality robustness comparisons.
- Scenario-level sign-test framing for robust-vs-champion wins and losses.
- Subgroup confidence intervals and small-n flags.
- Deep-learning candidate ranking with explicit paired-test limitations.
- Patient temporal CV summary with patient-overlap and temporal-order checks.
- A list of raw prediction exports needed for stronger future paired tests.

## What It Does Not Prove

The current dossier mostly wraps summary-level internal artifacts. That means several statistical tests are approximate or descriptive because row-level paired predictions are not always exported yet.

It does not establish:

- clinical validity
- real patient calibration
- real-world subgroup performance
- treatment utility
- patient benefit
- production healthcare safety

## Stronger Future Tests

The next statistical upgrade is to export row-level predictions for each model head and variant, then run:

- paired bootstrap confidence intervals for AUROC, AUPRC, Brier, MAE, and R2
- McNemar tests for paired classification errors
- DeLong-style AUROC comparison where dependencies allow
- calibration slope/intercept with uncertainty
- subgroup calibration intervals
- missingness-stratified confidence intervals
- decision-curve style utility only after clinically meaningful labels exist

All of those remain synthetic-only until clinician-reviewed or external real-world labels are available.
