# Model Card: Breast Cancer Treatment Monitoring PoC

> **Headline AUROC — always read both numbers together.**
>
> | Setting                                          | AUROC      | What it means                                                                                                       |
> | ------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------- |
> | External BreastDCEDL / I-SPY1 (logistic baseline)| **0.637**  | The honest signal for real-world generalisation. **Quote this whenever the synthetic number is quoted.**            |
> | External BreastDCEDL CNN baseline (validation)   | **0.420**  | Weak external signal; useful as an honest floor, **not clinical validation**.                                       |
> | TCGA-BRCA public clinical snapshot               | Not computed | Distribution/applicability scaffold only; public TCGA-BRCA does not provide the same longitudinal CBC monitoring target. |
> | Synthetic holdout (champion: gradient boosting)  | **0.995**  | Model fitting its **own** simulator. Useful for simulator regression testing, not a clinical claim.                  |
>
> The external numbers lead because they are the closest available proxy for real-world generalisation. The synthetic generator is part of the project, so a 0.995 there describes the simulator, not patients. Any downstream README, dashboard, or pitch that quotes 0.995 must also quote 0.637 immediately next to it. The model benchmark script (`scripts/run_model_benchmark.py`) emits `honest_reporting_note` to the same effect, so the artifact carries the rule with it. TCGA-BRCA is tracked as an applicability/distribution benchmark until compatible labels and predictions are curated.
>
> **Subgroup calibration finding (added v2 of model benchmark):** Global ECE 0.141 hides per-subgroup ECE up to 0.246 on `HR+/HER2+`/`stage IIIB` buckets — a 9-point disparity in the largest band. Status: `needs_attention`. Detail at `Data/evals/models/latest_model_benchmark.json → subgroup_calibration`.

## Model Purpose

This project is an AI-assisted oncology monitoring proof of concept. The models are intended to support longitudinal review of already-diagnosed breast cancer treatment journeys. They do not diagnose cancer, confirm treatment response, choose chemotherapy, or replace clinician judgment.

## Intended Use

- Summarize longitudinal treatment signals for review.
- Estimate a synthetic treatment-response probability from cycle-level monitoring features.
- Estimate a continuous synthetic MRI response score from cycle-level monitoring features.
- Route simulated clinician-review categories such as routine monitoring, close monitoring, toxicity review, or response-trend review.
- Support MLE/admin evaluation with calibration, threshold, subgroup, drift, and feedback metrics.

## Not Intended For

- Clinical diagnosis.
- Treatment selection.
- Real-world patient triage without clinician oversight.
- Claims of clinical safety, efficacy, or regulatory readiness.

## Current Model Families

### Deterministic Monitoring Rules

- CBC range validation and trend checks.
- Symptom severity and urgent-wording checks.
- Treatment-decision, diagnosis/outcome, prompt-injection, and privacy-boundary routing.

These rules run before LLM adjudication and are part of the safety architecture, not a replacement for clinician review.

### Classical Tabular Models

- Logistic regression
- Random forest
- Extra trees
- Gradient boosting
- SVM with RBF kernel
- MLP classifier

### Temporal Neural Baselines

- Baseline temporal CNN over patient treatment-cycle sequences.
- Temporal 1D CNN over patient treatment-cycle sequences.
- Temporal GRU over patient treatment-cycle sequences.

These are sequence baselines, not clinically validated temporal foundation models.

### Continuous Response Regressors

- Ridge regression
- Random forest regressor
- Extra trees regressor
- Gradient boosting regressor
- Gradient boosting regressor with Huber loss
- Huber regressor
- SVR with RBF kernel
- Robust median response ensemble

These estimate `response_score_percent`, a synthetic continuous MRI response signal. Positive values represent percent tumor-size reduction from baseline in the simulator; negative values represent growth/progression signal in the simulator. This is a model-engineering target, not a clinically validated response measurement.

### Imaging / CNN Direction

Small CNN and imaging-preprocessing experiments are baseline demonstrations only. They should be described as raw-imaging workflow exploration or proof of deep-learning implementation, not as validated breast MRI interpretation.

### LLM Behavior / QLoRA Direction

QLoRA is a planned local learning experiment for behavior, not medical knowledge. The safe intended target is:

- structured JSON validity
- non-diagnostic wording
- refusal and escalation behavior
- patient-friendly explanations
- citation-aware formatting
- insufficient-evidence responses

RAG remains necessary for factual grounding and current knowledge. A safe future claim would be:

> QLoRA was used experimentally to improve structured summary behavior and safety-boundary adherence. RAG remained necessary for factual grounding.

## Input Representation

The synthetic longitudinal response models use one row per treatment cycle with:

- CBC values: WBC, ANC, hemoglobin, platelets, nadirs, and recovery values.
- Treatment context: cycle number, regimen, dose delay, dose reduction.
- MRI-derived tabular features: tumor size and percent change from baseline.
- Symptom/intervention context: max symptom severity, symptom count, intervention count.
- Demographics and diagnosis context: age, cancer stage, molecular subtype.

The current longitudinal model does not consume raw DICOM or NIfTI voxels. Raw MRI computer vision is a planned integration path. Current monitoring models use MRI-derived numeric trend features.

## Training Objective

Main target:

- `treatment_success_binary`

This is a synthetic end-of-journey outcome label generated by the simulator. The model learns patterns in the synthetic patient journey generator, not real clinical outcomes.

Additional cycle-level tasks:

- `toxicity_risk_binary`
- `support_intervention_needed`
- `urgent_intervention_needed`

These are simulator-learning tasks based on generated CBC, symptom, and intervention rules.

Continuous exploratory target:

- `response_score_percent`

This target is derived from MRI percent change from baseline. It is useful for regression experiments, calibration/error analysis, and patient-level response-trajectory modeling, but it must not be described as confirmed treatment response.

Hybrid MLE signal:

- Combines the best patient-level classifier probability with the best response-regression score.
- Current formula: 65% classification probability score + 35% normalized regression response score.
- Reports agreement between classifier and regressor bands (`aligned`, `partially_aligned`, `conflicting`, or `single_signal_available`).
- Used as an exploratory monitoring signal for dashboards and summaries, not as clinical treatment-response confirmation.
- Current selected response regressor: robust median response ensemble, selected with an outlier-aware score: patient-level MAE + 0.15 * patient-level RMSE.

Calibrated probability head:

- The current champion exports an isotonic-regression calibrated probability column.
- Calibration is fitted on a synthetic holdout calibration split and evaluated on its validation half.
- This improves engineering probability behavior, but it is still synthetic-data evidence and not clinical validation.

## Current Evaluation

The admin/MLE dashboard reports:

- AUROC and AUPRC.
- Sensitivity, specificity, precision, Brier score.
- Expected calibration error.
- Bootstrap confidence intervals.
- False-negative review cases.
- Regression MAE, RMSE, and R2 for the synthetic response score.
- Decision-curve net benefit.
- Threshold operating points.
- Cost-sensitive threshold policies.
- Subgroup performance by stage, subtype, age band, and regimen.
- Drift and missingness checks.
- Clinician-loop review and LLM-summary quality proxies.
- Temporal leakage audit status.
- Dataset lineage hashes, schema signatures, generation seed, and feature lineage.
- Locked holdout manifest and patient split hash.
- Error taxonomy: delayed toxicity detection, subtype confusion, sparse-history instability, regimen-shift uncertainty, false-negative favorable response, false-positive overoptimism, and response-regression outliers.

Metric statuses such as `passed`, `strong`, or `acceptable` are project engineering gates only. They are not clinical validation.

### Read this first — honest framing

The headline numbers below come from a **synthetic** dataset (600 simulated patients, 3,600 treatment-cycle rows). The synthetic generator is part of the project — a model that fits its own simulator perfectly is not a clinical claim. The real-data direction lands at **AUROC 0.637**, which is the honest signal for how well this work currently generalises.

External real-data direction (the number that matters for clinical readability):

- BreastDCEDL/I-SPY1 MRI-derived tabular baseline: 159 rows.
- Best exploratory real-data baseline: logistic regression.
- **AUROC: 0.637.**
- AUPRC: 0.388.
- Balanced accuracy: 0.630.
- Small CNN baseline validation AUROC: 0.420.
- Interpretation: weak external signal, useful as an honest real-data direction, **not clinical validation**.

Locked synthetic holdout (engineering monitoring evidence — synthetic only):

- Dataset: 600 synthetic patients, 3,600 treatment-cycle rows.
- Training discipline: train/calibrate on 480-patient development split, then evaluate once on a frozen 120-patient locked holdout.
- Champion classifier: gradient boosting.
- Locked holdout calibrated AUROC: 0.963.
- Locked holdout calibrated AUPRC: 0.966.
- Locked holdout calibrated Brier score: 0.048.
- Locked holdout calibrated confusion matrix: TN 52, FP 2, FN 4, TP 62.
- Best continuous response regressor: random forest regressor for this locked workflow.
- Locked holdout response MAE: 1.242 percentage points.
- Locked holdout response RMSE: 5.226 percentage points.
- Locked holdout response R2: 0.984.
- Response uncertainty on locked holdout: 113 narrow, 5 moderate, 2 wide; mean 10th-to-90th percentile width 2.039.
- MLE readiness: acceptable for supervised PoC demo with limitations; hard gates passed.

Internal development-test reference (do not use this as a portfolio headline — it is essentially the model fitting the simulator):

- Internal development-test patient-level AUROC: 0.995.
- Internal development-test AUPRC: 0.996.
- Internal development-test Brier score: 0.047.
- Internal development-test calibrated validation ECE: 0.0597.

## Current Known Limitations

- Training labels are synthetic.
- No real-world clinical validation has been performed.
- Synthetic data does not fully capture real hospital noise, scanner variation, missingness, clinical practice variation, or bias.
- Subgroup metrics are synthetic simulator checks. They can reveal generator/model brittleness, but they do not prove subgroup safety in real populations.
- Cost-sensitive thresholds are engineering policy simulations. They should be interpreted as "which synthetic threshold is safer under assumed FN/FP costs," not as clinical operating points.
- MRI integration is currently MRI-derived feature integration, not full raw-image clinical interpretation.
- QLoRA/local fine-tuning, when added, should be treated as behavior-formatting research only.
- The model has no authority to recommend treatment changes.
- Clinician feedback metrics are workflow proxies, not ground truth.

## Human Oversight

The system is designed around clinician-in-the-loop review:

- AI summaries can be approved, edited, rejected, or marked for follow-up.
- Reviews are logged for audit and future evaluation.
- Patient-facing text uses monitoring/support language instead of diagnosis or treatment instructions.

## Safe Project Positioning

Best one-line description:

> AI-assisted breast cancer treatment monitoring and clinician-review support platform using synthetic longitudinal oncology journeys, MRI-derived trend features, deterministic safety rules, explainable summaries, and MLE/admin evaluation.
