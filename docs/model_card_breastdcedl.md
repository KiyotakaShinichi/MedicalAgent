# Model Card: BreastDCEDL Baseline Response Classifier

## Purpose
Provide a baseline treatment-response classification signal using MRI-derived tabular features from BreastDCEDL/I-SPY1. This is an exploratory PoC signal for clinician review, not a clinical decision tool.

Evidence: [backend/services/breastdcedl_baseline.py](backend/services/breastdcedl_baseline.py), [backend/services/model_artifacts.py](backend/services/model_artifacts.py), [backend/services/inference_service.py](backend/services/inference_service.py)

## Intended use
- Produce a monitoring signal for clinician review in a breast cancer treatment journey PoC.
- Support engineering evaluation and model lifecycle practice.

## Not intended use
- Clinical diagnosis or confirmation of treatment response.
- Treatment recommendations or medication changes.
- Use in real-world clinical care without validation.

## Inputs
- MRI-derived tabular features (tumor size, enhancement metrics, molecular subtype).

## Outputs
- `pcr_probability` and a demo classification label.
- Monitoring signal with non-diagnostic safety note.

## Dataset/source
- BreastDCEDL/I-SPY1 MRI dataset with derived tabular features.

## Synthetic data limitations
- This model is trained on a real dataset, but the broader system uses synthetic data for workflow evidence.
- Do not generalize from this PoC baseline to clinical claims.

## Metrics
- ROC AUC and basic accuracy metrics from cross-validated baselines.
- Metrics are exploratory and not clinically validated.

## Calibration
- Not fully characterized for this baseline. Calibration work is planned.

## Subgroup behavior
- Not fully characterized for this baseline. Subgroup analysis is planned.

## Failure modes
- Small sample sizes and dataset bias.
- Derived features may not capture full clinical complexity.

## Ethical and safety risks
- Risk of false reassurance or false alarm if treated as clinical truth.
- Requires clinician oversight and explicit non-diagnostic framing.

## Clinician review requirement
All outputs are monitoring signals for clinician review and must not be used as diagnoses.

## Deployment status
PoC baseline only. Not clinically validated.
