# Model Card: Synthetic Toxicity Risk Model

## Purpose
Estimate a synthetic toxicity risk flag from longitudinal monitoring features to support clinician review in a PoC workflow.

Evidence: [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/synthetic_journey.py](backend/services/synthetic_journey.py)

## Intended use
- Provide a monitoring signal for clinician review in synthetic workflows.
- Support MLE evaluation of model training, calibration, and registry lifecycle.

## Not intended use
- Clinical diagnosis or treatment decisions.
- Real-world patient triage or therapy changes.

## Inputs
- Cycle-level CBC values, treatment context, symptom summaries, and MRI-derived tabular trends.

## Outputs
- Synthetic toxicity risk probability or label.

## Dataset/source
- Complete synthetic longitudinal breast cancer journeys.

## Synthetic data limitations
- Synthetic data does not represent real clinical variability.
- Results measure simulator learning, not clinical performance.

## Metrics
- Metrics are generated in the synthetic training pipeline and evaluation reports.
- Use as engineering evidence only.
- Temporal generalization report: `Data/mle_monitoring/temporal_eval_report.json`
- High-noise robustness report: `Data/mle_monitoring/noise_eval_report.json`
- Calibration comparison report: `Data/mle_monitoring/calibration_eval_report.json`

## Calibration
- Calibration metrics are available for synthetic runs; clinical calibration is not established.
- The calibration comparison tests raw probability, isotonic regression, Platt scaling, and temperature scaling on synthetic holdout predictions.

## Subgroup behavior
- Subgroup checks are available for synthetic runs; not clinically validated.

## Failure modes
- Simulator bias and simplified patterns.
- Potential mismatch with real-world toxicity distributions.
- Temporal instability if later patient timelines differ from earlier training timelines.
- Data-quality brittleness under missing CBC values, lab jitter, unit-entry errors, site batch effects, and contradictory symptom records.

## Ethical and safety risks
- Risk of over-reliance on synthetic signals.
- Must be presented as a monitoring flag only.

## Clinician review requirement
All outputs require clinician review and non-diagnostic wording.

## Deployment status
PoC only. Not clinically validated or approved for clinical use.
