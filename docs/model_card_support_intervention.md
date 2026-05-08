# Model Card: Synthetic Support-Intervention Flag Model

## Purpose
Estimate a synthetic support-intervention flag from longitudinal monitoring features to highlight cases that may need clinician follow-up.

Evidence: [backend/services/complete_synthetic_training.py](backend/services/complete_synthetic_training.py), [backend/services/synthetic_journey.py](backend/services/synthetic_journey.py)

## Intended use
- Provide a monitoring signal for clinician review in synthetic workflows.
- Support MLE evaluation and lifecycle practice.

## Not intended use
- Clinical diagnosis, triage, or treatment recommendations.
- Use in real-world care without validation.

## Inputs
- Cycle-level CBC values, treatment context, symptom summaries, and intervention markers.

## Outputs
- Synthetic support-intervention probability or label.

## Dataset/source
- Complete synthetic longitudinal breast cancer journeys.

## Synthetic data limitations
- Synthetic data does not represent real clinical variability.
- Results measure simulator learning, not clinical performance.

## Metrics
- Metrics are generated in the synthetic training pipeline and evaluation reports.
- Use as engineering evidence only.

## Calibration
- Calibration metrics are available for synthetic runs; clinical calibration is not established.

## Subgroup behavior
- Subgroup checks are available for synthetic runs; not clinically validated.

## Failure modes
- Simulator bias and simplified patterns.
- Potential mismatch with real-world intervention rates.

## Ethical and safety risks
- Risk of over-reliance on synthetic signals.
- Must be presented as a monitoring flag only.

## Clinician review requirement
All outputs require clinician review and non-diagnostic wording.

## Deployment status
PoC only. Not clinically validated or approved for clinical use.
