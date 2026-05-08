# Synthetic Data

MedicalAgent relies on synthetic longitudinal breast cancer journeys for training, evaluation, and workflow demonstration.

## What the synthetic dataset contains
- CBC values across treatment cycles
- Symptoms, medications, and interventions
- Imaging summary text and MRI-derived tabular signals
- Final synthetic outcomes and cycle-level monitoring labels

Evidence: [backend/services/synthetic_journey.py](backend/services/synthetic_journey.py), [DATA_CARD.md](DATA_CARD.md)

## Why synthetic data is used
- Safe POC workflow testing without real PHI
- Repeatable engineering evidence for MLE readiness gates
- Scenario coverage for guardrails and RAG behavior

## Limitations
- Synthetic data is not clinical evidence and does not prove real-world performance.
- Simulator patterns may be simpler than real clinical variability.
- The project does not claim clinical validation.
