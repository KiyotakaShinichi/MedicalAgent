# Patient Portal Workflow Reference

## Metadata

- title: Patient Portal Workflow Reference
- topic: portal_help
- audience: patient and reviewer
- allowed_use: portal_help, education
- source_tier: T4 project documentation
- source_urls:
  - README.md
  - docs/reviewer_evidence.md
- last_reviewed: 2026-05-18

## Patient-safe summary

The NLCare portal is a proof-of-concept workspace for organizing symptoms,
CBC/lab values, medications, treatment notes, imaging report text, uploads,
model-signal context, and questions for clinician review. It is not for
diagnosis or treatment decisions.

## What the assistant may say

- Use the symptom tool to log symptom name, severity, date, and notes.
- Use the CBC/lab tool to enter WBC, hemoglobin, platelets, and related report
  values when available.
- Use the imaging tool to upload or paste MRI, CT, ultrasound, mammogram, or
  other report text for clinician review.
- Explain that the model signal is an exploratory synthetic-only engineering
  output, not a clinical prediction.
- Explain that "PoC - not for clinical use" means the system is a demo and
  should not guide care.

## What the assistant must not say

- "The portal replaces your care team."
- "The model signal proves your treatment is working."
- "Upload another person's identifiable records without consent."
- "The system is clinically validated."

## Escalation and review boundary

Portal help can explain where to enter data and what fields mean. Medical
interpretation, urgent symptoms, genetic results, biomarker/tumor-marker
meaning, and treatment decisions must route to the care team.
