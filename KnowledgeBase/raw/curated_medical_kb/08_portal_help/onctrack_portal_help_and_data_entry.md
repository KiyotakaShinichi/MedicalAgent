# OncoTrack Portal Help and Data Entry

## Purpose

This document helps the support agent answer questions about how to use OncoTrack without drifting into medical advice.

## What patients can log

Patients can log symptoms, CBC/lab values, medication mentions, and imaging report text. The support chat can save complete entries when enough structured information is present. The portal also shows trends, timeline events, AI summaries, and safety notes for clinician review.

## Logging symptoms

A symptom entry needs a symptom name and severity from 0 to 10. A date is helpful. If the patient says "nausea severity 6/10 today," the assistant can save it. If severity is missing, the assistant should ask for the severity before claiming it was saved.

## Saving CBC values

A complete CBC entry needs WBC, hemoglobin, and platelets. A date is helpful. The assistant can save complete values and deterministic rules can flag concerning values for clinician review. The assistant must not decide treatment from CBC values.

## Saving imaging reports

Patients may paste MRI, CT, ultrasound, PET/CT, or other imaging report text. The assistant can save report wording and flag concerning terms for clinician review. It must not diagnose metastasis or response.

## Saving medication mentions

Patients may mention medications or supplements they are taking. The assistant can log the mention, but must not prescribe, change dose, or decide whether a medication or supplement is safe.

## What "model signal" means

The model signal is an exploratory engineering signal in this proof of concept. It is not a diagnosis, not a treatment recommendation, and not clinical validation. It helps organize review, but the care team makes decisions.

## What "PoC not for clinical use" means

OncoTrack is a proof-of-concept monitoring support system. Demo data may be synthetic or public-data-derived. It is not certified for clinical deployment and should not be used for actual medical decisions.

## Sources

- Project system card: docs/system_card.md
- Project model card: MODEL_CARD.md
- OncoTrack README: README.md

Last reviewed: 2026-05-15
