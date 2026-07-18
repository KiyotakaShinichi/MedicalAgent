# Patient XAI Readability Dossier

Status: strong

No clinical validation. No clinical authority. No patient benefit claim.

## Claim Boundary

Patient-facing XAI explains how NLCare produced synthetic engineering indicators. It does not explain clinical causality, diagnose, recommend treatment, estimate prognosis, interpret genetic risk, interpret tumor markers, prove patient benefit, or validate real-world safety.

## Required Patient Explanation Surfaces

- **Items for review**: what_count_means, how_count_is_calculated, why_not_urgency_or_diagnosis, safe_next_steps
- **Synthetic model pattern**: synthetic_class_probability, confidence_is_not_patient_outcome_probability, modalities_used_and_missing, abstention_or_low_confidence_reason
- **Latest CBC**: population_default_reference_bands, not_personalized_reference_ranges, not_diagnosis, care_team_discussion_boundary
- **Record coverage**: available_vs_missing_record_areas, why_missingness_changes_confidence, not_health_status, what_user_can_update
- **Removed 0-100 headline score**: why_removed_from_patient_headline, why_not_cancer_status, why_not_treatment_response, why_not_prognosis

## Weakness Visibility

- RAG citation precision: 0.4649
- RAG unsupported context rate: 0.1622
- ML attention items: shortcut_risk_boundaries
- Automation emergency coverage claim: False

## UI Copy Rules

- Every patient-visible number needs a meaning, calculation, safe next steps, and nonclinical boundary.
- Use 'synthetic class probability' or 'model confidence', never 'chance of response' or 'survival'.
- When data is missing, say which record types are missing instead of inventing confidence.
- When automation triggers, say 'queued for review workflow' rather than 'doctor notified' unless a verified human acknowledgement exists.
- Keep negative RAG and ML weaknesses visible in admin/reviewer surfaces.
