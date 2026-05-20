# FHIR-Aligned Schema Readiness

NLCare includes internal canonical objects inspired by FHIR resource shapes:

- Observation-like records for CBC/labs.
- MedicationStatement-like records for medications.
- DiagnosticReport-like records for imaging summaries.
- FamilyMemberHistory-like records for family cancer history.
- Condition-like records for medical context.

This is not certified FHIR interoperability, not hospital integration, and not connected to a real EHR.

## Design Rules

- Missing or unmapped codes are allowed.
- Code mappings are optional metadata, not clinical authority.
- No medical claim depends solely on a code mapping.
- Mapping artifacts report coverage, missing fields, unit normalization, and validation errors.

## Evaluation

Run:

```bash
python scripts/run_fhir_alignment_readiness.py
```

Artifact:

```text
Data/evals/medical/latest_fhir_alignment_readiness.json
```

Use this as future interoperability preparation only.
