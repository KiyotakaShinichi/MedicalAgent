# External Data Bridge

OncoTrack now has a small external-data bridge for mapping public breast-cancer
benchmark rows into a canonical oncology schema. This is an engineering
interoperability layer, not clinical validation.

## What It Builds

Run:

```bash
python scripts/run_external_data_bridge.py
python scripts/run_treatment_sequence_feature_eval.py
```

Artifacts:

- `Data/external_bridge/canonical_oncology_schema.json`
- `Data/external_bridge/canonical_breastdcedl_spy1.csv`
- `Data/external_bridge/synthetic_treatment_sequences.csv`
- `Data/evals/models/latest_external_data_bridge_eval.json`
- `Data/evals/models/latest_external_failure_case_gallery.json`
- `Data/evals/models/latest_treatment_sequence_feature_eval.json`

## Current External Bridge

The local BreastDCEDL/I-SPY1 snapshot is mapped as an external pCR/imaging
benchmark. It contributes age, molecular subtype, MRI-derived imaging features,
and pCR label context.

It does **not** provide the full OncoTrack patient journey:

- no CBC timeline
- no symptom timeline
- no medication-by-cycle history
- no radiation/surgery/endocrine sequence details in the local bridge
- no tumor-marker timeline
- no clinician-reviewed OncoTrack outcome labels

## Treatment Sequence Artifact

The treatment-sequence artifact derives synthetic treatment-context features from
the existing simulated rows:

- chemotherapy
- HER2-targeted context
- endocrine context
- planned surgery context
- planned radiation context
- supportive-care context

These are timeline organization features only. They are not treatment
recommendations and do not compare real treatment efficacy.

## Failure-Case Gallery

The external failure-case gallery collects BreastDCEDL baseline false positives
and false negatives for review. This is useful because it keeps weak external
behavior visible instead of hiding it behind a single average metric.

## What This Does Not Prove

This bridge does not prove that OncoTrack predicts real treatment response. It
only proves that the codebase can:

- define a canonical oncology schema
- map a public pCR/imaging benchmark into that schema
- preserve source and claim boundaries
- compare synthetic candidates against a public external sanity-check artifact
- document failure cases honestly

Future real-world validation would require clinician-reviewed labels, real
longitudinal treatment/CBC/symptom/imaging timelines, governed data access, and
prospective or carefully designed retrospective validation.
