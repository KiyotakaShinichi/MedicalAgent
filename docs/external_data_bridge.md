# External Data Bridge

OncoTrack now has a small external-data bridge for mapping public breast-cancer
benchmark rows into a canonical oncology schema. This is an engineering
interoperability layer, not clinical validation.

## What It Builds

Run:

```bash
python scripts/run_external_data_bridge.py
python scripts/run_treatment_sequence_feature_eval.py
python scripts/run_cbioportal_clinical_export.py
python scripts/run_external_distribution_alignment.py
python scripts/run_common_feature_transfer_stress.py
python scripts/run_public_distribution_realism_candidate.py
python scripts/run_realism_candidate_ab_gate.py
python scripts/run_dataset_expansion_deep_search.py
```

Artifacts:

- `Data/external_bridge/canonical_oncology_schema.json`
- `Data/external_bridge/canonical_breastdcedl_spy1.csv`
- `Data/external_bridge/cbioportal/canonical_cbioportal_breast_rows.csv`
- `Data/external_bridge/synthetic_treatment_sequences.csv`
- `Data/external_bridge/realism_candidate/temporal_ml_rows_public_realism_candidate.csv`
- `Data/evals/models/latest_external_data_bridge_eval.json`
- `Data/evals/models/latest_cbioportal_clinical_export.json`
- `Data/evals/models/latest_external_distribution_alignment.json`
- `Data/evals/models/latest_common_feature_transfer_stress.json`
- `Data/evals/models/latest_public_distribution_realism_candidate.json`
- `Data/evals/models/latest_realism_candidate_ab_gate.json`
- `Data/evals/models/latest_dataset_expansion_deep_search.json`
- `Data/evals/models/latest_external_failure_case_gallery.json`
- `Data/evals/models/latest_treatment_sequence_feature_eval.json`

## Current External Bridge

The local BreastDCEDL/I-SPY1 snapshot is mapped as an external pCR/imaging
benchmark. It contributes age, molecular subtype, MRI-derived imaging features,
and pCR label context.

cBioPortal TCGA-BRCA/METABRIC rows are exported into the same canonical schema
for public demographic, receptor/subtype, mutation-count, survival/recurrence,
and treatment-context distribution checks. They are not longitudinal OncoTrack
monitoring rows and should not be used as response-score or toxicity labels.

It does **not** provide the full OncoTrack patient journey:

- no CBC timeline
- no symptom timeline
- no medication-by-cycle history
- no radiation/surgery/endocrine sequence details in the local bridge
- no tumor-marker timeline
- no clinician-reviewed OncoTrack outcome labels

## Common-Feature Transfer Stress

The strict common-feature transfer stress test uses only:

- `age`
- tumor-size proxy (`baseline_tumor_size_mm`)
- `hr_positive`
- `her2_positive`
- `triple_negative`

It trains within-source sanity models and then scores cross-source rows to
expose distribution shift and brittle transfer behavior. Cross-source metrics
are labeled as mismatched-endpoint stress signals, not validation, because
synthetic treatment success, BreastDCEDL pCR, and cBioPortal survival/recurrence
are different endpoints.

## Public-Distribution Realism Candidate

`scripts/run_public_distribution_realism_candidate.py` writes a separate
synthetic candidate CSV that shifts selected age and tumor-size proxy
distributions toward public cohort summaries. This is useful for A/B testing
generator realism under controlled gates. It is **not** the production synthetic
dataset and should not be promoted without leakage, shortcut, calibration,
counterfactual-stability, and release-gate review.

`scripts/run_realism_candidate_ab_gate.py` performs that first controlled A/B
check. The current policy is conservative: keep the current generator as the
default and use the public-distribution candidate only for A/B experiments until
exact-label external temporal validation and clinician review are available.

## Dataset Expansion Deep Search

`scripts/run_dataset_expansion_deep_search.py` produces a governed catalog of
next data sources. The current highest-priority bridges are:

- AACR GENIE BPC Breast Cancer for treatment-history and clinico-genomic context.
- Duke Breast Cancer MRI for imaging, receptor/pathology, treatment, recurrence,
  and follow-up context.

The catalog also tracks BreastDCEDL, I-SPY2, QIN-BREAST, TCGA-BRCA, CPTAC,
SEER, MIMIC-IV, and EDRN as source-specific realism or context aids.

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
