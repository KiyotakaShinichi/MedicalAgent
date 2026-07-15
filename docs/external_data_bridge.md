# External Data Bridge

NLCare now has a small external-data bridge for mapping public breast-cancer
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
python scripts/run_priority_dataset_bridge.py
python scripts/run_priority_external_stress.py
python scripts/run_mutation_context_mapping.py
python scripts/run_dataset_fit_matrix.py
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
- `Data/evals/models/latest_priority_dataset_bridge.json`
- `Data/evals/models/latest_priority_external_stress.json`
- `Data/evals/models/latest_mutation_context_mapping.json`
- `Data/evals/models/latest_dataset_fit_matrix.json`
- `Data/external_bridge/priority_dataset_templates/genie_bpc_brca_field_contract.csv`
- `Data/external_bridge/priority_dataset_templates/duke_breast_mri_field_contract.csv`
- `Data/evals/models/latest_external_failure_case_gallery.json`
- `Data/evals/models/latest_treatment_sequence_feature_eval.json`

## Current External Bridge

The local BreastDCEDL/I-SPY1 snapshot is mapped as an external pCR/imaging
benchmark. It contributes age, molecular subtype, MRI-derived imaging features,
and pCR label context.

cBioPortal TCGA-BRCA/METABRIC rows are exported into the same canonical schema
for public demographic, receptor/subtype, mutation-count, survival/recurrence,
and treatment-context distribution checks. They are not longitudinal NLCare
monitoring rows and should not be used as response-score or toxicity labels.

It does **not** provide the full NLCare patient journey:

- no CBC timeline
- no symptom timeline
- no medication-by-cycle history
- no radiation/surgery/endocrine sequence details in the local bridge
- no tumor-marker timeline
- no clinician-reviewed NLCare outcome labels

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

## Priority Dataset Bridge

`scripts/run_priority_dataset_bridge.py` turns the top two dataset targets into
executable mapping contracts:

- **GENIE BPC BRCA**: treatment-history and clinico-genomic field contract for
  systemic regimens, receptor context, genomic alterations, and real-world
  outcome semantics.
- **Duke Breast MRI**: imaging/treatment-context field contract for MRI-derived
  features, receptor/pathology context, pCR/recurrence/follow-up endpoints, and
  treatment-context fields.

If no local CSV export is supplied, the artifact intentionally reports
`ready_for_mapping`. That is a good status: it means the project has templates,
expected aliases, claim boundaries, and release-gate coverage without pretending
that restricted or large external data has already been integrated.

`scripts/run_priority_external_stress.py` is the next control layer. It checks
mapped priority rows for common-feature coverage and endpoint compatibility, but
keeps `promotion_allowed = false` unless exact-label temporal validation exists.

`scripts/run_mutation_context_mapping.py` adds mutation-context readiness for
genes such as PIK3CA, TP53, GATA3, ESR1, ERBB2, BRCA1, BRCA2, PALB2, ATM, CHEK2,
and PTEN. These are context and review-routing signals only, not genetic-risk or
treatment-response predictions.

`scripts/run_dataset_fit_matrix.py` scores candidate sources by treatment,
temporal, imaging, biomarker, genomic, tumor-marker, lab, and student-access fit
so the data roadmap remains evidence-driven instead of hype-driven.

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

This bridge does not prove that NLCare predicts real treatment response. It
only proves that the codebase can:

- define a canonical oncology schema
- map a public pCR/imaging benchmark into that schema
- preserve source and claim boundaries
- compare synthetic candidates against a public external sanity-check artifact
- document failure cases honestly

Future real-world validation would require clinician-reviewed labels, real
longitudinal treatment/CBC/symptom/imaging timelines, governed data access, and
prospective or carefully designed retrospective validation.
