# OncoTrack Data Dictionary

This dictionary explains the field groups used by the synthetic timeline,
clinical ontology, and public-data bridge. It is an engineering contract, not a
clinical terminology authority.

## Claim Boundary

The fields below organize records and benchmark artifacts. They do not diagnose,
confirm treatment response, interpret genetic risk, prove recurrence, or support
treatment recommendations.

## Core Synthetic Timeline Fields

| Field | Meaning | Allowed use | Must not claim |
|---|---|---|---|
| `patient_id` | Synthetic patient identifier. | Patient-level grouping and leakage checks. | Real patient identity. |
| `cycle` | Treatment-cycle index. | Temporal ordering and prior-only prediction checks. | Clinical treatment schedule correctness. |
| `treatment_date` | Synthetic cycle date. | Timeline ordering and leakage audits. | Real appointment date. |
| `age` | Synthetic age. | Demographic/context feature and subgroup checks. | Personalized clinical risk. |
| `stage` | Synthetic stage text. | Context and subgroup analysis. | Diagnosis/staging confirmation. |
| `molecular_subtype` | Synthetic receptor/subtype bucket. | Feature grouping and public-distribution checks. | Treatment selection advice. |
| `regimen` | Synthetic regimen text. | Treatment-context feature organization. | Recommended regimen. |

## CBC / Lab Fields

| Field | Meaning | Allowed use | Review boundary |
|---|---|---|---|
| `pre_wbc`, `nadir_wbc`, `recovery_wbc` | White blood cell trend around a cycle. | Monitoring signal and toxicity-review hint. | Does not diagnose infection or neutropenia alone. |
| `pre_anc`, `nadir_anc` | Absolute neutrophil count trend. | Urgent-review routing when paired with fever/red flags. | Does not replace clinician/lab interpretation. |
| `pre_hemoglobin`, `nadir_hemoglobin`, `recovery_hemoglobin` | Hemoglobin trend. | Anemia-review context. | No transfusion or medication advice. |
| `pre_platelets`, `nadir_platelets`, `recovery_platelets` | Platelet trend. | Bleeding-risk review context. | No bleeding diagnosis or treatment advice. |

## Imaging / Response Proxy Fields

| Field | Meaning | Allowed use | Must not claim |
|---|---|---|---|
| `mri_tumor_size_cm` | Synthetic MRI tumor-size proxy in centimeters. | Response-score regression input and distribution checks. | Raw-image diagnosis or confirmed response. |
| `mri_percent_change_from_baseline` | Synthetic percent change from baseline size. | Imaging-supported monitoring signal. | Clinical response confirmation. |
| `response_score_percent` | Synthetic response-score label. | Regression target for engineering practice. | Real treatment effectiveness. |
| `final_response_category` | Simulator outcome category. | Synthetic evaluation label. | Clinician-adjudicated response. |

## Symptom / Intervention Fields

| Field | Meaning | Allowed use | Review boundary |
|---|---|---|---|
| `max_symptom_severity` | Highest patient-reported severity for the cycle. | Review-priority feature. | Not a CTCAE grade diagnosis. |
| `symptom_count` | Number of reported symptoms. | Timeline burden/context. | Not severity by itself. |
| `intervention_count` | Synthetic count of support interventions. | Support-signal feature. | Not a care recommendation. |
| `dose_delayed`, `dose_reduced` | Synthetic treatment adjustment flags. | Historical context and leakage checks. | Do not suggest dose changes. |

## Synthetic Labels

| Label | Meaning | Use | Boundary |
|---|---|---|---|
| `treatment_success_binary` | Simulator-generated favorable/not-favorable outcome. | Classification target. | Not clinical treatment success. |
| `toxicity_risk_binary` | Legacy simulator toxicity flag. | Shortcut audit target. | Shortcut-prone, not a toxicity diagnosis. |
| `urgent_intervention_needed` | Simulator urgent-review signal. | Safety/evaluation target. | Not emergency diagnosis. |
| `support_intervention_needed` | Simulator support-review signal. | Workflow target. | Not treatment advice. |
| `maintenance_needed` | Simulator maintenance flag. | Synthetic outcome context. | Not a maintenance recommendation. |

## Strict Common Public-Bridge Features

These fields are the only common feature set used by the transfer-stress
artifact:

| Field | Synthetic source | Public bridge source | Boundary |
|---|---|---|---|
| `age` | Synthetic patient age. | BreastDCEDL/cBioPortal age attributes. | Demographic alignment only. |
| `baseline_tumor_size_mm` | Tumor-size proxy derived from synthetic MRI field. | BreastDCEDL baseline longest diameter or cBioPortal imaging metadata when present. | Proxy field; source semantics differ. |
| `hr_positive` | Derived from synthetic subtype. | ER/PR/subtype status. | Context feature, not treatment advice. |
| `her2_positive` | Derived from synthetic subtype. | HER2/subtype status. | Context feature, not treatment advice. |
| `triple_negative` | Derived from synthetic subtype. | Subtype or receptor negativity. | Context feature, not prognosis. |

## Public Bridge Labels

| Source | Label field | Meaning | Why it is not interchangeable |
|---|---|---|---|
| Synthetic timeline | `treatment_success_binary` | Simulator-defined outcome. | Generated by project rules. |
| BreastDCEDL/I-SPY1 | `pcr_label` / `outcome_label_value` where `outcome_label_name = pCR` | Pathologic complete response context. | pCR is a pathology endpoint, not the OncoTrack synthetic response label. |
| cBioPortal TCGA/METABRIC | survival/recurrence attributes | Retrospective clinical/genomic context. | Not longitudinal monitoring, CBC, symptom, or response-score labels. |

## Dataset Expansion Roles

| Dataset family | Fields to map first | Safe role |
|---|---|---|
| GENIE BPC BRCA | systemic therapy intervals, pathology, genomic alterations, outcomes | treatment-history and clinico-genomic readiness; no treatment recommendation |
| Duke Breast MRI | DCE-MRI features, receptor status, treatment fields, pathologic response, recurrence/follow-up | imaging/treatment-context external bridge; no clinical validation claim |
| BreastDCEDL / I-SPY | DCE-MRI, pCR, HR/HER2, age | MRI pCR benchmark; pCR is not OncoTrack treatment-success |
| TCGA-BRCA / METABRIC | mutations, subtype, expression, survival | mutation/subtype distribution context; no patient-specific genetic-risk prediction |
| CPTAC | genomic/proteomic assays | biomarker/proteomic context; not treatment-cycle monitoring |
| SEER | ER/PR/HER2/Ki-67 SSDI, stage, treatment utilization, survival | population distribution and coding discipline |
| MIMIC-IV | labs, medications, EHR structure | lab missingness/unit robustness only |
| EDRN | biospecimen/tumor-marker reference context | tumor-marker limitation education, not response prediction |

## Priority Dataset Bridge Contracts

`scripts/run_priority_dataset_bridge.py` writes executable templates for the two
highest-priority next sources:

| Artifact | Purpose | Status without local export |
|---|---|---|
| `Data/external_bridge/priority_dataset_templates/genie_bpc_brca_field_contract.csv` | Expected GENIE BPC aliases for patient ID, age, stage, ER/PR/HER2, genomic alteration context, regimen/treatment history, and real-world outcome fields. | `ready_for_mapping` |
| `Data/external_bridge/priority_dataset_templates/duke_breast_mri_field_contract.csv` | Expected Duke MRI aliases for patient ID, age, receptor status, MRI feature columns, treatment context, pCR, recurrence, and follow-up fields. | `ready_for_mapping` |

If real permitted CSV exports are later supplied, the same bridge maps them to
`canonical_genie_bpc_brca.csv` and `canonical_duke_breast_mri.csv`. These rows
remain external stress/schema rows, not clinical validation rows.

## Mutation Context Fields

Mutation and gene fields are allowed only as context and review-routing inputs.
They are not treatment-response proof and they are not inherited-risk diagnosis.

| Gene group | Example genes | Safe role | Must not claim |
|---|---|---|---|
| Somatic pathway context | PIK3CA, TP53, GATA3, ESR1, ERBB2 | External molecular context and ablation candidate | Treatment is working/failing because of the mutation |
| Germline-sensitive context | BRCA1, BRCA2, PALB2, ATM, CHEK2, PTEN | Genetic-counselor readiness and record organization | Inherited cancer risk, family risk, or treatment recommendation |
| Unknown source type | Any gene where somatic/germline source is unclear | Clinician/genetic-counselor review flag | Somatic alteration equals inherited family risk |

`scripts/run_mutation_context_mapping.py` creates the executable mapping artifact
and keeps `promotion_allowed = false`.

## Clinical Ontology Pointer

The executable clinical ontology lives in
`backend/services/clinical_ontology.py`. It defines allowed values, units,
blocked claims, and review notes for labs, symptoms, imaging, biomarkers,
genetic tests, tumor markers, supplements, and medications.

When adding a field, update this data dictionary, the clinical ontology if it
is patient/clinician-facing, and any release-gate artifact that depends on the
field.
