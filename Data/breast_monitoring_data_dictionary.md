# Breast Cancer Monitoring Data Dictionary

This project uses a canonical longitudinal format so public datasets and hospital exports can be mapped into the same patient journey model.

## patients

| Field | Required | Description | Example |
| --- | --- | --- | --- |
| patient_id | yes | Stable de-identified patient identifier | P001 |
| name | no | Display name or de-identified label | Patient P001 |
| diagnosis | no | Doctor-confirmed diagnosis context | Breast cancer - doctor-confirmed |

## breast_profiles

| Field | Required | Description | Example |
| --- | --- | --- | --- |
| patient_id | yes | Links profile to patient | P001 |
| cancer_stage | no | Clinical stage if known | Stage II |
| er_status | no | Estrogen receptor status | positive |
| pr_status | no | Progesterone receptor status | positive |
| her2_status | no | HER2 status | negative |
| molecular_subtype | no | Clinical subtype | HR-positive / HER2-negative |
| treatment_intent | no | Monitoring context | neoadjuvant chemotherapy monitoring |
| menopausal_status | no | Menopausal status at diagnosis | premenopausal |

## labs

| Field | Required | Description | Example |
| --- | --- | --- | --- |
| patient_id | yes | Links lab to patient | P001 |
| date | yes | Lab date, ISO format preferred | 2026-01-15 |
| wbc | yes | White blood cell count | 2.1 |
| hemoglobin | yes | Hemoglobin | 11.8 |
| platelets | yes | Platelet count | 150 |

Default units are WBC and platelets in x10^3/uL, hemoglobin in g/dL.

## treatments

| Field | Required | Description | Example |
| --- | --- | --- | --- |
| patient_id | yes | Links treatment to patient | P001 |
| date | yes | Cycle or treatment date | 2026-01-15 |
| cycle | yes | Treatment cycle number | 2 |
| drug | yes | Regimen or drug name | Doxorubicin/Cyclophosphamide |

## imaging_reports

| Field | Required | Description | Example |
| --- | --- | --- | --- |
| patient_id | yes | Links report to patient | P001 |
| date | yes | Imaging report date | 2026-02-01 |
| modality | yes | Imaging modality | Breast MRI |
| report_type | yes | Baseline, follow-up, staging, etc. | Follow-up breast MRI |
| body_site | no | Body site or exam region | Breast |
| findings | yes | Findings report text | Right breast mass measuring 3.1 cm |
| impression | yes | Impression report text | Interval decrease |

## symptoms

| Field | Required | Description | Example |
| --- | --- | --- | --- |
| patient_id | yes | Links symptom to patient | P001 |
| date | yes | Symptom report date | 2026-01-18 |
| symptom | yes | Symptom name | mouth sores |
| severity | yes | 0-10 patient-reported score | 7 |
| notes | no | Optional context | Painful eating after cycle 2 |

## Dataset Adapter Notes

Duke Breast Cancer MRI and I-SPY2 metadata should map into `patients`, `breast_profiles`, and `imaging_reports`. Raw DICOM/NIfTI imaging is intentionally not imported into this app yet.

MIMIC-style laboratory exports should map into `labs`, but they are usually not linked to the public breast MRI datasets at patient level. Use them to validate CBC trend logic unless you have an approved linked institutional dataset.
