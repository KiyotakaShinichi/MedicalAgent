# Priority Dataset Bridge

Generated at: 2026-05-19T02:54:43.236760+00:00

Status: **ready_for_mapping**

Priority dataset bridge artifacts map external dataset fields into the OncoTrack canonical schema for interoperability, stress testing, and future access readiness only. They do not establish clinical validation, treatment superiority, survival prediction, genetic-risk interpretation, or patient-facing treatment recommendations.

## Datasets

| Dataset | Status | Rows mapped | Best role | Source |
|---|---:|---:|---|---|
| [AACR GENIE BPC Breast Cancer v1.0-public](https://www.aacr.org/professionals/research/aacr-project-genie/bpc/early-onset-brca/) | ready_for_mapping | 0 | treatment_history_bridge, genomic_context_bridge | public/access-controlled workflow; local CSV not bundled |
| [Duke Breast Cancer MRI / TCIA](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/) | ready_for_mapping | 0 | treatment_history_bridge, mri_image_bridge | public TCIA collection; large image/metadata download required |

## Next Actions
- Download/export permitted GENIE BPC BRCA tables, then run this bridge with --genie-csv.
- Download Duke Breast MRI clinical-and-other-features metadata, then run this bridge with --duke-csv.
- Use mapped rows for external stress tests and schema coverage only; keep production models monitor-only.
- Do not mix outcome labels across datasets unless endpoint semantics are explicitly documented.

## Blocked Claims
- real-world clinical validation
- treatment recommendation or treatment superiority
- genetic-risk diagnosis or inherited-risk prediction
- tumor-marker recurrence conclusion
- survival or prognosis estimate
