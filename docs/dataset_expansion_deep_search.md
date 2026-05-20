# Dataset Expansion Deep Search

Dataset expansion deep search is a planning and governance artifact. It identifies public or controlled-access sources that can improve OncoTrack's realism, schema coverage, or external-readiness checks. It does not mean the data has been downloaded, licensed, mapped, clinically validated, or approved for patient-facing prediction.

## Highest Priority Sources

- **AACR GENIE BPC Breast Cancer v1.0-public** - treatment-history + genomic context external-readiness
  - Source: https://www.aacr.org/professionals/research/aacr-project-genie/bpc/early-onset-brca/
  - Next action: Run `python scripts/run_priority_dataset_bridge.py --genie-csv <permitted_export.csv>` after access/export; do not train patient-facing treatment recommendations.

- **Duke Breast Cancer MRI / TCIA** - MRI + pathology + treatment/outcome/radiogenomic external bridge
  - Source: https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/
  - Next action: Run `python scripts/run_priority_dataset_bridge.py --duke-csv <metadata.csv>` after download/export; use as imaging/treatment-context external stress, not clinical validation.

## Full Candidate Catalog

| Dataset | Best use | Fit | Access | Next action |
|---|---|---|---|---|
| [AACR GENIE BPC Breast Cancer v1.0-public](https://www.aacr.org/professionals/research/aacr-project-genie/bpc/early-onset-brca/) | treatment-history + genomic context external-readiness | highest_priority | public with AACR/GENIE data-use terms | Priority bridge template exists; run with `--genie-csv` after permitted export. |
| [Duke Breast Cancer MRI / TCIA](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/) | MRI + pathology + treatment/outcome/radiogenomic external bridge | highest_priority | public TCIA collection, license-sensitive | Priority bridge template exists; run with `--duke-csv` after metadata export. |
| [BreastDCEDL](https://zenodo.org/records/18114231) | deep-learning-ready MRI pCR benchmark | already_integrated_expand | public Zenodo, CC BY-NC 4.0 derivative license | Expand from current tabular bridge to a small image-feature smoke benchmark if local storage allows. |
| [I-SPY2 / TCIA](https://www.cancerimagingarchive.net/collection/ispy2/) | serial MRI response/pCR temporal imaging benchmark | high_priority | public TCIA, large download | Keep as future temporal imaging benchmark; integrate only metadata first. |
| [QIN-BREAST / TCIA](https://www.cancerimagingarchive.net/collection/qin-breast/) | longitudinal PET/CT + quantitative MRI workflow exploration | medium_high_priority | public TCIA | Map as imaging-workflow readiness; do not treat as response-label validation until labels are audited. |
| [TCGA-BRCA / NCI GDC](https://gdc.cancer.gov/about-data/publications/brca_2012) | somatic mutation, expression, subtype, and molecular-distribution priors | high_priority_context | mixed open/controlled GDC data | Add mutation-frequency/context mapping for PIK3CA, TP53, GATA3, ESR1 when available. |
| [CPTAC Breast Cancer](https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac) | proteogenomic and assay-rich biomarker context | medium_priority_context | GDC/PDC public and controlled data depending on file | Track as future biomarker/proteomics context source after simpler public bridges are stable. |
| [SEER Research Plus / SEER SSDI breast biomarkers](https://seer.cancer.gov/) | population-level demographics, stage, subtype, treatment, survival distribution priors | medium_priority_distribution | research data request / SEER*Stat terms | Prepare SEER field dictionary mapping; use for distribution sanity checks only. |
| [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) | lab missingness/unit realism and EHR pipeline practice | supporting_lab_realism | credentialed PhysioNet access | Use only for lab-distribution/unit robustness after credentialed access; keep oncology labels separate. |
| [NCI EDRN Breast Cancer Reference Set](https://edrn.nci.nih.gov/documents/34/breast_refset_summary.pdf) | tumor-marker assay limitation/context source | context_only | reference-set/biospecimen context; availability must be confirmed | Keep as tumor-marker limitation/governance source rather than predictor training data. |

## What To Build Next

1. Download/export permitted GENIE BPC BRCA fields and run the priority bridge mapper.
2. Download/export Duke Breast MRI clinical-and-other-features metadata and run the priority bridge mapper.
3. Run priority external stress after any mapped export.
4. Use mutation-context mapping for common breast cancer genes without using mutations as direct treatment-response claims.
5. Use the dataset fit matrix to decide whether I-SPY2, TCGA-BRCA, or MIMIC-IV is the best next bridge for the specific question.

## Must Not Claim

- real-world clinical validation
- treatment superiority
- genetic mutation diagnosis or inherited-risk prediction
- tumor-marker recurrence conclusion
- patient-facing treatment recommendation
