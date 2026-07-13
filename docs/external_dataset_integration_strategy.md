# External Dataset Integration Strategy

External dataset integration is an engineering roadmap artifact. It ranks public and restricted datasets for schema mapping, stress testing, synthetic-noise design, and future access planning. It does not mean the datasets have been downloaded, licensed, harmonized, used for clinical validation, or approved for patient-facing prediction, diagnosis, treatment, prognosis, genetic-risk interpretation, tumor-marker interpretation, or medication decisions.

## Highest-ROI Integrations

1. **breastdcedl** - metadata-only pCR/imaging-response stress benchmark
   - Why: Best immediate public bridge for external response-context stress without pretending it is the same label.

2. **duke_breast_mri_tcia** - canonical imaging/pathology/treatment-context schema mapping
   - Why: Broad public imaging cohort with receptor/pathology/treatment/follow-up context for schema discipline.

3. **aacr_genie_bpc_brca** - field-contract and access-packet hardening
   - Why: Closest future bridge for treatment-history plus genomic/outcome context, but access and target mismatch remain blockers.

4. **mimic_iv** - lab-unit/missingness prior plan
   - Why: Useful to make synthetic CBC/lab noise less clean, while staying explicit that it is not breast-response data.

5. **clinvar** - VUS/genetics safety-boundary fixture generation
   - Why: Improves unsafe-genetics eval coverage without giving patient-specific variant interpretation.

## Recommended Sequence

1. Integrate metadata-only BreastDCEDL/I-SPY pCR stress tests.
2. Map Duke MRI clinical/pathology/treatment context into canonical schema.
3. Use MIMIC-IV only for missingness/unit/noise priors after credentialed access.
4. Use ClinVar/BRCA Exchange only for genetics/VUS boundary and schema tests.
5. Keep GENIE BPC and SEER-Medicare as future restricted-access field contracts until access exists.

## Dataset Matrix

| Priority | Dataset | Category | Recommended use | Not allowed use |
|---:|---|---|---|---|
| 1 | [BreastDCEDL](https://zenodo.org/records/17274053) | response_pcr_stress_testing | Use as the first external imaging-response stress benchmark and feature-schema bridge. | Do not use pCR performance to claim NLCare predicts clinical treatment response or patient outcomes. |
| 2 | [I-SPY2 / TCIA](https://www.cancerimagingarchive.net/collection/ispy2/) | temporal_imaging_response | Map serial imaging timepoints into the canonical timeline as response-context stress data. | Do not infer treatment efficacy, regimen choice, or patient-specific response from NLCare outputs. |
| 3 | [Duke Breast Cancer MRI / TCIA](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/) | imaging_pathology_schema_bridge | Use as the next public schema bridge after BreastDCEDL for imaging plus treatment-context mapping. | Do not treat Duke MRI mapping as evidence of clinical response prediction or workflow benefit. |
| 4 | [MAMA-MIA](https://pmc.ncbi.nlm.nih.gov/articles/PMC11923173/) | imaging_domain_shift_stress | Use to test imaging domain shift, segmentation-derived feature robustness, and fairness stress planning. | Do not present segmentation performance as NLCare clinical monitoring validation. |
| 5 | [I-SPY1 / ACRIN 6657](https://wiki.cancerimagingarchive.net/display/Public/ISPY1) | temporal_imaging_response | Use after BreastDCEDL/I-SPY2 as a smaller imaging-response consistency check. | Do not use recurrence/follow-up fields for prognosis claims. |
| 6 | [AACR GENIE BPC Breast Cancer](https://www.aacr.org/professionals/research/aacr-project-genie/bpc/) | treatment_genomic_outcome_bridge | Prepare a restricted/public access packet and field-contract mapper for treatment-context governance. | Do not train or claim patient-facing treatment recommendations. |
| 7 | [TCGA-BRCA / NCI GDC](https://gdc.cancer.gov/about-data/publications/brca_2012) | biomarker_genomic_context | Use for canonical molecular-schema mapping, mutation-context coverage, and target-mismatch education. | Do not train survival/prognosis models and present them as NLCare monitoring performance. |
| 8 | [METABRIC](https://www.cbioportal.org/study/summary?id=brca_metabric) | biomarker_genomic_context | Use as second molecular external cohort for schema consistency and subtype stress. | Do not call METABRIC survival modeling clinical validation of NLCare. |
| 9 | [CPTAC Breast Cancer](https://gdc.cancer.gov/about-data/publications/CPTAC-3_2020_1) | biomarker_genomic_context | Track as future biomarker/proteogenomics context after simpler bridges are stable. | Do not add proteomic outputs as patient-facing predictor authority. |
| 10 | [SEER Breast Cancer / Research Plus](https://seer.cancer.gov/) | population_distribution_context | Use for subtype/stage/treatment-distribution sanity checks and schema coding discipline. | Do not use survival endpoints for patient-facing prognosis or response prediction. |
| 11 | [SEER-Medicare](https://healthcaredelivery.cancer.gov/seermedicare/) | restricted_treatment_sequence_context | Keep in restricted-data access roadmap for treatment-combination realism. | Do not imply access has been granted or that claims-derived patterns are treatment advice. |
| 12 | [MIMIC-IV](https://mimic.mit.edu/docs/IV/) | synthetic_noise_realism_priors | Use only for synthetic-noise and missingness priors after credentialing. | Do not use MIMIC-IV to train breast cancer response, prognosis, or treatment-selection models. |
| 13 | [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) | genetics_vus_safety_boundary | Use to generate safe VUS wording tests, conflict examples, and genetics-record schema checks. | Do not interpret a user's variant, infer inherited risk, or provide genetic counseling. |
| 14 | [BRCA Exchange](https://brcaexchange.org/factsheet) | genetics_vus_safety_boundary | Use as a genetics-boundary and record-normalization reference. | Do not convert BRCA Exchange classifications into patient-facing risk estimates. |
| 15 | [NCI EDRN Breast Cancer Reference Set](https://edrn.cancer.gov/data-and-resources/publications/25471344-2344-construction-and-analysis-of-the-nci-edrn-breast-cancer-reference-set-for-circulating-markers-of-disease/) | tumor_marker_limitation_context | Keep as tumor-marker limitation and source-governance context only. | Do not train response predictors or make recurrence/tumor-marker conclusions from this resource. |

## Blocked Claims

- clinical validation
- real-world patient safety
- patient benefit
- production healthcare readiness
- diagnostic authority
- treatment recommendation
- prognosis or survival prediction
- genetic-risk interpretation
- tumor-marker interpretation
- medication or supplement safety advice

## Notes

- This matrix can strengthen ML credibility by making external stress tests and schema mapping concrete.
- It does not authorize model promotion, patient-facing predictions, or clinical claims.
- Every dataset row has `clinical_validation: false` in the machine-readable artifact.
