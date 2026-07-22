# External dataset integration strategy

## Decision rule

External data is admitted only when its task, label, access terms, and feature
semantics are explicit. A public breast-cancer dataset is not automatically a
valid test of NLCare's synthetic longitudinal monitoring heads. Target-matched
stress tests stay separate from target-mismatched schema or distribution
bridges, and no external dataset can promote patient-facing behavior in the
current project.

## Priority matrix

| Priority | Dataset | Useful role | Main mismatch or blocker | Current decision |
| --- | --- | --- | --- | --- |
| A1 | TCIA I-SPY2 | Clinical plus serial MRI feature engineering; separate pCR benchmark | pCR is not an NLCare monitoring target | Integrated as isolated stress test |
| A2 | Duke Breast Cancer MRI / TCIA | MRI, pathology, receptor, treatment, outcome, genomic, and radiomic pipeline stress | Imaging workload and endpoint mapping; only a pCR subset | Next public bridge |
| B1 | AACR GENIE BPC Breast | Treatment-history and genomic-context research | Controlled terms/access and different observational endpoints | Prepare access request; do not claim availability |
| B2 | METABRIC | Subtype/genomic/outcome distribution and schema checks | Cross-sectional/survival outcomes are target-mismatched | Schema/distribution bridge only |
| B3 | TCGA-BRCA | Multi-omic and pathology schema checks | Coarse treatment timing and target mismatch | Schema bridge only |
| C1 | SEER / SEER-Medicare | Population and treatment-distribution stress | Access restrictions, first-course treatment limits, claims confounding | Distribution or future restricted bridge |
| C2 | MIMIC-IV | Generic EHR ingestion, units, missingness, and temporal pipeline tests | Not breast-treatment-specific; credentialed access | Infrastructure stress only |
| C3 | QIN-BREAST-02 | Quantitative MRI ingestion proof | Only 13 subjects | Never a primary model benchmark |

## Implemented I-SPY2 bridge

The repository checksum-locks the official TCIA clinical and multi-feature MRI
spreadsheets, joins 384 MRI rows to the 985-row clinical cohort, hashes the
trial subject identifier, and excludes treatment arm from both export and
features. It compares logistic regression and gradient boosting over five
fixed stratified splits using clinical-only, baseline-MRI, and early-change
feature sets.

The strongest internal reading is that early-change MRI features improve this
separate pCR task on these repeated splits. That finding does not transfer to
NLCare's synthetic response, regression, or toxicity heads and is not clinical
validation.

Run:

```text
python scripts/run_ispy2_tcia_tabular_bridge.py
```

Artifacts:

- `Data/external_bridge/ispy2_tcia/canonical_ispy2_tabular.csv`
- `Data/evals/models/latest_ispy2_tcia_external_stress.json`

## Official primary sources

- TCIA I-SPY2: <https://www.cancerimagingarchive.net/collection/ispy2/>
- TCIA Duke Breast Cancer MRI: <https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/>
- AACR GENIE BPC Breast: <https://www.synapse.org/Synapse:syn27056172/files/>
- SEER data access: <https://seer.cancer.gov/data/access.html>
- MIMIC-IV 3.1: <https://physionet.org/content/mimiciv/3.1/>

## What still cannot be claimed

- No external cohort validates an NLCare model head.
- No public dataset establishes patient benefit, safety, or clinical utility.
- No treatment-arm or treatment-selection model is authorized.
- No real patient data is used in the NLCare live application.
- No clinical, institutional, IRB, or regulatory review has occurred.
