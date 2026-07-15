# Public Biomarker / Tumor-Marker Dataset Readiness

This artifact is an engineering planning benchmark, not clinical validation.

Run:

```bash
python scripts/run_public_biomarker_dataset_readiness.py --live-enrich
```

Output:

```text
Data/evals/models/latest_public_biomarker_dataset_readiness.json
```

## What It Checks

- Which public sources are immediately useful for biomarker or imaging-response experiments.
- Which sources are only schema-mapping or future-access candidates.
- Whether any public tumor-marker source is suitable for treatment-response training.
- Whether NLCare should retrain/promote a biomarker/tumor-marker model now.

## Current Direction

- `BreastDCEDL` is the best immediate public imaging plus HR/HER2/pCR benchmark.
- `METABRIC` and `TCGA-BRCA` are good external schema/distribution checks for subtype and genomic fields.
- `CPTAC` is a future proteogenomic candidate after manual download and normalization.
- `AACR GENIE BPC` is high-value but access-controlled.
- `NCI-EDRN` and NCI tumor-marker references support tumor-marker limitations and education, not standalone response prediction.

## Retraining Boundary

The benchmark may recommend offline candidate training, but production promotion stays blocked unless:

- leakage audits pass,
- calibration and counterfactual stability do not regress,
- safety/refusal behavior does not regress,
- public/external evaluation is reported separately from synthetic holdout performance,
- tumor markers are never used as standalone recurrence or progression proof.

## What Must Not Be Claimed

- Do not claim clinical validation.
- Do not claim tumor markers predict recurrence or response by themselves.
- Do not claim a biomarker-enhanced model should replace clinician review.
- Do not use external-readiness mapping as proof of patient benefit.
