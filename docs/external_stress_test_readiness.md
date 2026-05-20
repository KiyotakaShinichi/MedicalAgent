# External Stress-Test Readiness

External/public dataset mappings are used for engineering stress tests only.
They are not clinical validation, not real-world model validation, and not a
reason to promote model outputs.

Run:

```bash
python scripts/run_external_stress_readiness.py
```

Output:

```text
Data/evals/models/latest_external_stress_test_readiness.json
```

The readiness artifact covers:

- TCGA-BRCA schema rows
- METABRIC schema rows
- BreastDCEDL / I-SPY common-feature rows
- Duke MRI / TCIA schema candidates

For each dataset it reports:

- mapped fields
- missing NLCare fields
- expected abstention behavior
- failure cases
- why the dataset cannot support clinical claims yet

## Why This Is Not Clinical Validation

Most public/external rows do not contain the full NLCare longitudinal journey:
CBC trends, patient-reported symptoms, medication/treatment-cycle sequence,
imaging timeline, tumor-marker context, and clinician-reviewed labels for the
exact monitoring question.

Outcome labels such as pCR, survival, recurrence, or real-world response are
useful stress endpoints but are not identical to the synthetic monitoring heads.
They cannot establish treatment-response validity for this project.
