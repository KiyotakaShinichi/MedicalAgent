# Public Treatment-Combination Dataset Readiness

This artifact maps public or accessible breast-cancer treatment data sources for NLCare. It is an engineering planning artifact, not a treatment recommendation system.

Run:

```bash
python scripts/run_public_treatment_dataset_readiness.py
```

Output:

```text
Data/evals/models/latest_public_treatment_dataset_readiness.json
```

## Treatment Modalities Tracked

- surgery
- radiation therapy
- chemotherapy
- HER2-targeted therapy
- endocrine therapy
- immunotherapy
- PARP inhibitor context
- supportive-care medications

## Combination Patterns

The schema supports patients having one or many modalities:

- endocrine only
- chemotherapy only
- chemotherapy plus HER2-targeted therapy
- chemotherapy plus immunotherapy
- surgery plus radiation plus endocrine therapy
- chemotherapy plus surgery plus radiation plus endocrine therapy
- chemotherapy plus targeted therapy plus surgery plus radiation plus endocrine therapy

These are timeline categories and ML feature-ablation candidates only. They must not be used to recommend a regimen.

## Best Sources

- `AACR GENIE BPC BrCa`: best future real-world treatment-history and outcome candidate, but access/terms are required.
- `SEER`: useful population-level treatment distribution source with coarse treatment fields and limitations.
- `SEER-Medicare`: useful claims-based treatment-combination source after application/DUA approval.
- `BreastDCEDL I-SPY2`: strong imaging/pCR response benchmark, but not a full treatment-combination timeline.
- `Duke Breast MRI`: useful imaging plus receptor/treatment-context source after manual mapping.
- `TCGA-BRCA/GDC`: useful coarse treatment-schema sanity check, not detailed regimen training.
- `ClinicalTrials.gov`: useful regimen vocabulary source, not patient-level outcomes.

## Boundary

Treatment data can help NLCare organize timeline context and benchmark model robustness. It cannot make treatment decisions, compare real-world treatment efficacy, or tell a patient to start, stop, delay, or switch therapy.
