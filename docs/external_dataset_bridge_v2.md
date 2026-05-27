# External Dataset Bridge V2

The external dataset bridge ranks public and restricted breast-cancer-relevant datasets for future schema mapping and stress testing.

Run:

```bash
python scripts/run_external_dataset_bridge_v2.py
```

Output:

```text
Data/evals/models/latest_external_dataset_bridge_v2.json
```

Highest-priority bridges:

- AACR GENIE BPC Breast Cancer: treatment-history, genomic, outcome-context bridge when access terms are satisfied.
- Duke Breast Cancer MRI / TCIA: MRI, pathology, receptor status, treatment, recurrence/follow-up schema bridge.

Supporting schema bridges include TCGA-BRCA/GDC, METABRIC/cBioPortal, and I-SPY/TCIA-style imaging-response resources.

Boundary: this is dataset-readiness and external-stress planning only. It is not clinical validation, not hospital interoperability, and not treatment-efficacy evidence.
