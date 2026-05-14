# Public Data Strategy

MedicalAgent does not assume that one public dataset can support the whole product story.
The current data plan is a source-calibrated synthetic benchmark: real public datasets are
used where they are strong, and synthetic fields are disclosed where public coverage is weak.

## Data Reality

No single public dataset was identified that combines all required signals:

- breast-cancer treatment cycles
- CBC/lab trends
- patient-reported symptoms
- medications
- MRI treatment response
- CT/ultrasound metastatic indicators
- clinician notes
- final outcomes

This is why the system uses a timeline simulator and evaluates it with explicit realism,
lineage, leakage, noise, and external-direction checks.

## Source Roles

| Source | Role | Main Limitation |
|---|---|---|
| TCIA I-SPY2 | Serial breast MRI response and pCR labels | No CBC/symptom journey |
| TCIA Duke Breast Cancer MRI | MRI and response-label priors | Not a complete treatment timeline |
| BreastDCEDL | DL-ready breast DCE-MRI benchmark | Imaging-focused |
| MIMIC-IV | Lab distribution and missingness realism | Not breast-cancer treatment specific |
| SEER | Demographic, stage, subtype, and outcome priors | Registry data, no labs/images/symptoms |
| TCGA-BRCA / METABRIC | Clinical/genomic subtype and survival priors | Not longitudinal monitoring |

The generated manifest is stored at:

`Data/data_lineage/public_data_manifest.json`

Regenerate it with:

```bash
python scripts/build_public_data_manifest.py
```

## CT and Ultrasound Roadmap

The project can later support CT and ultrasound workflows, but the safe first step is not
"diagnose metastasis from images." The safer roadmap is:

1. Ingest CT/ultrasound report text and extract metastatic-indicator wording.
2. Route mentions such as ascites, pleural effusion, liver lesions, bone lesions, or
   suspicious lymph nodes to clinician review.
3. Add DICOM/series metadata support for CT and ultrasound studies.
4. Only after labeled public data is selected, add image-model experiments for specific,
   narrow tasks such as ascites detection or lesion segmentation.
5. Keep all outputs as non-diagnostic monitoring signals requiring clinician review.

## Claim Boundary

Public datasets improve realism and external-direction testing. They do not make this a
clinically validated diagnostic system.
