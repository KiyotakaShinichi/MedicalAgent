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
| TCIA QIN-BREAST | Longitudinal PET/CT and quantitative MRI in neoadjuvant breast cancer | Imaging-focused, no CBC/symptom journey |
| TCIA FDG-PET-CT-Lesions | Whole-body PET/CT lesion segmentations | Not breast-cancer specific |
| NIH DeepLesion | Large CT lesion boxes and measurements | Lesion detection only, no cancer-origin labels |
| BUSI breast ultrasound | Breast ultrasound images with labels/masks | Small, breast-only, not metastatic/ascites |
| BUS-UCLM breast ultrasound | Breast ultrasound lesion segmentation | Breast-only, not a treatment timeline |
| MIMIC-IV | Lab distribution and missingness realism | Not breast-cancer treatment specific |
| SEER | Demographic, stage, subtype, and outcome priors | Registry data, no labs/images/symptoms |
| TCGA-BRCA / METABRIC | Clinical/genomic subtype and survival priors | Not longitudinal monitoring |

The generated manifest is stored at:

`Data/data_lineage/public_data_manifest.json`

Regenerate it with:

```bash
python scripts/build_public_data_manifest.py
```

Public imaging readiness artifacts are stored at:

```text
Data/public_imaging/
```

Build them with:

```bash
python scripts/build_public_imaging_manifest.py
python scripts/run_ultrasound_baseline.py --dataset-root Datasets/BUSI
python scripts/run_ct_lesion_workflow.py --dataset-root Datasets/DeepLesion
python scripts/run_sim_to_public_imaging_report.py
```

If the datasets are not downloaded locally, these scripts still write explicit
`unavailable` artifacts. That is intentional so the dashboard shows missing data
instead of implying hidden validation.

## CT and Ultrasound Roadmap

The project can later support CT and ultrasound workflows, but the safe first step is not
"diagnose metastasis from images." The safer roadmap is:

1. Ingest CT/ultrasound report text and extract metastatic-indicator wording.
2. Route mentions such as ascites, pleural effusion, liver lesions, bone lesions, or
   suspicious lymph nodes to clinician review.
3. Add DICOM/series metadata support for CT and ultrasound studies.
4. Use QIN-BREAST for breast cancer PET/CT workflow exploration, FDG-PET-CT-Lesions
   and DeepLesion for CT/PET-CT lesion baselines, and BUSI/BUS-UCLM for breast
   ultrasound segmentation baselines.
5. Only after task-specific labels are selected, add image-model experiments for specific,
   narrow tasks such as lesion segmentation or ascites-related report/image review.
6. Keep all outputs as non-diagnostic monitoring signals requiring clinician review.

## Claim Boundary

Public datasets improve realism and external-direction testing. They do not make this a
clinically validated diagnostic system.
