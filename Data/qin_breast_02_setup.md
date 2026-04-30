# QIN-BREAST-02 Setup

Chosen dataset: QIN-BREAST-02 from The Cancer Imaging Archive.

Official page: https://www.cancerimagingarchive.net/collection/qin-breast-02/

## Why This Dataset

QIN-BREAST-02 is small enough for limited hardware compared with large breast MRI datasets.

- Subjects: 13
- Imaging size: about 4.19 GB
- Imaging type: breast MRI DICOM
- Supporting data: clinical, diagnosis, molecular test, treatment, follow-up, protocol
- Clinical spreadsheet size: about 17 KB
- Use case: neoadjuvant therapy monitoring with repeated breast MRI time points

## Recommended Download Order

1. Download the clinical XLSX first.
2. Convert the relevant sheet to CSV.
3. Import metadata into this app with dataset type `qin_breast_02`.
4. Download DICOM images only after metadata import works.
5. Start with 1-2 subjects before downloading the full 4.19 GB image set.

## Import Into This App

For clinical/profile rows, map or rename columns to the canonical fields in:

`Data/breast_monitoring_data_dictionary.md`

For imaging timeline rows, use:

```csv
patient_id,date,modality,report_type,body_site,findings,impression
QINB02_001,2026-01-01,Breast MRI,Pre-treatment MRI,Breast,Right breast enhancing mass measuring 3.4 cm. BI-RADS 6.,Baseline breast MRI before neoadjuvant therapy.
```

You can paste CSV text in the dashboard under **Import CSV**:

- import type: `imaging_reports`
- dataset: `qin_breast_02`

Or call the API:

```json
{
  "import_type": "imaging_reports",
  "dataset": "qin_breast_02",
  "csv_text": "patient_id,date,modality,report_type,body_site,findings,impression\nQINB02_001,2026-01-01,Breast MRI,Pre-treatment MRI,Breast,Right breast mass measuring 3.4 cm,Baseline MRI\n"
}
```

## Important Limitation

QIN-BREAST-02 does not solve the CBC problem by itself. Public breast MRI datasets usually do not include serial CBC values linked to each MRI patient. For now, use QIN-BREAST-02 for breast MRI/treatment metadata and synthetic CBC values for chemotherapy toxicity monitoring.

## Future Breast MRI ML Target

Use QIN-BREAST-02 first for:

1. local DICOM loading checks,
2. MRI time-point organization,
3. metadata-linked timeline generation,
4. later tumor segmentation or response modeling.

Do not train on all image series immediately. First identify the useful DCE/T1/DWI series and prepare a small subject-level subset.
