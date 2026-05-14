from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = "Data/data_lineage/public_data_manifest.json"


PUBLIC_DATA_SOURCES: list[dict[str, Any]] = [
    {
        "id": "tcia_ispy2",
        "name": "I-SPY2 Breast MRI Collection",
        "provider": "The Cancer Imaging Archive",
        "url": "https://www.cancerimagingarchive.net/collection/ispy2/",
        "access": "public TCIA collection; follow TCIA data citation and use terms",
        "modalities": ["DCE-MRI", "clinical metadata", "pathologic response"],
        "use_in_project": [
            "External imaging-response validation",
            "pCR/non-pCR label priors",
            "MRI response feature sanity checks",
        ],
        "covers": {
            "breast_cancer": True,
            "treatment_response": True,
            "longitudinal_imaging": True,
            "cbc_labs": False,
            "symptoms": False,
            "ct_ultrasound": False,
            "metastatic_indicators": False,
        },
        "limitations": [
            "Imaging-response focused; does not provide a full patient monitoring journey.",
            "Does not provide longitudinal CBC or patient-reported symptom data.",
        ],
    },
    {
        "id": "tcia_duke_breast_mri",
        "name": "Duke Breast Cancer MRI",
        "provider": "The Cancer Imaging Archive",
        "url": "https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/",
        "access": "public TCIA collection; follow TCIA data citation and use terms",
        "modalities": ["breast MRI", "clinical metadata", "pathologic response fields"],
        "use_in_project": [
            "External MRI feature distribution comparison",
            "Tumor-size and response-label priors",
            "Imaging model stress testing",
        ],
        "covers": {
            "breast_cancer": True,
            "treatment_response": True,
            "longitudinal_imaging": False,
            "cbc_labs": False,
            "symptoms": False,
            "ct_ultrasound": False,
            "metastatic_indicators": False,
        },
        "limitations": [
            "Primarily pre-operative MRI; not a complete treatment timeline.",
            "Response labels are not available for every patient.",
        ],
    },
    {
        "id": "breastdcedl",
        "name": "BreastDCEDL",
        "provider": "Scientific Data / TCIA-derived cohorts",
        "url": "https://www.nature.com/articles/s41597-026-06589-6",
        "access": "public research dataset/paper; use source-specific access instructions",
        "modalities": ["DCE-MRI", "standardized image tensors", "clinical metadata", "pCR labels"],
        "use_in_project": [
            "Deep-learning-ready breast MRI benchmark",
            "External validation direction for response modeling",
            "Feature distribution comparison against synthetic journeys",
        ],
        "covers": {
            "breast_cancer": True,
            "treatment_response": True,
            "longitudinal_imaging": False,
            "cbc_labs": False,
            "symptoms": False,
            "ct_ultrasound": False,
            "metastatic_indicators": False,
        },
        "limitations": [
            "Strong for DCE-MRI response modeling, weak for daily clinical monitoring.",
            "Does not include CBC, medication administration, or patient-reported symptoms.",
        ],
    },
    {
        "id": "mimic_iv",
        "name": "MIMIC-IV",
        "provider": "PhysioNet / MIT-LCP",
        "url": "https://physionet.org/content/mimiciv/3.0/",
        "access": "credentialed PhysioNet access required",
        "modalities": ["EHR", "labs", "medications", "diagnoses", "procedures"],
        "use_in_project": [
            "Lab realism calibration",
            "CBC missingness and unit-noise priors",
            "Hospital-EHR timing irregularity simulation",
        ],
        "covers": {
            "breast_cancer": False,
            "treatment_response": False,
            "longitudinal_imaging": False,
            "cbc_labs": True,
            "symptoms": False,
            "ct_ultrasound": False,
            "metastatic_indicators": False,
        },
        "limitations": [
            "Not breast-cancer treatment specific.",
            "Best used for lab-distribution realism, not oncology response labels.",
        ],
    },
    {
        "id": "seer",
        "name": "SEER Research Data",
        "provider": "National Cancer Institute",
        "url": "https://seer.cancer.gov/data/seerstat/",
        "access": "public-use research data with SEER data-use agreement",
        "modalities": ["cancer registry", "stage", "subtype", "survival", "first-course treatment variables"],
        "use_in_project": [
            "Demographic and stage priors",
            "Breast subtype distribution priors",
            "Population-level outcome context",
        ],
        "covers": {
            "breast_cancer": True,
            "treatment_response": False,
            "longitudinal_imaging": False,
            "cbc_labs": False,
            "symptoms": False,
            "ct_ultrasound": False,
            "metastatic_indicators": False,
        },
        "limitations": [
            "Registry data, not a patient monitoring timeline.",
            "No CBC trajectories, imaging series, or free-text symptom stream.",
        ],
    },
    {
        "id": "tcga_brca_metabric",
        "name": "TCGA-BRCA / METABRIC via cBioPortal",
        "provider": "NCI / cBioPortal",
        "url": "https://www.cbioportal.org/",
        "access": "public cBioPortal exploration/download where permitted by study",
        "modalities": ["clinical", "genomic", "subtype", "survival"],
        "use_in_project": [
            "Subtype and survival priors",
            "Clinical covariate sanity checks",
            "Documentation of non-imaging/non-lab external context",
        ],
        "covers": {
            "breast_cancer": True,
            "treatment_response": False,
            "longitudinal_imaging": False,
            "cbc_labs": False,
            "symptoms": False,
            "ct_ultrasound": False,
            "metastatic_indicators": False,
        },
        "limitations": [
            "Useful for priors, not for treatment-cycle monitoring.",
            "No CBC nadirs, patient chat symptoms, or serial response imaging workflow.",
        ],
    },
    {
        "id": "future_metastatic_imaging",
        "name": "Future metastatic-imaging extension",
        "provider": "To be selected from TCIA/Kaggle/Grand Challenge/clinical-public sources",
        "url": "",
        "access": "not yet integrated",
        "modalities": ["CT", "ultrasound", "radiology report text"],
        "use_in_project": [
            "Support extraction of metastatic indicator wording from reports",
            "Possible segmentation/classification experiments for ascites, liver lesions, pleural effusion, or bone lesions",
            "Route suspicious findings to clinician review instead of diagnosis",
        ],
        "covers": {
            "breast_cancer": False,
            "treatment_response": False,
            "longitudinal_imaging": False,
            "cbc_labs": False,
            "symptoms": False,
            "ct_ultrasound": True,
            "metastatic_indicators": True,
        },
        "limitations": [
            "Not implemented as a diagnostic reader.",
            "Requires modality-specific labels and radiology expertise before any model claim.",
        ],
    },
]


FEATURE_NEEDS: list[dict[str, Any]] = [
    {
        "need": "Breast MRI treatment response",
        "status": "covered_by_public_data",
        "sources": ["tcia_ispy2", "tcia_duke_breast_mri", "breastdcedl"],
        "project_action": "Use for external response-label direction and imaging feature calibration.",
    },
    {
        "need": "Longitudinal CBC realism",
        "status": "partially_covered",
        "sources": ["mimic_iv"],
        "project_action": "Use EHR lab distributions to tune synthetic missingness, units, timing jitter, and outliers.",
    },
    {
        "need": "Patient-reported symptom trajectories",
        "status": "not_well_covered_publicly",
        "sources": [],
        "project_action": "Keep synthetic symptom generation, calibrate severity rates from trial adverse-event tables where available.",
    },
    {
        "need": "Treatment-cycle schedules",
        "status": "partially_covered",
        "sources": ["seer", "tcia_ispy2"],
        "project_action": "Use public trial schemas and guideline/paper schedules as priors, not as patient-level ground truth.",
    },
    {
        "need": "Metastatic CT/ultrasound indicators",
        "status": "future_extension",
        "sources": ["future_metastatic_imaging"],
        "project_action": "Start with report-text extraction and clinician-review routing; only later add image models with labeled data.",
    },
    {
        "need": "End-to-end real treatment journey",
        "status": "not_available_as_single_public_dataset",
        "sources": [],
        "project_action": "Use source-calibrated synthetic journeys and clearly disclose the limitation.",
    },
]


def build_public_data_manifest(output_path: str | None = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    payload = {
        "schema_version": "public_data_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "usable_with_limitations",
        "central_data_reality": (
            "No single public dataset was identified that combines breast-cancer treatment cycles, CBC trends, "
            "symptoms, medications, MRI response, CT/ultrasound metastatic indicators, clinician notes, and outcomes."
        ),
        "recommended_strategy": (
            "Use a source-calibrated synthetic patient journey benchmark: real public MRI-response datasets for imaging "
            "signals, MIMIC-IV for lab-distribution realism, SEER/TCGA/METABRIC for subtype and outcome priors, and "
            "explicit documentation for all synthetic fields."
        ),
        "sources": PUBLIC_DATA_SOURCES,
        "feature_feasibility": FEATURE_NEEDS,
        "claim_boundary": (
            "These sources improve realism and external-direction testing, but they do not make the system clinically "
            "validated or diagnostic."
        ),
    }
    payload["manifest_hash"] = _stable_hash(payload)
    if output_path:
        _write_json(output_path, payload)
    return payload


def _stable_hash(payload: dict[str, Any]) -> str:
    material = json.dumps(
        {k: v for k, v in payload.items() if k not in {"generated_at", "manifest_hash"}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
