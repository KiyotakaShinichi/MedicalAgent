from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_dataset_fit_matrix.json"
DEFAULT_DOC_PATH = "docs/dataset_fit_matrix.md"

CLAIM_BOUNDARY = (
    "Dataset fit scores are planning heuristics for engineering roadmap priority. They do not mean data "
    "has been accessed, licensed, mapped, clinically validated, or approved for patient-facing prediction."
)


DATASETS: list[dict[str, Any]] = [
    {
        "id": "genie_bpc_brca",
        "name": "AACR GENIE BPC Breast Cancer",
        "source_url": "https://www.aacr.org/professionals/research/aacr-project-genie/bpc/early-onset-brca/",
        "access": "public subset plus Synapse terms",
        "scores": {"treatment": 5, "temporal": 3, "imaging": 0, "biomarker": 4, "genomic": 5, "tumor_marker": 0, "cbc_labs": 0, "student_access": 3},
        "best_next_use": "systemic treatment history plus clinico-genomic context bridge",
    },
    {
        "id": "duke_breast_mri",
        "name": "Duke Breast Cancer MRI / TCIA",
        "source_url": "https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/",
        "access": "public TCIA, large download",
        "scores": {"treatment": 3, "temporal": 1, "imaging": 5, "biomarker": 4, "genomic": 2, "tumor_marker": 0, "cbc_labs": 0, "student_access": 4},
        "best_next_use": "MRI/pathology/treatment-context external stress",
    },
    {
        "id": "ispy2_tcia",
        "name": "I-SPY2 / TCIA",
        "source_url": "https://www.cancerimagingarchive.net/collection/ispy2/",
        "access": "public TCIA, large download",
        "scores": {"treatment": 3, "temporal": 5, "imaging": 5, "biomarker": 3, "genomic": 1, "tumor_marker": 0, "cbc_labs": 0, "student_access": 3},
        "best_next_use": "serial MRI pCR response benchmark after metadata mapping",
    },
    {
        "id": "breastdcedl",
        "name": "BreastDCEDL",
        "source_url": "https://zenodo.org/records/18114231",
        "access": "public Zenodo license terms",
        "scores": {"treatment": 2, "temporal": 2, "imaging": 5, "biomarker": 3, "genomic": 0, "tumor_marker": 0, "cbc_labs": 0, "student_access": 5},
        "best_next_use": "already integrated pCR/MRI bridge; extend only if storage allows",
    },
    {
        "id": "tcga_brca",
        "name": "TCGA-BRCA / GDC",
        "source_url": "https://gdc.cancer.gov/about-data/publications/brca_2012",
        "access": "public and controlled tiers",
        "scores": {"treatment": 1, "temporal": 0, "imaging": 0, "biomarker": 4, "genomic": 5, "tumor_marker": 0, "cbc_labs": 0, "student_access": 4},
        "best_next_use": "mutation/subtype context mapping, not response validation",
    },
    {
        "id": "metabric",
        "name": "METABRIC",
        "source_url": "https://www.cbioportal.org/study/summary?id=brca_metabric",
        "access": "public via cBioPortal",
        "scores": {"treatment": 1, "temporal": 0, "imaging": 0, "biomarker": 4, "genomic": 4, "tumor_marker": 0, "cbc_labs": 0, "student_access": 4},
        "best_next_use": "subtype/outcome distribution context",
    },
    {
        "id": "qin_breast",
        "name": "QIN-BREAST / TCIA",
        "source_url": "https://www.cancerimagingarchive.net/collection/qin-breast/",
        "access": "public TCIA",
        "scores": {"treatment": 2, "temporal": 4, "imaging": 5, "biomarker": 1, "genomic": 0, "tumor_marker": 0, "cbc_labs": 0, "student_access": 3},
        "best_next_use": "quantitative imaging workflow exploration",
    },
    {
        "id": "seer_research",
        "name": "SEER Research Plus",
        "source_url": "https://seer.cancer.gov/",
        "access": "research data agreement",
        "scores": {"treatment": 2, "temporal": 0, "imaging": 0, "biomarker": 3, "genomic": 0, "tumor_marker": 0, "cbc_labs": 0, "student_access": 2},
        "best_next_use": "population distribution and coding discipline",
    },
    {
        "id": "mimic_iv",
        "name": "MIMIC-IV",
        "source_url": "https://physionet.org/content/mimiciv/2.2/",
        "access": "credentialed PhysioNet",
        "scores": {"treatment": 1, "temporal": 4, "imaging": 0, "biomarker": 0, "genomic": 0, "tumor_marker": 0, "cbc_labs": 5, "student_access": 3},
        "best_next_use": "lab missingness, unit, and EHR workflow realism only",
    },
    {
        "id": "edrn_breast_reference",
        "name": "NCI EDRN Breast Cancer Reference Set",
        "source_url": "https://edrn.nci.nih.gov/documents/34/breast_refset_summary.pdf",
        "access": "reference/biospecimen context",
        "scores": {"treatment": 0, "temporal": 0, "imaging": 0, "biomarker": 2, "genomic": 0, "tumor_marker": 3, "cbc_labs": 0, "student_access": 2},
        "best_next_use": "tumor-marker limitation context only",
    },
]


def build_dataset_fit_matrix(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    doc_path: str = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    rows = [_scored_dataset(item) for item in DATASETS]
    rows.sort(key=lambda item: item["fit_score"], reverse=True)
    payload = {
        "schema_version": "dataset_fit_matrix_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "dataset_count": len(rows),
        "top_5": [item["id"] for item in rows[:5]],
        "datasets": rows,
        "recommendation": {
            "next_two_to_map": ["genie_bpc_brca", "duke_breast_mri"],
            "next_after_that": ["ispy2_tcia", "tcga_brca", "mimic_iv"],
            "production_training_allowed": False,
            "reason": "Best sources improve schema and stress testing, but none provide full OncoTrack clinician-reviewed temporal labels.",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_markdown(_resolve(doc_path), payload)
    return payload


def _scored_dataset(item: dict[str, Any]) -> dict[str, Any]:
    scores = item["scores"]
    fit_score = (
        scores["treatment"] * 1.4
        + scores["temporal"] * 1.2
        + scores["imaging"] * 1.2
        + scores["biomarker"] * 1.0
        + scores["genomic"] * 1.0
        + scores["cbc_labs"] * 0.9
        + scores["student_access"] * 0.8
        + scores["tumor_marker"] * 0.4
    )
    return {
        **item,
        "fit_score": round(fit_score, 2),
        "safe_role": item["best_next_use"],
        "blocked_claim": "Do not use this dataset to claim OncoTrack is clinically validated.",
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Fit Matrix",
        "",
        f"Generated at: {payload['generated_at']}",
        "",
        payload["claim_boundary"],
        "",
        "| Dataset | Fit score | Best next use | Access |",
        "|---|---:|---|---|",
    ]
    for item in payload["datasets"]:
        lines.append(
            f"| [{item['name']}]({item['source_url']}) | {item['fit_score']} | "
            f"{item['best_next_use']} | {item['access']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
