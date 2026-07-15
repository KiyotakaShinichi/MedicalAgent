from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_dataset_expansion_deep_search.json"
DEFAULT_DOC_PATH = "docs/dataset_expansion_deep_search.md"

CLAIM_BOUNDARY = (
    "Dataset expansion deep search is a planning and governance artifact. It identifies public or controlled-access "
    "sources that can improve NLCare's realism, schema coverage, or external-readiness checks. It does not mean "
    "the data has been downloaded, licensed, mapped, clinically validated, or approved for patient-facing prediction."
)


DATASET_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "genie_bpc_brca_public",
        "name": "AACR GENIE BPC Breast Cancer v1.0-public",
        "source_url": "https://www.aacr.org/professionals/research/aacr-project-genie/bpc/early-onset-brca/",
        "access": "public with AACR/GENIE data-use terms",
        "best_use": "treatment-history + genomic context external-readiness",
        "signals": ["systemic anti-neoplastic treatment histories", "tumor pathology", "clinical outcomes", "genomic alterations"],
        "oncoTrack_fit": "highest_priority",
        "why": (
            "It is the closest public source for treatment history plus clinico-genomic breast cancer context. "
            "It can improve treatment-combination schema mapping and external failure-case review."
        ),
        "limitations": [
            "Not a CBC/symptom/imaging longitudinal monitoring dataset",
            "No dosing information for investigational drugs",
            "Response/outcome semantics must be mapped carefully under PRISSMM-style definitions",
        ],
        "next_action": "Build a GENIE BPC BRCA mapper/readiness artifact; do not train patient-facing treatment recommendations.",
    },
    {
        "id": "duke_breast_mri",
        "name": "Duke Breast Cancer MRI / TCIA",
        "source_url": "https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/",
        "access": "public TCIA collection, license-sensitive",
        "best_use": "MRI + pathology + treatment/outcome/radiogenomic external bridge",
        "signals": ["DCE-MRI", "radiologist lesion boxes", "529 imaging features", "ER/PR/HER2", "Oncotype", "therapy fields", "recurrence/follow-up"],
        "oncoTrack_fit": "highest_priority",
        "why": (
            "It has unusually broad tabular context for a public breast MRI dataset, including treatment, response, "
            "recurrence, follow-up, receptor status, and imaging features."
        ),
        "limitations": [
            "Pre-operative/single-institution retrospective design",
            "Not CBC/symptom timeline",
            "License and image-download size require careful handling",
        ],
        "next_action": "Map clinical-and-other-features into canonical schema; use as imaging/treatment-context external stress, not clinical validation.",
    },
    {
        "id": "breastdcedl",
        "name": "BreastDCEDL",
        "source_url": "https://zenodo.org/records/18114231",
        "access": "public Zenodo, CC BY-NC 4.0 derivative license",
        "best_use": "deep-learning-ready MRI pCR benchmark",
        "signals": ["3D DCE-MRI", "tumor segmentation", "pCR", "HR status", "HER2 status", "age/race metadata"],
        "oncoTrack_fit": "already_integrated_expand",
        "why": "It is standardized, DL-ready, and already aligns with the response-imaging side of NLCare.",
        "limitations": ["Pre-treatment imaging focus", "No CBC/symptom/tumor-marker timeline", "pCR is not NLCare synthetic treatment success"],
        "next_action": "Expand from current tabular bridge to a small image-feature smoke benchmark if local storage allows.",
    },
    {
        "id": "ispy2_tcia",
        "name": "I-SPY2 / TCIA",
        "source_url": "https://www.cancerimagingarchive.net/collection/ispy2/",
        "access": "public TCIA, large download",
        "best_use": "serial MRI response/pCR temporal imaging benchmark",
        "signals": ["serial breast MRI", "neoadjuvant context", "pCR/response metadata", "treatment-arm context"],
        "oncoTrack_fit": "high_priority",
        "why": "It is one of the best public directions for temporal imaging response under neoadjuvant treatment.",
        "limitations": ["Large imaging download", "No CBC/symptom portal journey", "Treatment-arm access/metadata needs careful parsing"],
        "next_action": "Keep as future temporal imaging benchmark; integrate only metadata first.",
    },
    {
        "id": "qin_breast",
        "name": "QIN-BREAST / TCIA",
        "source_url": "https://www.cancerimagingarchive.net/collection/qin-breast/",
        "access": "public TCIA",
        "best_use": "longitudinal PET/CT + quantitative MRI workflow exploration",
        "signals": ["PET/CT", "quantitative MRI", "neoadjuvant treatment-assessment imaging"],
        "oncoTrack_fit": "medium_high_priority",
        "why": "It helps the CT/PET/MRI workflow side, especially study organization and modality-aware report handling.",
        "limitations": ["Imaging-focused", "No full treatment/CBC/symptom journey"],
        "next_action": "Map as imaging-workflow readiness; do not treat as response-label validation until labels are audited.",
    },
    {
        "id": "tcga_brca_gdc",
        "name": "TCGA-BRCA / NCI GDC",
        "source_url": "https://gdc.cancer.gov/about-data/publications/brca_2012",
        "access": "mixed open/controlled GDC data",
        "best_use": "somatic mutation, expression, subtype, and molecular-distribution priors",
        "signals": ["TP53/PIK3CA/GATA3 mutations", "copy number", "mRNA", "miRNA", "methylation", "RPPA", "clinical"],
        "oncoTrack_fit": "high_priority_context",
        "why": "It expands genetic mutation context and subtype realism without pretending to predict treatment response.",
        "limitations": ["Not longitudinal monitoring", "germline data controlled", "limited treatment timeline"],
        "next_action": "Add mutation-frequency/context mapping for PIK3CA, TP53, GATA3, ESR1 when available.",
    },
    {
        "id": "cptac_breast",
        "name": "CPTAC Breast Cancer",
        "source_url": "https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac",
        "access": "GDC/PDC public and controlled data depending on file",
        "best_use": "proteogenomic and assay-rich biomarker context",
        "signals": ["genomics", "proteomics", "proteogenomics", "clinical data"],
        "oncoTrack_fit": "medium_priority_context",
        "why": "Useful for advanced biomarker/proteomic context and future model cards, not core monitoring prediction today.",
        "limitations": ["Not treatment-cycle/CBC/symptom timeline", "more complex molecular data processing"],
        "next_action": "Track as future biomarker/proteomics context source after simpler public bridges are stable.",
    },
    {
        "id": "seer_research_plus",
        "name": "SEER Research Plus / SEER SSDI breast biomarkers",
        "source_url": "https://seer.cancer.gov/",
        "access": "research data request / SEER*Stat terms",
        "best_use": "population-level demographics, stage, subtype, treatment, survival distribution priors",
        "signals": ["ER", "PR", "HER2", "Ki-67 SSDI fields", "stage", "treatment utilization", "mortality"],
        "oncoTrack_fit": "medium_priority_distribution",
        "why": "Good for population/subtype distribution checks and biomarker coding discipline.",
        "limitations": ["Registry-level, not timeline-level", "no labs/symptoms/imaging sequence", "not response-score labels"],
        "next_action": "Prepare SEER field dictionary mapping; use for distribution sanity checks only.",
    },
    {
        "id": "mimic_iv",
        "name": "MIMIC-IV",
        "source_url": "https://physionet.org/content/mimiciv/2.2/",
        "access": "credentialed PhysioNet access",
        "best_use": "lab missingness/unit realism and EHR pipeline practice",
        "signals": ["hospital labs", "medications", "procedures", "notes depending on module"],
        "oncoTrack_fit": "supporting_lab_realism",
        "why": "It can improve CBC unit/missingness realism, but it is not breast-cancer treatment monitoring.",
        "limitations": ["ICU/hospital cohort, not oncology-specific", "credentialing required", "not suitable for breast response labels"],
        "next_action": "Use only for lab-distribution/unit robustness after credentialed access; keep oncology labels separate.",
    },
    {
        "id": "edrn_breast_reference",
        "name": "NCI EDRN Breast Cancer Reference Set",
        "source_url": "https://edrn.nci.nih.gov/documents/34/breast_refset_summary.pdf",
        "access": "reference-set/biospecimen context; availability must be confirmed",
        "best_use": "tumor-marker assay limitation/context source",
        "signals": ["biospecimen marker validation context", "breast cancer reference set"],
        "oncoTrack_fit": "context_only",
        "why": "Useful for educating why tumor markers require validation context, not standalone response prediction.",
        "limitations": ["Not longitudinal treatment response", "not patient-monitoring labels", "may not be directly downloadable"],
        "next_action": "Keep as tumor-marker limitation/governance source rather than predictor training data.",
    },
]


def build_dataset_expansion_deep_search(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    doc_path: str = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    payload = {
        "schema_version": "dataset_expansion_deep_search_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "dataset_count": len(DATASET_CANDIDATES),
        "highest_priority": [row for row in DATASET_CANDIDATES if row["oncoTrack_fit"] == "highest_priority"],
        "candidates": DATASET_CANDIDATES,
        "next_three_actions": [
            "Build a GENIE BPC BRCA readiness/mapper artifact for treatment histories plus genomic context.",
            "Map Duke Breast MRI clinical-and-other-features into the canonical schema as the next public treatment/imaging bridge.",
            "Add TCGA-BRCA mutation-context mapping for common breast cancer genes without using mutations as direct treatment-response claims.",
        ],
        "blocked_claims": [
            "real-world clinical validation",
            "treatment superiority",
            "genetic mutation diagnosis or inherited-risk prediction",
            "tumor-marker recurrence conclusion",
            "patient-facing treatment recommendation",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Expansion Deep Search",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Highest Priority Sources",
        "",
    ]
    for row in payload["highest_priority"]:
        lines.extend([
            f"- **{row['name']}** - {row['best_use']}",
            f"  - Source: {row['source_url']}",
            f"  - Next action: {row['next_action']}",
            "",
        ])
    lines.extend(["## Full Candidate Catalog", ""])
    lines.append("| Dataset | Best use | Fit | Access | Next action |")
    lines.append("|---|---|---|---|---|")
    for row in payload["candidates"]:
        lines.append(
            f"| [{row['name']}]({row['source_url']}) | {row['best_use']} | "
            f"{row['oncoTrack_fit']} | {row['access']} | {row['next_action']} |"
        )
    lines.extend([
        "",
        "## What To Build Next",
        "",
    ])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_three_actions"], start=1))
    lines.extend([
        "",
        "## Must Not Claim",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["blocked_claims"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = ["DEFAULT_OUTPUT_PATH", "build_dataset_expansion_deep_search"]
