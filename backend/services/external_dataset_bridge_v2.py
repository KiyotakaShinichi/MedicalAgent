"""External dataset bridge v2 readiness map.

The bridge ranks public/restricted datasets for stress testing and future data
access.  It explicitly avoids clinical validation claims.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_external_dataset_bridge_v2.json")

CLAIM_BOUNDARY = (
    "External dataset bridge v2 is a schema/readiness map. It is not clinical "
    "validation, does not prove treatment utility, and is not connected to real "
    "patient care."
)


DATASETS: list[dict[str, Any]] = [
    {
        "dataset_id": "aacr_genie_bpc_brca_public",
        "name": "AACR GENIE BPC Breast Cancer",
        "access": "public_release_or_synapse_terms",
        "best_use": "future treatment-history plus genomic/outcome stress testing",
        "fit_score": 0.93,
        "mapped_fields": ["genomics", "prior_treatments", "tumor_pathology", "clinical_outcomes", "real_world_response_framework"],
        "missing_for_nlcare": ["longitudinal CBC", "patient-reported symptoms", "full local monitoring timeline"],
        "source_url": "https://www.aacr.org/professionals/research/aacr-project-genie/bpc/early-onset-brca/",
        "readiness": "highest_priority_access_bridge",
        "claim_boundary": "Can support future external stress tests; not direct validation of synthetic monitoring heads.",
    },
    {
        "dataset_id": "duke_breast_cancer_mri_tcia",
        "name": "Duke Breast Cancer MRI / TCIA",
        "access": "public_tcia",
        "best_use": "MRI/pathology/receptor/treatment/follow-up schema bridge",
        "fit_score": 0.91,
        "mapped_fields": ["MRI", "segmentation", "clinical", "pathology", "treatment", "follow_up", "genomics"],
        "missing_for_nlcare": ["CBC trend", "patient support chat", "synthetic toxicity review labels"],
        "source_url": "https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/",
        "readiness": "highest_priority_public_bridge",
        "claim_boundary": "Useful for imaging-response and schema stress tests; not proof of patient benefit.",
    },
    {
        "dataset_id": "tcga_brca_gdc",
        "name": "TCGA-BRCA / GDC",
        "access": "public_and_controlled_genomic_portal",
        "best_use": "biomarker/genomic canonical mapping and subtype stress testing",
        "fit_score": 0.78,
        "mapped_fields": ["copy_number", "methylation", "exome", "mrna", "mirna", "rppa", "subtype_context"],
        "missing_for_nlcare": ["treatment-cycle monitoring", "CBC", "symptoms", "temporal response labels"],
        "source_url": "https://gdc.cancer.gov/about-data/publications/brca_2012",
        "readiness": "schema_mapping_bridge",
        "claim_boundary": "Good for schema/context checks; survival/progression endpoints are target-mismatched.",
    },
    {
        "dataset_id": "metabric_cbioportal",
        "name": "METABRIC via cBioPortal",
        "access": "public_cBioPortal_terms",
        "best_use": "clinical/genomic subtype and outcome schema stress testing",
        "fit_score": 0.74,
        "mapped_fields": ["clinical", "genomic", "expression", "subtype", "outcome_context"],
        "missing_for_nlcare": ["monitoring cycles", "CBC", "symptoms", "imaging-response timeline"],
        "source_url": "https://www.cbioportal.org/",
        "readiness": "schema_mapping_bridge",
        "claim_boundary": "Not a substitute for clinician-reviewed temporal monitoring labels.",
    },
    {
        "dataset_id": "ispy2_tcia_like",
        "name": "I-SPY / TCIA-style neoadjuvant imaging-response data",
        "access": "public_or_application_dependent",
        "best_use": "pCR/imaging-response common-feature stress testing",
        "fit_score": 0.82,
        "mapped_fields": ["age", "tumor_size", "HR_HER2_context", "MRI_response_context", "pCR_label"],
        "missing_for_nlcare": ["same label definition", "patient-reported symptoms", "CBC monitoring"],
        "source_url": "https://www.cancerimagingarchive.net/",
        "readiness": "common_feature_stress_bridge",
        "claim_boundary": "pCR is not the same as NLCare synthetic response-pattern labels.",
    },
]


def build_external_dataset_bridge_v2(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    ranked = sorted(DATASETS, key=lambda row: row["fit_score"], reverse=True)
    payload = {
        "schema_version": "external_dataset_bridge_v2_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "headline_metric": "5 external/public bridges ranked; 0 clinical validation claims",
        "total_n": len(ranked),
        "pass_count": len(ranked),
        "fail_count": 0,
        "skipped_count": 0,
        "ranked_datasets": ranked,
        "highest_priority": [ranked[0]["dataset_id"], ranked[1]["dataset_id"]],
        "recommended_next_experiments": [
            "Map Duke MRI fields into the canonical imaging/treatment/follow-up schema.",
            "Run a strict common-feature stress test on GENIE BPC BrCa when access terms are satisfied.",
            "Keep TCGA/METABRIC survival or outcome fields marked target-mismatched for monitoring heads.",
            "Create a failure-case gallery before any promotion discussion.",
        ],
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "contamination_note": (
            "Dataset ranking is based on public documentation and existing schema-fit goals. "
            "Actual access, cleaning, and reviewer analysis are still future work."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = ["build_external_dataset_bridge_v2"]
