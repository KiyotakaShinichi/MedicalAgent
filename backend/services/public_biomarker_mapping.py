from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.public_biomarker_datasets import load_public_biomarker_dataset_manifest


DEFAULT_BREASTDCEDL_FEATURES = "Data/breastdcedl_spy1_features.csv"
DEFAULT_OUTPUT_PATH = "Data/mle_monitoring/public_biomarker_mapping_readiness.json"


def build_public_biomarker_mapping_readiness(
    breastdcedl_features_path: str = DEFAULT_BREASTDCEDL_FEATURES,
    output_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = 42,
) -> dict[str, Any]:
    manifest = load_public_biomarker_dataset_manifest()
    breastdcedl = _inspect_breastdcedl(breastdcedl_features_path)
    datasets = {
        "breastdcedl": breastdcedl,
        "tcga_brca_cbioportal": {
            "status": "schema_candidate",
            "mapped_now": False,
            "predictors_to_map": ["subtype", "ER/PR/HER2-derived fields", "stage", "genomic alterations"],
            "target_to_map": "survival/progression endpoint, not direct pCR",
            "next_action": "Use cBioPortal/GDC exports to build a compatible subtype/genomic external validation table.",
        },
        "metabric_cbioportal": {
            "status": "schema_candidate",
            "mapped_now": False,
            "predictors_to_map": ["ER", "PR", "HER2", "PAM50/subtype", "grade", "stage", "expression/copy-number"],
            "target_to_map": "survival/outcome endpoint",
            "next_action": "Create a METABRIC subtype/genomic feature table for distribution and robustness checks.",
        },
        "aacr_genie_bpc_brca": {
            "status": "future_access_candidate",
            "mapped_now": False,
            "predictors_to_map": ["clinical-grade NGS", "ER/PR/HER2", "Oncotype DX", "multigene signatures"],
            "target_to_map": "real-world response/PFS/OS",
            "next_action": "Use only after access/terms are handled; best future real-world biomarker benchmark.",
        },
        "nci_edrn_breast_reference_set": {
            "status": "monitoring_context_only",
            "mapped_now": False,
            "predictors_to_map": ["CA15-3", "CEA-family/CEACAM5", "CA125", "CRP", "EGFR", "ERBB2"],
            "target_to_map": "reference-set labels, not treatment response",
            "next_action": "Use for tumor-marker limitations and distribution priors, not for autonomous prediction.",
        },
    }

    report = {
        **build_artifact_manifest(seed=seed, dataset_paths={"breastdcedl_features": breastdcedl_features_path}),
        "schema_version": "public_biomarker_mapping_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if breastdcedl.get("status") == "mapped" else "needs_attention",
        "source_manifest_hash": manifest.get("manifest_hash"),
        "datasets": datasets,
        "three_stage_ablation_plan": {
            "clinical_timeline_only": "CBC/labs, symptoms, treatment-cycle context, stage, regimen; excludes biomarkers and imaging response.",
            "clinical_plus_biomarkers": "Adds ER/PR/HER2/Ki-67/genetic-readiness and tumor-marker trend features; excludes imaging response.",
            "clinical_plus_biomarkers_plus_imaging": "Adds MRI/CT/ultrasound response features for the full candidate model.",
        },
        "tumor_marker_boundary": (
            "CA 15-3, CA 27.29, CEA, CA125, and related circulating markers are treated as contextual monitoring "
            "features only. They are not standalone recurrence/progression labels and cannot drive treatment advice."
        ),
        "recommended_next_order": [
            "Use BreastDCEDL HR/HER2/pCR plus MRI features as the first mapped external benchmark.",
            "Add TCGA-BRCA/METABRIC subtype/genomic schema checks as non-longitudinal external validation.",
            "Keep AACR GENIE BPC as the future highest-value benchmark after access/terms are handled.",
            "Keep serum tumor-marker sources as cautionary/supportive monitoring evidence.",
            "Run the three-stage ablation and promote no model until external/temporal checks support it.",
        ],
        "claim_boundary": (
            "This report maps public sources and ablation design. It is not clinical validation and does not prove "
            "biomarker or tumor-marker predictive utility."
        ),
    }
    report["mapping_hash"] = _stable_hash(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_public_biomarker_mapping_readiness(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return build_public_biomarker_mapping_readiness(output_path=output_path)


def _inspect_breastdcedl(features_path: str) -> dict[str, Any]:
    path = Path(features_path)
    if not path.exists():
        return {
            "status": "missing",
            "mapped_now": False,
            "path": features_path,
            "reason": "BreastDCEDL feature CSV not found.",
        }
    rows = pd.read_csv(path)
    columns = set(rows.columns)
    subtype_counts = rows.get("molecular_subtype", pd.Series(dtype=str)).fillna("unknown").value_counts().to_dict()
    required = {"patient_id", "pcr_label", "age", "molecular_subtype"}
    imaging_columns = sorted(column for column in columns if any(token in column for token in ["enhancement", "washout", "diameter", "voxel"]))
    return {
        "status": "mapped" if required.issubset(columns) else "partial",
        "mapped_now": required.issubset(columns),
        "path": features_path,
        "rows": int(len(rows)),
        "patients": int(rows["patient_id"].nunique()) if "patient_id" in rows else 0,
        "target": "pcr_label" if "pcr_label" in rows else None,
        "direct_predictors": sorted(required - {"patient_id", "pcr_label"}),
        "derived_predictors": ["hr_positive_proxy", "her2_positive_proxy", "triple_negative_proxy"],
        "imaging_predictors": imaging_columns,
        "subtype_counts": {str(key): int(value) for key, value in subtype_counts.items()},
        "next_action": "Use this as the first public HR/HER2/pCR plus imaging benchmark.",
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    material = json.dumps(
        {key: value for key, value in payload.items() if key not in {"generated_at", "mapping_hash"}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
