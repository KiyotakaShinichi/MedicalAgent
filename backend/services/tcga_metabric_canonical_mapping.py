from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.cbioportal_biomarker_mapper import (
    DEFAULT_OUTPUT_PATH as CBIOPORTAL_MAPPING_PATH,
    build_cbioportal_biomarker_schema_mapping,
)
from backend.services.oncology_canonical_schema import (
    DEFAULT_OUTPUT_PATH as DEFAULT_CANONICAL_SCHEMA_PATH,
    ROOT_DIR,
    build_canonical_oncology_schema,
)


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_tcga_metabric_canonical_mapping.json"
DEFAULT_MAPPING_PATH = "Data/external_bridge/tcga_metabric_canonical_mapping.json"

CLAIM_BOUNDARY = (
    "TCGA/METABRIC canonical mapping is external schema and distribution readiness only. "
    "It is not validation of NLCare's temporal monitoring model and must not be used "
    "to claim treatment-response, recurrence, survival, or genetic-risk prediction."
)


def build_tcga_metabric_canonical_mapping(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    mapping_path: str = DEFAULT_MAPPING_PATH,
    source_mapping_path: str = CBIOPORTAL_MAPPING_PATH,
    live_fetch: bool = False,
    schema_output_path: str = DEFAULT_CANONICAL_SCHEMA_PATH,
) -> dict[str, Any]:
    build_canonical_oncology_schema(output_path=schema_output_path)
    cbio = _load_or_build_cbioportal_mapping(source_mapping_path=source_mapping_path, live_fetch=live_fetch)
    datasets = {
        "tcga_brca_pan_can_atlas_2018": _dataset_mapping(
            cbio,
            source_key="tcga_brca_pan_can_atlas_2018",
            canonical_dataset_id="tcga_brca_pan_can_atlas_2018",
            role="external genomic/subtype distribution check",
        ),
        "metabric": _dataset_mapping(
            cbio,
            source_key="metabric",
            canonical_dataset_id="brca_metabric",
            role="external biomarker/subtype/outcome schema check",
        ),
    }
    mapped = sum(1 for item in datasets.values() if item["status"] in {"mapped", "partial", "schema_available"})
    payload = {
        "schema_version": "tcga_metabric_canonical_mapping_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if mapped >= 2 else "acceptable" if mapped else "needs_attention",
        "canonical_schema_path": "Data/external_bridge/canonical_oncology_schema.json",
        "source_mapping_path": source_mapping_path,
        "datasets": datasets,
        "mapped_dataset_count": mapped,
        "strict_common_feature_set": [
            "age",
            "sex",
            "stage",
            "molecular_subtype",
            "er_status",
            "pr_status",
            "her2_status",
        ],
        "not_supported_for_oncotrack_training": [
            "serial CBC trends",
            "patient-reported symptom timeline",
            "medication-by-cycle records",
            "MRI/CT/ultrasound monitoring sequence",
            "tumor-marker treatment-response trajectory",
            "NLCare response-score label",
        ],
        "recommended_use": [
            "canonical schema sanity check",
            "biomarker/subtype distribution alignment",
            "future common-feature A/B testing after export",
            "never direct patient-facing treatment-response prediction",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(mapping_path), payload)
    _write_json(_resolve(output_path), payload)
    return payload


def _load_or_build_cbioportal_mapping(*, source_mapping_path: str, live_fetch: bool) -> dict[str, Any]:
    path = _resolve(source_mapping_path)
    if path.exists() and not live_fetch:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return build_cbioportal_biomarker_schema_mapping(output_path=source_mapping_path, live_fetch=live_fetch)


def _dataset_mapping(cbio: dict[str, Any], *, source_key: str, canonical_dataset_id: str, role: str) -> dict[str, Any]:
    source = (cbio.get("datasets") or {}).get(source_key, {})
    mapped_groups = source.get("mapped_groups") or {}
    field_map = {
        "source_dataset": canonical_dataset_id,
        "patient_id": "case/sample identifier after export",
        "age": _first_mapped(mapped_groups, ["age"], fallback="age field may require clinical export inspection"),
        "sex": _first_mapped(mapped_groups, ["sex"], fallback="demographic field may require clinical export inspection"),
        "stage": _first_mapped(mapped_groups, ["stage"]),
        "molecular_subtype": _first_mapped(mapped_groups, ["subtype"]),
        "er_status": _first_mapped(mapped_groups, ["er_status"]),
        "pr_status": _first_mapped(mapped_groups, ["pr_status"]),
        "her2_status": _first_mapped(mapped_groups, ["her2_status"]),
        "genetic_variant_classification": _first_mapped(mapped_groups, ["genomic"]),
        "outcome_label_name": _first_mapped(mapped_groups, ["survival", "metastasis_recurrence"]),
    }
    available = [key for key, value in field_map.items() if isinstance(value, dict)]
    missing_core = [
        key for key in ("er_status", "pr_status", "her2_status", "molecular_subtype")
        if key not in available
    ]
    source_status = source.get("status") or "not_fetched"
    status = "schema_available" if source_status in {"mapped", "partial"} else source_status
    return {
        "status": status,
        "source_status": source_status,
        "study_id": source.get("study_id") or canonical_dataset_id,
        "label": source.get("label") or canonical_dataset_id,
        "role": role,
        "canonical_field_map": field_map,
        "available_canonical_fields": available,
        "missing_core_fields": missing_core,
        "target_mismatch": (
            "Survival/progression fields, when present, are not equivalent to NLCare's synthetic "
            "response classification, response-score regression, or toxicity-review labels."
        ),
        "next_action": source.get("next_action") or "Export permitted clinical attributes, then map values into canonical rows.",
    }


def _first_mapped(
    mapped_groups: dict[str, list[dict[str, str]]],
    groups: list[str],
    *,
    fallback: str | None = None,
) -> dict[str, str] | str | None:
    for group in groups:
        entries = mapped_groups.get(group) or []
        if entries:
            return entries[0]
    return fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
