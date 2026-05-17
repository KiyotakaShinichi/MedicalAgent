from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.artifact_manifest import build_artifact_manifest


DEFAULT_OUTPUT_PATH = "Data/mle_monitoring/cbioportal_biomarker_schema_mapping.json"
CBIOPORTAL_API_BASE = "https://www.cbioportal.org/api"


STUDIES = {
    "tcga_brca_pan_can_atlas_2018": {
        "study_id": "brca_tcga_pan_can_atlas_2018",
        "label": "TCGA-BRCA PanCancer Atlas",
        "role": "external genomic/subtype schema check",
    },
    "metabric": {
        "study_id": "brca_metabric",
        "label": "METABRIC Breast Cancer",
        "role": "external biomarker/subtype/outcome schema check",
    },
}

FIELD_GROUPS = {
    "er_status": ["ER_STATUS", "ER_STATUS_BY_IHC", "ER_IHC", "ESTROGEN"],
    "pr_status": ["PR_STATUS", "PR_STATUS_BY_IHC", "PR_IHC", "PROGESTERONE"],
    "her2_status": ["HER2_STATUS", "HER2_STATUS_BY_IHC", "HER2", "ERBB2"],
    "subtype": ["SUBTYPE", "PAM50", "CLAUDIN", "MOLECULAR_SUBTYPE"],
    "stage": ["STAGE", "AJCC", "TUMOR_STAGE"],
    "grade": ["GRADE", "NEOPLASM_HISTOLOGIC_GRADE"],
    "survival": ["OS_", "OVERALL_SURVIVAL", "DSS", "DFS", "RFS", "PFS"],
    "metastasis_recurrence": ["METAST", "RECURREN", "PROGRESSION", "DISTANT"],
    "genomic": ["MUTATION", "COPY", "CNA", "FRACTION", "ANEUPLOIDY", "TMB", "MSI"],
}


def build_cbioportal_biomarker_schema_mapping(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    timeout_seconds: int = 20,
    live_fetch: bool = True,
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for key, spec in STUDIES.items():
        datasets[key] = _inspect_study(spec, timeout_seconds=timeout_seconds, live_fetch=live_fetch)

    ready_count = sum(1 for dataset in datasets.values() if dataset.get("status") in {"mapped", "partial"})
    report: dict[str, Any] = {
        **build_artifact_manifest(
            dataset_paths={"cbioportal_output": output_path},
            seed=42,
        ),
        "schema_version": "cbioportal_biomarker_schema_mapping_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if ready_count else "unavailable",
        "source": {
            "provider": "cBioPortal public API",
            "api_base": CBIOPORTAL_API_BASE,
            "studies": [spec["study_id"] for spec in STUDIES.values()],
        },
        "datasets": datasets,
        "mapped_dataset_count": ready_count,
        "recommended_use": [
            "Use TCGA-BRCA/METABRIC as external schema and distribution checks for receptor/subtype/genomic fields.",
            "Do not treat survival endpoints as pCR or treatment-response labels.",
            "Use these sources to stress-test feature mapping, missingness, subtype drift, and calibration transfer.",
        ],
        "claim_boundary": (
            "This artifact maps public cBioPortal schema availability. It is not a clinical validation result and "
            "does not imply that OncoTrack can predict outcomes from TCGA/METABRIC without task-specific modeling."
        ),
    }
    report["mapping_hash"] = _stable_hash(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_cbioportal_biomarker_schema_mapping(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return build_cbioportal_biomarker_schema_mapping(output_path=output_path, live_fetch=False)


def _inspect_study(spec: dict[str, str], *, timeout_seconds: int, live_fetch: bool) -> dict[str, Any]:
    study_id = spec["study_id"]
    if not live_fetch:
        return {
            "status": "not_fetched",
            "study_id": study_id,
            "label": spec["label"],
            "role": spec["role"],
            "reason": "Live fetch disabled; run the builder script with network enabled to inspect cBioPortal.",
            "mapped_groups": {},
        }
    try:
        attributes = _fetch_clinical_attributes(study_id, timeout_seconds=timeout_seconds)
    except Exception as exc:
        return {
            "status": "unavailable",
            "study_id": study_id,
            "label": spec["label"],
            "role": spec["role"],
            "reason": str(exc)[:300],
            "mapped_groups": {},
        }

    mapped_groups = _map_attributes(attributes)
    core_hits = sum(1 for name in ["er_status", "pr_status", "her2_status", "subtype"] if mapped_groups.get(name))
    status = "mapped" if core_hits >= 3 else "partial" if core_hits else "needs_attention"
    return {
        "status": status,
        "study_id": study_id,
        "label": spec["label"],
        "role": spec["role"],
        "clinical_attribute_count": len(attributes),
        "mapped_groups": mapped_groups,
        "core_biomarker_group_hits": core_hits,
        "next_action": _next_action(status, study_id),
    }


def _fetch_clinical_attributes(study_id: str, *, timeout_seconds: int) -> list[dict[str, Any]]:
    url = f"{CBIOPORTAL_API_BASE}/studies/{study_id}/clinical-attributes?projection=SUMMARY"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"cBioPortal HTTP {exc.code} for {study_id}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cBioPortal unavailable for {study_id}: {exc.reason}") from exc


def _map_attributes(attributes: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    mapped: dict[str, list[dict[str, str]]] = {group: [] for group in FIELD_GROUPS}
    for attribute in attributes:
        attr_id = str(attribute.get("clinicalAttributeId") or attribute.get("attributeId") or "").upper()
        display_name = str(attribute.get("displayName") or attribute.get("description") or "").upper()
        material = f"{attr_id} {display_name}"
        for group, tokens in FIELD_GROUPS.items():
            if any(_token_matches(material, token) for token in tokens):
                mapped[group].append({
                    "id": str(attribute.get("clinicalAttributeId") or attribute.get("attributeId") or ""),
                    "display_name": str(attribute.get("displayName") or ""),
                    "description": str(attribute.get("description") or "")[:180],
                    "datatype": str(attribute.get("datatype") or ""),
                })
    return {group: rows[:10] for group, rows in mapped.items() if rows}


def _token_matches(material: str, token: str) -> bool:
    token = token.upper()
    if token.endswith("_"):
        return re.search(rf"(?<![A-Z0-9]){re.escape(token)}", material) is not None
    if len(token) <= 3:
        return re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", material) is not None
    if "_" in token:
        return re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", material) is not None
    return token in material


def _next_action(status: str, study_id: str) -> str:
    if status == "mapped":
        return f"Export clinical data for {study_id}, map receptor/subtype/outcome fields, then run an external distribution check."
    if status == "partial":
        return f"Use mapped fields from {study_id} as schema checks only; inspect missing receptor/subtype columns manually."
    return f"Do not use {study_id} for modeling until the clinical schema is mapped and documented."


def _stable_hash(payload: dict[str, Any]) -> str:
    material = json.dumps(
        {key: value for key, value in payload.items() if key not in {"generated_at", "mapping_hash"}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
