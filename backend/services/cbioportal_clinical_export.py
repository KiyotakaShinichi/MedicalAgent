from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import (
    DEFAULT_OUTPUT_PATH as DEFAULT_CANONICAL_SCHEMA_PATH,
    ROOT_DIR,
    build_canonical_oncology_schema,
    validate_canonical_rows,
)


CBIOPORTAL_API_BASE = "https://www.cbioportal.org/api"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_cbioportal_clinical_export.json"
DEFAULT_OUTPUT_DIR = "Data/external_bridge/cbioportal"
DEFAULT_COMBINED_CSV = "Data/external_bridge/cbioportal/canonical_cbioportal_breast_rows.csv"

CLAIM_BOUNDARY = (
    "cBioPortal clinical export maps public TCGA/METABRIC clinical attributes into the NLCare canonical schema "
    "for distribution and interoperability checks only. These rows are not NLCare longitudinal monitoring rows "
    "and do not validate treatment response, toxicity, recurrence, prognosis, or treatment decisions."
)


STUDY_CONFIGS: dict[str, dict[str, Any]] = {
    "brca_metabric": {
        "label": "METABRIC Breast Cancer",
        "source_dataset": "metabric_cbioportal",
        "patient_attributes": [
            "AGE_AT_DIAGNOSIS",
            "SEX",
            "ER_IHC",
            "HER2_SNP6",
            "CLAUDIN_SUBTYPE",
            "THREEGENE",
            "TUMOR_STAGE",
            "CHEMOTHERAPY",
            "HORMONE_THERAPY",
            "RADIO_THERAPY",
            "BREAST_SURGERY",
            "OS_STATUS",
            "OS_MONTHS",
            "RFS_STATUS",
            "RFS_MONTHS",
        ],
        "sample_attributes": [
            "ER_STATUS",
            "PR_STATUS",
            "HER2_STATUS",
            "GRADE",
            "TUMOR_SIZE",
            "MUTATION_COUNT",
            "TMB_NONSYNONYMOUS",
        ],
        "source_urls": [
            "https://www.cbioportal.org/study/summary?id=brca_metabric",
            "https://www.cbioportal.org/api/studies/brca_metabric?projection=SUMMARY",
        ],
    },
    "brca_tcga_pan_can_atlas_2018": {
        "label": "TCGA-BRCA PanCancer Atlas",
        "source_dataset": "tcga_brca_pan_can_atlas_2018",
        "patient_attributes": [
            "AGE",
            "SEX",
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "SUBTYPE",
            "HISTORY_NEOADJUVANT_TRTYN",
            "RADIATION_THERAPY",
            "OS_STATUS",
            "OS_MONTHS",
            "DFS_STATUS",
            "DFS_MONTHS",
            "PFS_STATUS",
            "PFS_MONTHS",
            "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
        ],
        "sample_attributes": [
            "CANCER_TYPE_DETAILED",
            "GRADE",
            "MUTATION_COUNT",
            "TMB_NONSYNONYMOUS",
            "FRACTION_GENOME_ALTERED",
        ],
        "source_urls": [
            "https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018",
            "https://www.cbioportal.org/api/studies/brca_tcga_pan_can_atlas_2018?projection=SUMMARY",
        ],
    },
}


def build_cbioportal_clinical_export(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    combined_csv: str = DEFAULT_COMBINED_CSV,
    live_fetch: bool = True,
    page_size: int = 100000,
    timeout_seconds: int = 45,
    fixture_records: dict[str, list[dict[str, Any]]] | None = None,
    schema_output_path: str = DEFAULT_CANONICAL_SCHEMA_PATH,
) -> dict[str, Any]:
    build_canonical_oncology_schema(output_path=schema_output_path)
    all_rows: list[dict[str, Any]] = []
    studies: dict[str, Any] = {}
    base_dir = _resolve(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for study_id, config in STUDY_CONFIGS.items():
        if fixture_records is not None:
            canonical_rows = [_record_to_canonical(study_id, config, record) for record in fixture_records.get(study_id, [])]
            fetch_status = "fixture"
            fetch_error = None
        elif live_fetch:
            try:
                records = _fetch_study_records(study_id, config, page_size=page_size, timeout_seconds=timeout_seconds)
                canonical_rows = [_record_to_canonical(study_id, config, record) for record in records]
                fetch_status = "downloaded"
                fetch_error = None
            except Exception as exc:  # noqa: BLE001
                canonical_rows = []
                fetch_status = "failed"
                fetch_error = str(exc)[:500]
        else:
            canonical_rows = []
            fetch_status = "skipped"
            fetch_error = None

        validation = validate_canonical_rows(canonical_rows)
        study_csv = base_dir / f"canonical_{config['source_dataset']}.csv"
        _write_csv(study_csv, canonical_rows)
        all_rows.extend(canonical_rows)
        studies[study_id] = {
            "label": config["label"],
            "source_dataset": config["source_dataset"],
            "status": "mapped" if canonical_rows and validation["status"] == "passed" else fetch_status,
            "fetch_status": fetch_status,
            "fetch_error": fetch_error,
            "row_count": len(canonical_rows),
            "canonical_csv": _display_path(study_csv),
            "validation": validation,
            "coverage": _coverage(canonical_rows),
            "source_urls": config["source_urls"],
        }

    combined_path = _resolve(combined_csv)
    _write_csv(combined_path, all_rows)
    combined_validation = validate_canonical_rows(all_rows)
    payload = {
        "schema_version": "cbioportal_clinical_export_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if all_rows and combined_validation["status"] == "passed" else "needs_attention",
        "studies": studies,
        "combined": {
            "row_count": len(all_rows),
            "canonical_csv": combined_csv,
            "validation": combined_validation,
            "coverage": _coverage(all_rows),
        },
        "not_supported_for_model_promotion": [
            "serial CBC timeline",
            "patient-reported symptom timeline",
            "treatment-cycle medication sequence",
            "imaging response timeline",
            "tumor-marker response trajectory",
            "NLCare response-score label",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    return payload


def _fetch_study_records(
    study_id: str,
    config: dict[str, Any],
    *,
    page_size: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    patient_data = _fetch_clinical_data(study_id, "PATIENT", page_size=page_size, timeout_seconds=timeout_seconds)
    sample_data = _fetch_clinical_data(study_id, "SAMPLE", page_size=page_size, timeout_seconds=timeout_seconds)
    patient_pivot = _pivot_clinical_data(patient_data, id_key="patientId")
    sample_pivot = _pivot_clinical_data(sample_data, id_key="patientId")
    ids = sorted(set(patient_pivot) | set(sample_pivot))
    records: list[dict[str, Any]] = []
    wanted = set(config["patient_attributes"]) | set(config["sample_attributes"])
    for patient_id in ids:
        merged = {"patient_id": patient_id}
        for source in (patient_pivot.get(patient_id, {}), sample_pivot.get(patient_id, {})):
            for key, value in source.items():
                if key in wanted and key not in merged:
                    merged[key] = value
        records.append(merged)
    return records


def _fetch_clinical_data(
    study_id: str,
    clinical_type: str,
    *,
    page_size: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    params = {
        "clinicalDataType": clinical_type,
        "projection": "SUMMARY",
        "pageSize": str(page_size),
        "pageNumber": "0",
    }
    url = f"{CBIOPORTAL_API_BASE}/studies/{study_id}/clinical-data?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _pivot_clinical_data(rows: list[dict[str, Any]], *, id_key: str) -> dict[str, dict[str, str]]:
    pivot: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        record_id = str(row.get(id_key) or "")
        attr = str(row.get("clinicalAttributeId") or "")
        if not record_id or not attr:
            continue
        pivot[record_id][attr] = str(row.get("value") or "")
    return dict(pivot)


def _record_to_canonical(study_id: str, config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    patient_id = str(record.get("patient_id") or record.get("PATIENT_ID") or record.get("patientId") or "")
    age = _first_float(record, ["AGE_AT_DIAGNOSIS", "AGE"])
    subtype = _first_text(record, ["CLAUDIN_SUBTYPE", "THREEGENE", "SUBTYPE", "CANCER_TYPE_DETAILED"])
    er_raw = _first_text(record, ["ER_IHC", "ER_STATUS"])
    pr_raw = _first_text(record, ["PR_STATUS"])
    her2_raw = _first_text(record, ["HER2_STATUS", "HER2_SNP6"])
    treatment_modalities = _treatment_modalities(record)
    return {
        "source_dataset": config["source_dataset"],
        "source_record_id": patient_id,
        "patient_id": f"CBIO:{study_id}:{patient_id}",
        "timepoint_index": 0,
        "age": age,
        "sex": _normalize_sex(_first_text(record, ["SEX"])),
        "stage": _first_text(record, ["TUMOR_STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE"]),
        "molecular_subtype": subtype or "unknown",
        "er_status": _normalize_receptor(er_raw),
        "pr_status": _normalize_receptor(pr_raw),
        "her2_status": _normalize_her2(her2_raw),
        "ki67_percent": "",
        "genetic_context_available": bool(_first_text(record, ["MUTATION_COUNT", "TMB_NONSYNONYMOUS", "FRACTION_GENOME_ALTERED"])),
        "genetic_variant_classification": _genomic_context(record),
        "treatment_phase": "unknown",
        "treatment_modalities": treatment_modalities,
        "treatment_combination_pattern": "+".join(treatment_modalities) if treatment_modalities else "unknown",
        "regimen_text": _regimen_text(record),
        "cbc_available": False,
        "symptoms_available": False,
        "imaging_available": False,
        "imaging_modality": "unknown",
        "imaging_features": {},
        "tumor_marker_available": False,
        "tumor_marker_context_only": True,
        "outcome_label_name": _outcome_label_name(record),
        "outcome_label_value": _outcome_label_value(record),
        "source_urls": config["source_urls"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _treatment_modalities(record: dict[str, Any]) -> list[str]:
    modalities: list[str] = []
    if _yesish(record.get("CHEMOTHERAPY")):
        modalities.append("chemotherapy")
    if _yesish(record.get("HORMONE_THERAPY")):
        modalities.append("endocrine")
    if _yesish(record.get("RADIO_THERAPY")) or _yesish(record.get("RADIATION_THERAPY")):
        modalities.append("radiation")
    if _truthy_text(record.get("BREAST_SURGERY")):
        modalities.append("surgery")
    if _truthy_text(record.get("HISTORY_NEOADJUVANT_TRTYN")):
        modalities.append("neoadjuvant_context")
    return modalities


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subtype = Counter(str(row.get("molecular_subtype") or "unknown") for row in rows)
    stage = Counter(str(row.get("stage") or "unknown") for row in rows)
    modalities = Counter()
    known = Counter()
    for row in rows:
        for item in row.get("treatment_modalities") or []:
            modalities[item] += 1
        for key in ("er_status", "pr_status", "her2_status", "stage", "molecular_subtype"):
            if row.get(key) not in {None, "", "unknown"}:
                known[key] += 1
    return {
        "subtype_counts": dict(subtype.most_common(12)),
        "stage_counts": dict(stage.most_common(12)),
        "treatment_modality_counts": dict(modalities.most_common()),
        "known_field_counts": dict(known),
        "roles_supported": {
            "external_distribution_alignment": bool(rows),
            "biomarker_schema_mapping": bool(known),
            "full_oncotrack_temporal_validation": False,
            "treatment_response_training": False,
            "tumor_marker_response_training": False,
        },
    }


def _first_text(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value not in {None, ""}:
            return str(value)
    return ""


def _first_float(record: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = record.get(key)
        if value not in {None, ""}:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _normalize_sex(value: str) -> str:
    lower = value.lower()
    if lower.startswith("f"):
        return "female"
    if lower.startswith("m"):
        return "male"
    return "unknown"


def _normalize_receptor(value: str) -> str:
    lower = value.lower()
    if "pos" in lower or lower in {"positive", "yes"}:
        return "positive"
    if "neg" in lower or lower in {"negative", "no"}:
        return "negative"
    return "unknown"


def _normalize_her2(value: str) -> str:
    lower = value.lower()
    if "pos" in lower or "amplified" in lower:
        return "positive"
    if "neg" in lower:
        return "negative"
    if "equiv" in lower:
        return "equivocal"
    return "unknown"


def _yesish(value: Any) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "received"}


def _truthy_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text and text not in {"no", "none", "unknown", "nan", "na", "not reported"})


def _genomic_context(record: dict[str, Any]) -> str:
    fields = []
    for key in ("MUTATION_COUNT", "TMB_NONSYNONYMOUS", "FRACTION_GENOME_ALTERED"):
        if record.get(key) not in {None, ""}:
            fields.append(f"{key}={record[key]}")
    return "; ".join(fields) if fields else "not_reported"


def _regimen_text(record: dict[str, Any]) -> str:
    parts = []
    for key in ("CHEMOTHERAPY", "HORMONE_THERAPY", "RADIO_THERAPY", "RADIATION_THERAPY", "HISTORY_NEOADJUVANT_TRTYN", "BREAST_SURGERY"):
        if record.get(key) not in {None, ""}:
            parts.append(f"{key}: {record[key]}")
    return "; ".join(parts)


def _outcome_label_name(record: dict[str, Any]) -> str:
    for key in ("PFS_STATUS", "DFS_STATUS", "RFS_STATUS", "OS_STATUS", "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"):
        if record.get(key) not in {None, ""}:
            return key
    return ""


def _outcome_label_value(record: dict[str, Any]) -> str:
    key = _outcome_label_name(record)
    return str(record.get(key) or "") if key else ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)
