from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import (
    ROOT_DIR,
    build_canonical_oncology_schema,
    validate_canonical_rows,
)


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_priority_dataset_bridge.json"
DEFAULT_DOC_PATH = "docs/priority_dataset_bridge.md"
DEFAULT_TEMPLATE_DIR = "Data/external_bridge/priority_dataset_templates"
DEFAULT_GENIE_CANONICAL_CSV = "Data/external_bridge/canonical_genie_bpc_brca.csv"
DEFAULT_DUKE_CANONICAL_CSV = "Data/external_bridge/canonical_duke_breast_mri.csv"

GENIE_SOURCE_URL = "https://www.aacr.org/professionals/research/aacr-project-genie/bpc/early-onset-brca/"
DUKE_SOURCE_URL = "https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/"

CLAIM_BOUNDARY = (
    "Priority dataset bridge artifacts map external dataset fields into the NLCare canonical schema "
    "for interoperability, stress testing, and future access readiness only. They do not establish "
    "clinical validation, treatment superiority, survival prediction, genetic-risk interpretation, or "
    "patient-facing treatment recommendations."
)


GENIE_FIELD_CONTRACT: list[dict[str, Any]] = [
    {
        "canonical_field": "patient_id",
        "role": "identifier",
        "aliases": ["PATIENT_ID", "patient_id", "record_id", "GENIE_PATIENT_ID"],
        "required_for_mapping": True,
        "notes": "Namespace before storing locally; never expose as real patient identity.",
    },
    {
        "canonical_field": "age",
        "role": "demographics",
        "aliases": ["AGE_AT_SEQ_REPORT", "AGE_AT_DIAGNOSIS", "age", "age_at_diagnosis"],
        "required_for_mapping": False,
        "notes": "Age may be binned or shifted depending on release terms.",
    },
    {
        "canonical_field": "stage",
        "role": "clinical context",
        "aliases": ["STAGE_AT_DIAGNOSIS", "AJCC_STAGE", "stage", "stage_at_dx"],
        "required_for_mapping": False,
        "notes": "Use source-reported stage only; do not infer.",
    },
    {
        "canonical_field": "er_status/pr_status/her2_status",
        "role": "biomarker context",
        "aliases": ["ER_STATUS", "PR_STATUS", "HER2_STATUS", "ER", "PR", "HER2"],
        "required_for_mapping": False,
        "notes": "Organize receptor context; never select therapy from it.",
    },
    {
        "canonical_field": "genetic_variant_classification",
        "role": "genomic context",
        "aliases": ["HUGO_SYMBOL", "GENE", "ALTERATION", "VARIANT_CLASSIFICATION", "MUTATION_STATUS"],
        "required_for_mapping": False,
        "notes": "Somatic/genomic context only unless a source explicitly reports germline status.",
    },
    {
        "canonical_field": "treatment_modalities/treatment_combination_pattern",
        "role": "treatment-history context",
        "aliases": ["REGIMEN", "DRUG_NAME", "SYSTEMIC_THERAPY", "CANCER_DIRECTED_REGIMEN", "TREATMENT_TYPE"],
        "required_for_mapping": False,
        "notes": "Map to controlled modality buckets; do not recommend regimens.",
    },
    {
        "canonical_field": "outcome_label_name/outcome_label_value",
        "role": "external outcome context",
        "aliases": ["BEST_RESPONSE", "PFS_STATUS", "OS_STATUS", "RW_RESPONSE", "response"],
        "required_for_mapping": False,
        "notes": "Outcome semantics are not equivalent to NLCare synthetic labels.",
    },
]


DUKE_FIELD_CONTRACT: list[dict[str, Any]] = [
    {
        "canonical_field": "patient_id",
        "role": "identifier",
        "aliases": ["Patient ID", "patient_id", "TCIA Patient ID", "Case ID", "ID"],
        "required_for_mapping": True,
        "notes": "Namespace before storing locally.",
    },
    {
        "canonical_field": "age",
        "role": "demographics",
        "aliases": ["Age", "age", "Age at MRI", "age_at_mri"],
        "required_for_mapping": False,
        "notes": "Baseline/index age when available.",
    },
    {
        "canonical_field": "er_status/pr_status/her2_status/molecular_subtype",
        "role": "pathology and receptor context",
        "aliases": ["ER", "PR", "HER2", "Mol Subtype", "Molecular subtype", "Subtype"],
        "required_for_mapping": False,
        "notes": "Organize pathology context only.",
    },
    {
        "canonical_field": "imaging_features",
        "role": "MRI feature bridge",
        "aliases": ["Tumor Size", "DCE", "washout", "SER", "BPE", "enhancement", "lesion"],
        "required_for_mapping": False,
        "notes": "Use numeric/image-derived features as external stress inputs, not diagnosis.",
    },
    {
        "canonical_field": "treatment_modalities/treatment_combination_pattern",
        "role": "treatment context",
        "aliases": ["NAC", "Neoadjuvant Chemotherapy", "Chemotherapy", "Radiation Therapy", "Endocrine Therapy"],
        "required_for_mapping": False,
        "notes": "Treatment context can support A/B stress tests, not treatment decisions.",
    },
    {
        "canonical_field": "outcome_label_name/outcome_label_value",
        "role": "response/follow-up context",
        "aliases": ["pCR", "Pathologic complete response", "Pathologic Response", "Recurrence", "Follow-up"],
        "required_for_mapping": False,
        "notes": "pCR/recurrence/follow-up labels are external endpoints, not NLCare clinical validation.",
    },
]


def build_priority_dataset_bridge(
    *,
    genie_csv: str | None = None,
    duke_csv: str | None = None,
    output_path: str = DEFAULT_OUTPUT_PATH,
    doc_path: str = DEFAULT_DOC_PATH,
    template_dir: str = DEFAULT_TEMPLATE_DIR,
    genie_canonical_csv: str = DEFAULT_GENIE_CANONICAL_CSV,
    duke_canonical_csv: str = DEFAULT_DUKE_CANONICAL_CSV,
) -> dict[str, Any]:
    build_canonical_oncology_schema()
    templates = _write_templates(_resolve(template_dir))

    genie_rows = _read_csv(_resolve(genie_csv)) if genie_csv else []
    duke_rows = _read_csv(_resolve(duke_csv)) if duke_csv else []
    genie_canonical = [_genie_to_canonical(row) for row in genie_rows]
    duke_canonical = [_duke_to_canonical(row) for row in duke_rows]

    genie_report = _dataset_report(
        dataset_id="genie_bpc_brca",
        name="AACR GENIE BPC Breast Cancer v1.0-public",
        source_url=GENIE_SOURCE_URL,
        access_status="public/access-controlled workflow; local CSV not bundled",
        canonical_rows=genie_canonical,
        canonical_csv_path=genie_canonical_csv,
        field_contract=GENIE_FIELD_CONTRACT,
        supported_roles={
            "treatment_history_bridge": True,
            "genomic_context_bridge": True,
            "mri_image_bridge": False,
            "cbc_symptom_timeline": False,
        },
    )
    duke_report = _dataset_report(
        dataset_id="duke_breast_mri",
        name="Duke Breast Cancer MRI / TCIA",
        source_url=DUKE_SOURCE_URL,
        access_status="public TCIA collection; large image/metadata download required",
        canonical_rows=duke_canonical,
        canonical_csv_path=duke_canonical_csv,
        field_contract=DUKE_FIELD_CONTRACT,
        supported_roles={
            "treatment_history_bridge": True,
            "genomic_context_bridge": False,
            "mri_image_bridge": True,
            "cbc_symptom_timeline": False,
        },
    )

    if genie_canonical:
        _write_csv(_resolve(genie_canonical_csv), genie_canonical)
    if duke_canonical:
        _write_csv(_resolve(duke_canonical_csv), duke_canonical)

    payload: dict[str, Any] = {
        "schema_version": "priority_dataset_bridge_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _overall_status([genie_report, duke_report]),
        "canonical_schema_path": "Data/external_bridge/canonical_oncology_schema.json",
        "template_paths": templates,
        "datasets": {
            "genie_bpc_brca": genie_report,
            "duke_breast_mri": duke_report,
        },
        "summary": _summary([genie_report, duke_report]),
        "next_actions": [
            "Download/export permitted GENIE BPC BRCA tables, then run this bridge with --genie-csv.",
            "Download Duke Breast MRI clinical-and-other-features metadata, then run this bridge with --duke-csv.",
            "Use mapped rows for external stress tests and schema coverage only; keep production models monitor-only.",
            "Do not mix outcome labels across datasets unless endpoint semantics are explicitly documented.",
        ],
        "blocked_claims": [
            "real-world clinical validation",
            "treatment recommendation or treatment superiority",
            "genetic-risk diagnosis or inherited-risk prediction",
            "tumor-marker recurrence conclusion",
            "survival or prognosis estimate",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_markdown(_resolve(doc_path), payload)
    return payload


def _dataset_report(
    *,
    dataset_id: str,
    name: str,
    source_url: str,
    access_status: str,
    canonical_rows: list[dict[str, Any]],
    canonical_csv_path: str,
    field_contract: list[dict[str, Any]],
    supported_roles: dict[str, bool],
) -> dict[str, Any]:
    validation = validate_canonical_rows(canonical_rows) if canonical_rows else {
        "status": "not_run_no_local_rows",
        "row_count": 0,
        "issue_count": 0,
        "issues": [],
    }
    status = "mapped" if canonical_rows and validation["status"] == "passed" else "ready_for_mapping"
    coverage = _coverage(canonical_rows)
    return {
        "status": status,
        "dataset_id": dataset_id,
        "name": name,
        "source_url": source_url,
        "access_status": access_status,
        "row_count": len(canonical_rows),
        "validation": validation,
        "canonical_csv_path": canonical_csv_path if canonical_rows else None,
        "field_contract": field_contract,
        "coverage": coverage,
        "supported_roles": supported_roles,
        "not_supported": [
            "NLCare-style serial CBC timeline",
            "patient-reported symptom timeline",
            "patient-facing treatment choice",
            "clinical validation of treatment response",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _genie_to_canonical(row: dict[str, str]) -> dict[str, Any]:
    record_id = _pick(row, ["PATIENT_ID", "patient_id", "record_id", "GENIE_PATIENT_ID"]) or "unknown"
    regimen = _pick(row, ["REGIMEN", "DRUG_NAME", "SYSTEMIC_THERAPY", "CANCER_DIRECTED_REGIMEN", "TREATMENT_TYPE"])
    modalities = _modalities_from_text(regimen)
    gene = _pick(row, ["HUGO_SYMBOL", "GENE"])
    variant = _pick(row, ["VARIANT_CLASSIFICATION", "MUTATION_STATUS", "ALTERATION"])
    outcome_value = _pick(row, ["BEST_RESPONSE", "RW_RESPONSE", "PFS_STATUS", "OS_STATUS", "response"])
    return {
        "source_dataset": "genie_bpc_brca",
        "source_record_id": str(record_id),
        "patient_id": f"GENIE_BPC_BRCA:{record_id}",
        "timepoint_index": _to_int_like(_pick(row, ["LINE_NUMBER", "treatment_line", "timepoint_index"])) or 0,
        "age": _to_float(_pick(row, ["AGE_AT_SEQ_REPORT", "AGE_AT_DIAGNOSIS", "age", "age_at_diagnosis"])),
        "sex": _normalize_sex(_pick(row, ["SEX", "GENDER", "sex", "gender"])),
        "stage": _pick(row, ["STAGE_AT_DIAGNOSIS", "AJCC_STAGE", "stage", "stage_at_dx"]) or "unknown",
        "molecular_subtype": _pick(row, ["CANCER_TYPE_DETAILED", "ONCOTREE_CODE", "CANCER_TYPE", "molecular_subtype"]) or "unknown",
        "er_status": _normalize_receptor(_pick(row, ["ER_STATUS", "ER", "er_status"])),
        "pr_status": _normalize_receptor(_pick(row, ["PR_STATUS", "PR", "pr_status"])),
        "her2_status": _normalize_her2(_pick(row, ["HER2_STATUS", "HER2", "her2_status"])),
        "ki67_percent": _to_float(_pick(row, ["KI67", "KI67_PERCENT", "ki67_percent"])),
        "genetic_context_available": bool(gene or variant),
        "genetic_variant_classification": _join_present([gene, variant]) or "not_reported",
        "treatment_phase": _normalize_phase(_pick(row, ["TREATMENT_SETTING", "SETTING", "treatment_phase"])),
        "treatment_modalities": modalities,
        "treatment_combination_pattern": _combination_pattern(modalities),
        "regimen_text": regimen or "not_reported",
        "cbc_available": False,
        "symptoms_available": False,
        "imaging_available": False,
        "imaging_modality": "unknown",
        "imaging_features": {},
        "tumor_marker_available": False,
        "tumor_marker_context_only": True,
        "outcome_label_name": _outcome_name(outcome_value),
        "outcome_label_value": outcome_value or "",
        "source_urls": [GENIE_SOURCE_URL],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _duke_to_canonical(row: dict[str, str]) -> dict[str, Any]:
    record_id = _pick(row, ["Patient ID", "patient_id", "TCIA Patient ID", "Case ID", "ID"]) or "unknown"
    treatment_text = _join_present([
        _pick(row, ["NAC", "Neoadjuvant Chemotherapy", "Chemotherapy"]),
        _pick(row, ["Radiation Therapy", "Radiation"]),
        _pick(row, ["Endocrine Therapy", "Hormone Therapy"]),
        _pick(row, ["Targeted Therapy", "HER2 therapy"]),
    ])
    modalities = _modalities_from_text(treatment_text)
    imaging_features = _duke_imaging_features(row)
    pcr = _pick(row, ["pCR", "Pathologic complete response", "Pathologic Response", "pathologic_response"])
    recurrence = _pick(row, ["Recurrence", "recurrence"])
    return {
        "source_dataset": "duke_breast_mri",
        "source_record_id": str(record_id),
        "patient_id": f"DUKE_BREAST_MRI:{record_id}",
        "timepoint_index": 0,
        "age": _to_float(_pick(row, ["Age", "age", "Age at MRI", "age_at_mri"])),
        "sex": _normalize_sex(_pick(row, ["Sex", "sex", "Gender", "gender"])),
        "stage": _pick(row, ["Stage", "AJCC Stage", "stage"]) or "unknown",
        "molecular_subtype": _pick(row, ["Mol Subtype", "Molecular subtype", "Subtype", "molecular_subtype"]) or "unknown",
        "er_status": _normalize_receptor(_pick(row, ["ER", "ER Status", "er_status"])),
        "pr_status": _normalize_receptor(_pick(row, ["PR", "PR Status", "pr_status"])),
        "her2_status": _normalize_her2(_pick(row, ["HER2", "HER2 Status", "her2_status"])),
        "ki67_percent": _to_float(_pick(row, ["Ki67", "Ki-67", "ki67_percent"])),
        "genetic_context_available": bool(_pick(row, ["Oncotype", "Oncotype DX", "Genetic test"])),
        "genetic_variant_classification": _pick(row, ["Oncotype", "Oncotype DX", "Genetic test"]) or "not_reported",
        "treatment_phase": "neoadjuvant" if _contains(treatment_text, ["neoadjuvant", "nac"]) else "unknown",
        "treatment_modalities": modalities,
        "treatment_combination_pattern": _combination_pattern(modalities),
        "regimen_text": treatment_text or "not_reported",
        "cbc_available": False,
        "symptoms_available": False,
        "imaging_available": True,
        "imaging_modality": "MRI",
        "imaging_features": imaging_features,
        "tumor_marker_available": False,
        "tumor_marker_context_only": True,
        "outcome_label_name": "pCR" if pcr not in {None, ""} else "recurrence" if recurrence not in {None, ""} else "",
        "outcome_label_value": pcr if pcr not in {None, ""} else recurrence or "",
        "source_urls": [DUKE_SOURCE_URL],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modalities = Counter()
    fields = Counter()
    outcomes = 0
    for row in rows:
        if row.get("imaging_available"):
            modalities["imaging"] += 1
        if row.get("genetic_context_available"):
            modalities["genetic_context"] += 1
        for modality in row.get("treatment_modalities") or []:
            modalities[f"treatment::{modality}"] += 1
        for key in ("age", "er_status", "pr_status", "her2_status", "molecular_subtype"):
            if row.get(key) not in {None, "", "unknown"}:
                fields[key] += 1
        if row.get("outcome_label_name") and row.get("outcome_label_value") not in {None, ""}:
            outcomes += 1
    return {
        "row_count": len(rows),
        "modality_counts": dict(modalities),
        "field_counts": dict(fields),
        "outcome_label_count": outcomes,
        "roles_supported": {
            "response_benchmark": bool(rows and outcomes > 0 and modalities.get("imaging", 0) > 0),
            "treatment_sequence_benchmark": bool(rows and any(key.startswith("treatment::") for key in modalities)),
            "genetic_context_mapping": bool(rows and modalities.get("genetic_context", 0) > 0),
            "full_oncotrack_temporal_validation": False,
        },
    }


def _write_templates(template_dir: Path) -> dict[str, str]:
    template_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "genie_bpc_brca": template_dir / "genie_bpc_brca_field_contract.csv",
        "duke_breast_mri": template_dir / "duke_breast_mri_field_contract.csv",
    }
    _write_contract_csv(paths["genie_bpc_brca"], GENIE_FIELD_CONTRACT)
    _write_contract_csv(paths["duke_breast_mri"], DUKE_FIELD_CONTRACT)
    return {key: _display_path(path) for key, path in paths.items()}


def _write_contract_csv(path: Path, contract: list[dict[str, Any]]) -> None:
    fieldnames = ["canonical_field", "role", "acceptable_source_aliases", "required_for_mapping", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in contract:
            writer.writerow({
                "canonical_field": item["canonical_field"],
                "role": item["role"],
                "acceptable_source_aliases": "; ".join(item["aliases"]),
                "required_for_mapping": item["required_for_mapping"],
                "notes": item["notes"],
            })


def _summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    mapped = [item for item in reports if item["status"] == "mapped"]
    ready = [item for item in reports if item["status"] == "ready_for_mapping"]
    return {
        "dataset_count": len(reports),
        "mapped_dataset_count": len(mapped),
        "ready_for_mapping_count": len(ready),
        "template_count": len(reports),
        "full_oncotrack_temporal_validation_ready": 0,
        "highest_priority_next": ["genie_bpc_brca", "duke_breast_mri"],
    }


def _overall_status(reports: list[dict[str, Any]]) -> str:
    if any(report["validation"]["status"] == "failed" for report in reports):
        return "needs_attention"
    if any(report["status"] == "mapped" for report in reports):
        return "strong"
    return "ready_for_mapping"


def _modalities_from_text(text: str | None) -> list[str]:
    value = (text or "").lower()
    modalities: list[str] = []
    if _contains(value, ["paclitaxel", "doxorubicin", "cyclophosphamide", "carboplatin", "docetaxel", "chemo"]):
        modalities.append("chemotherapy")
    if _contains(value, ["trastuzumab", "pertuzumab", "herceptin", "t-dm1", "tdm1", "lapatinib", "neratinib", "tucatinib"]):
        modalities.append("targeted_anti_her2")
    if _contains(value, ["tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant", "aromatase", "endocrine", "hormone"]):
        modalities.append("endocrine")
    if _contains(value, ["pembrolizumab", "atezolizumab", "nivolumab", "immunotherapy"]):
        modalities.append("immunotherapy")
    if _contains(value, ["olaparib", "talazoparib", "parp"]):
        modalities.append("parp_inhibitor")
    if _contains(value, ["radiation", "radiotherapy"]):
        modalities.append("radiation")
    if _contains(value, ["lumpectomy", "mastectomy", "surgery", "surgical"]):
        modalities.append("surgery")
    return sorted(dict.fromkeys(modalities)) or ["unknown"]


def _combination_pattern(modalities: list[str]) -> str:
    filtered = [item for item in modalities if item != "unknown"]
    if not filtered:
        return "unknown"
    return "plus".join(sorted(filtered))


def _duke_imaging_features(row: dict[str, str]) -> dict[str, float]:
    feature_hints = ("tumor", "lesion", "washout", "enhancement", "ser", "bpe", "volume", "diameter", "size")
    features: dict[str, float] = {}
    for key, value in row.items():
        normalized = key.lower()
        if any(hint in normalized for hint in feature_hints):
            number = _to_float(value)
            if number is not None:
                features[key] = number
    return features


def _normalize_sex(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized.startswith("f"):
        return "female"
    if normalized.startswith("m"):
        return "male"
    return "unknown"


def _normalize_receptor(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"positive", "pos", "+", "1", "yes", "y"} or "positive" in normalized:
        return "positive"
    if normalized in {"negative", "neg", "-", "0", "no", "n"} or "negative" in normalized:
        return "negative"
    return "unknown"


def _normalize_her2(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if "equivocal" in normalized or normalized in {"2+", "borderline"}:
        return "equivocal"
    return _normalize_receptor(value)


def _normalize_phase(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if "neo" in normalized:
        return "neoadjuvant"
    if "adjuvant" in normalized:
        return "adjuvant"
    if "metastatic" in normalized or "advanced" in normalized:
        return "metastatic"
    if "survivorship" in normalized or "follow" in normalized:
        return "survivorship"
    return "unknown"


def _outcome_name(value: str | None) -> str:
    if value in {None, ""}:
        return ""
    normalized = str(value).lower()
    if "pfs" in normalized:
        return "PFS_status"
    if "os" in normalized:
        return "OS_status"
    return "real_world_response"


def _pick(row: dict[str, str], aliases: list[str]) -> str | None:
    lower_lookup = {key.lower().strip(): value for key, value in row.items()}
    for alias in aliases:
        value = lower_lookup.get(alias.lower().strip())
        if value not in {None, ""}:
            return str(value).strip()
    return None


def _join_present(values: list[str | None]) -> str:
    return "; ".join(str(value).strip() for value in values if value not in {None, ""})


def _contains(text: str | None, needles: list[str]) -> bool:
    value = (text or "").lower()
    return any(needle.lower() in value for needle in needles)


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_like(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(round(number))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Priority Dataset Bridge",
        "",
        f"Generated at: {payload['generated_at']}",
        "",
        f"Status: **{payload['status']}**",
        "",
        payload["claim_boundary"],
        "",
        "## Datasets",
        "",
        "| Dataset | Status | Rows mapped | Best role | Source |",
        "|---|---:|---:|---|---|",
    ]
    for item in payload["datasets"].values():
        role = ", ".join(key for key, enabled in item["supported_roles"].items() if enabled)
        lines.append(
            f"| [{item['name']}]({item['source_url']}) | {item['status']} | "
            f"{item['row_count']} | {role} | {item['access_status']} |"
        )
    lines.extend(["", "## Next Actions"])
    lines.extend(f"- {action}" for action in payload["next_actions"])
    lines.extend(["", "## Blocked Claims"])
    lines.extend(f"- {claim}" for claim in payload["blocked_claims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve(path: str | Path | None) -> Path:
    if path is None:
        return ROOT_DIR
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)
