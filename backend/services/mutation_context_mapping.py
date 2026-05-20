from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_mutation_context_mapping.json"

CONTEXT_GENES: list[dict[str, Any]] = [
    {"gene": "PIK3CA", "category": "somatic_pathway_context", "patient_facing_role": "record organization only"},
    {"gene": "TP53", "category": "somatic_pathway_context", "patient_facing_role": "record organization only"},
    {"gene": "GATA3", "category": "somatic_pathway_context", "patient_facing_role": "record organization only"},
    {"gene": "ESR1", "category": "somatic_endocrine_context", "patient_facing_role": "record organization only"},
    {"gene": "ERBB2", "category": "somatic_her2_context", "patient_facing_role": "record organization only"},
    {"gene": "BRCA1", "category": "germline_sensitive_context", "patient_facing_role": "genetic-counselor review readiness only"},
    {"gene": "BRCA2", "category": "germline_sensitive_context", "patient_facing_role": "genetic-counselor review readiness only"},
    {"gene": "PALB2", "category": "germline_sensitive_context", "patient_facing_role": "genetic-counselor review readiness only"},
    {"gene": "ATM", "category": "germline_sensitive_context", "patient_facing_role": "genetic-counselor review readiness only"},
    {"gene": "CHEK2", "category": "germline_sensitive_context", "patient_facing_role": "genetic-counselor review readiness only"},
    {"gene": "PTEN", "category": "germline_sensitive_context", "patient_facing_role": "genetic-counselor review readiness only"},
]

CLAIM_BOUNDARY = (
    "Mutation-context mapping organizes reported genes/alterations as external feature context and "
    "genetic-counseling readiness signals. It does not diagnose inherited cancer risk, interpret variants "
    "as medical advice, recommend treatment, or predict treatment response from mutations."
)


def build_mutation_context_mapping(
    *,
    mutation_csv: str | None = None,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = _read_csv(_resolve(mutation_csv)) if mutation_csv else []
    mapped = [_map_mutation_row(row) for row in rows]
    gene_counts = Counter(item["gene"] for item in mapped if item["gene"] != "unknown")
    category_counts = Counter(item["category"] for item in mapped)
    payload: dict[str, Any] = {
        "schema_version": "mutation_context_mapping_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if mapped else "ready_for_mapping",
        "supported_genes": CONTEXT_GENES,
        "mapped_row_count": len(mapped),
        "gene_counts": dict(gene_counts),
        "category_counts": dict(category_counts),
        "mapped_examples": mapped[:20],
        "feature_policy": {
            "may_use_as_predictor": "context_only_after_ablation",
            "default_model_weight": "none_without_external_validation",
            "requires_missingness_mask": True,
            "requires_source_type": "somatic_vs_germline_when_available",
            "promotion_allowed": False,
        },
        "blocked_claims": [
            "BRCA or other mutation means the patient will develop cancer",
            "VUS means positive",
            "mutation status alone means treatment is working or failing",
            "mutation status alone recommends PARP, HER2, endocrine, or chemotherapy changes",
            "somatic tumor alteration equals inherited family risk",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    return payload


def _map_mutation_row(row: dict[str, str]) -> dict[str, Any]:
    gene = (_pick(row, ["gene", "HUGO_SYMBOL", "hugo_symbol", "GENE", "Gene"]) or "unknown").upper()
    known = next((item for item in CONTEXT_GENES if item["gene"] == gene), None)
    classification = _pick(row, ["variant_classification", "VARIANT_CLASSIFICATION", "classification", "Mutation_Status"])
    source_type = _normalize_source_type(_pick(row, ["source_type", "test_type", "sample_type", "TEST_TYPE"]))
    return {
        "source_dataset": _pick(row, ["source_dataset", "dataset", "study_id"]) or "unknown",
        "patient_id": _pick(row, ["patient_id", "PATIENT_ID", "case_id", "sample_id"]) or "unknown",
        "gene": gene,
        "known_oncotrack_context_gene": known is not None,
        "category": known["category"] if known else "other_reported_gene",
        "variant_classification": classification or "not_reported",
        "source_type": source_type,
        "patient_facing_role": known["patient_facing_role"] if known else "record organization only",
        "review_route": "genetic_counselor_review" if "germline" in source_type else "clinician_review",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _normalize_source_type(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if "germline" in normalized or "blood" in normalized or "saliva" in normalized:
        return "germline_or_possible_germline"
    if "somatic" in normalized or "tumor" in normalized or "tissue" in normalized:
        return "somatic_tumor"
    return "unknown_source_type"


def _pick(row: dict[str, str], aliases: list[str]) -> str | None:
    lookup = {key.lower().strip(): value for key, value in row.items()}
    for alias in aliases:
        value = lookup.get(alias.lower().strip())
        if value not in {None, ""}:
            return str(value).strip()
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path | None) -> Path:
    if path is None:
        return ROOT_DIR
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
