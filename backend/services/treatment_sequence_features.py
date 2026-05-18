from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_SOURCE_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_SEQUENCE_CSV = "Data/external_bridge/synthetic_treatment_sequences.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_treatment_sequence_feature_eval.json"

CLAIM_BOUNDARY = (
    "Synthetic treatment-sequence features organize simulated treatment context only. This artifact does not compare "
    "real treatment efficacy, recommend a regimen, or tell a patient to start, stop, delay, or switch therapy."
)


def build_treatment_sequence_feature_eval(
    *,
    source_csv: str = DEFAULT_SOURCE_CSV,
    sequence_csv: str = DEFAULT_SEQUENCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = _read_csv(_resolve(source_csv))
    patient_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        patient_id = row.get("patient_id") or ""
        if patient_id:
            patient_rows[patient_id].append(row)

    sequence_rows = [
        _patient_sequence_row(patient_id, sorted(group, key=lambda item: _to_int(item.get("cycle")) or 0))
        for patient_id, group in sorted(patient_rows.items())
    ]
    _write_csv(_resolve(sequence_csv), sequence_rows)

    pattern_counts = Counter(row["treatment_combination_pattern"] for row in sequence_rows)
    modality_counts = Counter()
    for row in sequence_rows:
        for modality in row["treatment_modalities"]:
            modality_counts[modality] += 1

    payload: dict[str, Any] = {
        "schema_version": "treatment_sequence_feature_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if sequence_rows and len(pattern_counts) >= 3 else "acceptable" if sequence_rows else "needs_attention",
        "source_csv": source_csv,
        "sequence_csv": sequence_csv,
        "patient_count": len(sequence_rows),
        "source_row_count": len(rows),
        "pattern_count": len(pattern_counts),
        "pattern_counts": dict(pattern_counts.most_common()),
        "modality_counts": dict(modality_counts.most_common()),
        "modality_definitions": {
            "chemotherapy": "Regimen contains chemotherapy backbone terms.",
            "targeted_anti_her2": "HER2-positive subtype or TCHP-like regimen context.",
            "endocrine": "HR-positive subtype or endocrine therapy text.",
            "surgery_planned": "Synthetic non-metastatic treatment journey planning context.",
            "radiation_planned": "Synthetic non-metastatic post-local-control planning context.",
            "supportive_care": "Dose delay/reduction or intervention signal context.",
        },
        "examples": sequence_rows[:10],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _patient_sequence_row(patient_id: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    modalities: set[str] = set()
    cycle_flags = []
    for row in rows:
        flags = _row_modalities(row)
        modalities.update(flags)
        cycle_flags.append({
            "cycle": _to_int(row.get("cycle")),
            "regimen": row.get("regimen"),
            "modalities": sorted(flags),
            "dose_delayed": _to_bool(row.get("dose_delayed")),
            "dose_reduced": _to_bool(row.get("dose_reduced")),
        })
    ordered = _ordered_modalities(modalities)
    return {
        "patient_id": patient_id,
        "age": _to_float(first.get("age")),
        "stage": first.get("stage") or "unknown",
        "molecular_subtype": first.get("molecular_subtype") or "unknown",
        "cycle_count": len(rows),
        "treatment_modalities": ordered,
        "treatment_combination_pattern": "+".join(ordered) if ordered else "unknown",
        "has_chemo": "chemotherapy" in modalities,
        "has_targeted_anti_her2": "targeted_anti_her2" in modalities,
        "has_endocrine": "endocrine" in modalities,
        "has_surgery_planned": "surgery_planned" in modalities,
        "has_radiation_planned": "radiation_planned" in modalities,
        "has_supportive_care_context": "supportive_care" in modalities,
        "cycle_flags_json": json.dumps(cycle_flags, sort_keys=True),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _row_modalities(row: dict[str, str]) -> set[str]:
    regimen = (row.get("regimen") or "").lower()
    subtype = (row.get("molecular_subtype") or "").lower()
    stage = (row.get("stage") or "").upper()
    modalities: set[str] = set()

    if any(term in regimen for term in ("ac", "paclitaxel", "carboplatin", "tchp", "chemo")):
        modalities.add("chemotherapy")
    if "tchp" in regimen or "her2+" in subtype or "her2pos" in subtype:
        modalities.add("targeted_anti_her2")
    if "endocrine" in regimen or subtype.startswith("hr+") or "hr+/her2" in subtype:
        modalities.add("endocrine")
    if stage and stage != "IV":
        modalities.add("surgery_planned")
        modalities.add("radiation_planned")
    if _to_bool(row.get("dose_delayed")) or _to_bool(row.get("dose_reduced")) or (_to_int(row.get("intervention_count")) or 0) > 0:
        modalities.add("supportive_care")
    return modalities


def _ordered_modalities(modalities: set[str]) -> list[str]:
    order = [
        "chemotherapy",
        "targeted_anti_her2",
        "endocrine",
        "surgery_planned",
        "radiation_planned",
        "supportive_care",
    ]
    return [item for item in order if item in modalities]


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
                key: json.dumps(value, sort_keys=True) if isinstance(value, list) else value
                for key, value in row.items()
            })


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT_DIR / resolved


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
