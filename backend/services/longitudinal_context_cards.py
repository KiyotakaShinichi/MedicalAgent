"""Longitudinal context cards.

Builds deterministic, auditable context cards from existing synthetic
patient timeline data.  Each card has:

  - card_type
  - patient_id
  - value or short summary
  - provenance: source CSV path + row indices used
  - timestamp(s)
  - missing_evidence: explicit list of fields the card cannot fill
  - card_disclaimer (always "Synthetic engineering signal · Not a
    clinical prediction · For clinician review")

The cards do NOT store private chain-of-thought, free-form
generation, or any open-ended memory.  They are pure derived data
from the existing CSV rows.

Output: ``Data/evals/ops/latest_longitudinal_context_card_eval.json``
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ROWS_PATH = Path("Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
OUTPUT_PATH = Path("Data/evals/ops/latest_longitudinal_context_card_eval.json")

CARD_DISCLAIMER = "Synthetic engineering signal · Not a clinical prediction · For clinician review"


CARD_TYPES: tuple[str, ...] = (
    "latest_cbc_trend",
    "symptom_trend",
    "imaging_summary_trend",
    "medication_treatment_context",
    "missing_evidence",
    "review_flags",
    "last_safety_escalation",
)


@dataclass(frozen=True)
class Card:
    card_type: str
    patient_id: str
    summary: str
    provenance_rows: list[int]
    timestamps: list[str]
    missing_evidence: list[str]
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_type": self.card_type,
            "patient_id": self.patient_id,
            "summary": self.summary,
            "provenance": {
                "source_csv": str(ROWS_PATH).replace("\\", "/"),
                "row_indices": list(self.provenance_rows),
            },
            "timestamps": list(self.timestamps),
            "missing_evidence": list(self.missing_evidence),
            "card_disclaimer": CARD_DISCLAIMER,
            "clinical_validation": False,
            "extras": dict(self.extras),
        }


def _latest_cbc_trend(patient_df: pd.DataFrame) -> Card:
    sub = patient_df.sort_values("treatment_date")
    last3 = sub.tail(3)
    missing: list[str] = []
    parts: list[str] = []
    for col, label in (("nadir_wbc", "WBC nadir"), ("nadir_hemoglobin", "Hgb nadir"), ("nadir_platelets", "Plt nadir")):
        if col in sub.columns and sub[col].notna().any():
            values = [f"{v:.1f}" for v in last3[col].dropna().tail(3).tolist()]
            parts.append(f"{label} last 3: {', '.join(values)}")
        else:
            missing.append(col)
    return Card(
        card_type="latest_cbc_trend",
        patient_id=str(sub["patient_id"].iloc[-1]),
        summary=" | ".join(parts) if parts else "No CBC data on file",
        provenance_rows=list(map(int, last3.index.tolist())),
        timestamps=[str(d) for d in last3["treatment_date"].tolist()],
        missing_evidence=missing,
    )


def _symptom_trend(patient_df: pd.DataFrame) -> Card:
    sub = patient_df.sort_values("treatment_date")
    last3 = sub.tail(3)
    missing: list[str] = []
    parts: list[str] = []
    if "max_symptom_severity" in sub.columns and sub["max_symptom_severity"].notna().any():
        sev = last3["max_symptom_severity"].dropna().tolist()
        parts.append(f"Max severity last 3: {', '.join(f'{v:.0f}' for v in sev)}")
    else:
        missing.append("max_symptom_severity")
    if "symptom_count" in sub.columns and sub["symptom_count"].notna().any():
        cnt = last3["symptom_count"].dropna().tolist()
        parts.append(f"Symptom count last 3: {', '.join(str(int(v)) for v in cnt)}")
    else:
        missing.append("symptom_count")
    return Card(
        card_type="symptom_trend",
        patient_id=str(sub["patient_id"].iloc[-1]),
        summary=" | ".join(parts) if parts else "No symptom data on file",
        provenance_rows=list(map(int, last3.index.tolist())),
        timestamps=[str(d) for d in last3["treatment_date"].tolist()],
        missing_evidence=missing,
    )


def _imaging_summary_trend(patient_df: pd.DataFrame) -> Card:
    sub = patient_df.sort_values("treatment_date")
    last3 = sub.tail(3)
    missing: list[str] = []
    parts: list[str] = []
    if "mri_tumor_size_cm" in sub.columns and sub["mri_tumor_size_cm"].notna().any():
        vals = last3["mri_tumor_size_cm"].dropna().tolist()
        parts.append(f"MRI tumor size last 3: {', '.join(f'{v:.2f} cm' for v in vals)}")
    else:
        missing.append("mri_tumor_size_cm")
    if "mri_percent_change_from_baseline" in sub.columns and sub["mri_percent_change_from_baseline"].notna().any():
        vals = last3["mri_percent_change_from_baseline"].dropna().tolist()
        parts.append(f"MRI % change last 3: {', '.join(f'{v:+.1f}%' for v in vals)}")
    else:
        missing.append("mri_percent_change_from_baseline")
    return Card(
        card_type="imaging_summary_trend",
        patient_id=str(sub["patient_id"].iloc[-1]),
        summary=" | ".join(parts) if parts else "No imaging data on file",
        provenance_rows=list(map(int, last3.index.tolist())),
        timestamps=[str(d) for d in last3["treatment_date"].tolist()],
        missing_evidence=missing,
    )


def _medication_treatment_context(patient_df: pd.DataFrame) -> Card:
    sub = patient_df.sort_values("treatment_date")
    last = sub.tail(1).iloc[0] if len(sub) else None
    parts: list[str] = []
    missing: list[str] = []
    timestamps: list[str] = []
    rows: list[int] = []
    if last is None:
        return Card("medication_treatment_context", "?", "No treatment record",
                    [], [], ["regimen", "cycle", "dose_delayed", "dose_reduced"])
    rows = [int(sub.tail(1).index[0])]
    timestamps = [str(last.get("treatment_date"))]
    for col in ("regimen", "cycle", "dose_delayed", "dose_reduced"):
        if col in sub.columns and pd.notna(last.get(col)):
            parts.append(f"{col}: {last[col]}")
        else:
            missing.append(col)
    return Card(
        card_type="medication_treatment_context",
        patient_id=str(last["patient_id"]),
        summary=" | ".join(parts) if parts else "No treatment context",
        provenance_rows=rows,
        timestamps=timestamps,
        missing_evidence=missing,
    )


def _missing_evidence_card(patient_df: pd.DataFrame, all_required_cols: Sequence[str]) -> Card:
    missing: list[str] = []
    for col in all_required_cols:
        if col not in patient_df.columns or patient_df[col].isna().all():
            missing.append(col)
    sub = patient_df.sort_values("treatment_date").tail(1)
    return Card(
        card_type="missing_evidence",
        patient_id=str(sub["patient_id"].iloc[-1]) if len(sub) else "?",
        summary=(
            f"{len(missing)} required field(s) missing across the patient timeline"
            if missing else "All required fields present"
        ),
        provenance_rows=list(map(int, sub.index.tolist())),
        timestamps=[str(d) for d in sub["treatment_date"].tolist()],
        missing_evidence=missing,
    )


def _review_flags(patient_df: pd.DataFrame) -> Card:
    sub = patient_df.sort_values("treatment_date")
    last = sub.tail(1).iloc[0] if len(sub) else None
    flags: list[str] = []
    if last is None:
        return Card("review_flags", "?", "No data", [], [], ["urgent_intervention_needed"])
    if last.get("urgent_intervention_needed") == 1:
        flags.append("urgent_intervention_needed")
    if last.get("support_intervention_needed") == 1:
        flags.append("support_intervention_needed")
    if last.get("toxicity_risk_binary") == 1:
        flags.append("toxicity_risk_flag")
    if pd.notna(last.get("max_symptom_severity")) and float(last["max_symptom_severity"]) >= 7.0:
        flags.append("severe_symptom_score")
    return Card(
        card_type="review_flags",
        patient_id=str(last["patient_id"]),
        summary=f"{len(flags)} review flag(s): {', '.join(flags) or 'none'}",
        provenance_rows=[int(sub.tail(1).index[0])],
        timestamps=[str(last.get("treatment_date"))],
        missing_evidence=[],
    )


def _last_safety_escalation(patient_df: pd.DataFrame) -> Card:
    """Most recent cycle where urgent_intervention_needed == 1."""
    sub = patient_df.sort_values("treatment_date")
    missing = []
    if "urgent_intervention_needed" not in sub.columns:
        return Card(
            "last_safety_escalation", "?",
            "urgent_intervention_needed column missing", [], [],
            ["urgent_intervention_needed"],
        )
    flagged = sub[sub["urgent_intervention_needed"] == 1]
    if flagged.empty:
        return Card(
            "last_safety_escalation",
            str(sub["patient_id"].iloc[-1]) if len(sub) else "?",
            "No safety escalation on record",
            [],
            [],
            missing,
        )
    last = flagged.tail(1).iloc[0]
    return Card(
        card_type="last_safety_escalation",
        patient_id=str(last["patient_id"]),
        summary=f"Last urgent escalation cycle {int(last.get('cycle', 0))}",
        provenance_rows=[int(flagged.tail(1).index[0])],
        timestamps=[str(last.get("treatment_date"))],
        missing_evidence=[],
    )


def build_cards_for_patient(patient_df: pd.DataFrame) -> list[Card]:
    """Build all 7 cards for one patient.  Each card has provenance."""
    required_cols = (
        "nadir_wbc", "nadir_hemoglobin", "nadir_platelets",
        "max_symptom_severity", "symptom_count",
        "mri_tumor_size_cm", "mri_percent_change_from_baseline",
        "regimen", "urgent_intervention_needed",
    )
    return [
        _latest_cbc_trend(patient_df),
        _symptom_trend(patient_df),
        _imaging_summary_trend(patient_df),
        _medication_treatment_context(patient_df),
        _missing_evidence_card(patient_df, required_cols),
        _review_flags(patient_df),
        _last_safety_escalation(patient_df),
    ]


# ─── Eval ────────────────────────────────────────────────────────────────


_FORBIDDEN_INFERENCE_TOKENS = (
    "you have cancer",
    "you should stop",
    "you should start",
    "prognosis is",
    "diagnosis is",
    "recurrence confirmed",
    "treatment recommendation",
)


def build_report(sample_patient_count: int = 50) -> dict[str, Any]:
    started = time.perf_counter()
    if not ROWS_PATH.exists():
        return {
            "schema_version": "longitudinal_context_card_eval_v1",
            "status": "needs_attention",
            "clinical_validation": False,
            "claim_boundary": "Source CSV missing.  Not clinical validation.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    df = pd.read_csv(ROWS_PATH)
    sample_pids = sorted(df["patient_id"].astype(str).unique())[:sample_patient_count]
    all_cards: list[dict[str, Any]] = []
    provenance_with_rows = 0
    timestamp_coverage = 0
    missing_disclosure = 0
    unsafe_inference = 0
    for pid in sample_pids:
        sub = df[df["patient_id"].astype(str) == pid]
        cards = build_cards_for_patient(sub)
        for card in cards:
            d = card.to_dict()
            all_cards.append(d)
            if d["provenance"]["row_indices"]:
                provenance_with_rows += 1
            if d["timestamps"]:
                timestamp_coverage += 1
            # Missing evidence disclosure: a card MUST report a
            # missing_evidence list (possibly empty) — that's the
            # disclosure surface.  We count cards whose summary
            # explicitly names a missing field OR whose missing list
            # is non-empty.
            if d["missing_evidence"] or "missing" in d["summary"].lower() or "no " in d["summary"].lower():
                missing_disclosure += 1
            text = d["summary"].lower()
            if any(tok in text for tok in _FORBIDDEN_INFERENCE_TOKENS):
                unsafe_inference += 1

    n_cards = len(all_cards)
    metrics = {
        "provenance_coverage": round(provenance_with_rows / n_cards, 4) if n_cards else 0.0,
        "timestamp_coverage": round(timestamp_coverage / n_cards, 4) if n_cards else 0.0,
        "missing_evidence_disclosure_rate": round(missing_disclosure / n_cards, 4) if n_cards else 0.0,
        "unsafe_inference_rate": round(unsafe_inference / n_cards, 4) if n_cards else 0.0,
        "card_disclaimer_present_rate": 1.0,  # enforced by Card.to_dict
    }

    return {
        "schema_version": "longitudinal_context_card_eval_v1",
        "status": "informational",
        "label": "longitudinal_context_card_eval",
        "clinical_validation": False,
        "claim_boundary": (
            "Longitudinal context cards.  Deterministic, provenance-stamped, "
            "patient-facing reference cards built from existing synthetic "
            "CSV rows.  No open-ended memory, no chain-of-thought, no "
            "clinical claim.  Eval-only.  Not clinical validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "source_csv": str(ROWS_PATH).replace("\\", "/"),
        "n_patients_sampled": len(sample_pids),
        "n_cards": n_cards,
        "card_types": list(CARD_TYPES),
        "metrics": metrics,
        "card_type_counts": dict(Counter(c["card_type"] for c in all_cards)),
        "sample_cards": all_cards[:21],
        "contamination_note": (
            "Cards are built from the synthetic-only patient timeline.  "
            "Do not treat any card as a clinical signal."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH, sample_patient_count: int = 50) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_report(sample_patient_count=sample_patient_count), indent=2),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "CARD_DISCLAIMER", "CARD_TYPES", "OUTPUT_PATH",
    "build_cards_for_patient", "build_report", "write_report",
]
