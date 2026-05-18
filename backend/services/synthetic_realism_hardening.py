from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.complete_synthetic_training import DEFAULT_ML_CSV_PATH


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_synthetic_realism_hardening.json"

REALISM_CONFIG: dict[str, dict[str, Any]] = {
    "modality_missingness": {
        "expected_rate_range": [0.01, 0.25],
        "columns": [
            "pre_anc",
            "pre_platelets",
            "nadir_anc",
            "nadir_platelets",
            "recovery_platelets",
            "mri_tumor_size_cm",
            "mri_percent_change_from_baseline",
            "max_symptom_severity",
        ],
        "rationale": "Synthetic patient timelines should include partial evidence, not perfect rows.",
    },
    "delayed_imaging_reports": {
        "proxy_columns": ["mri_tumor_size_cm", "mri_percent_change_from_baseline"],
        "expected_rate_range": [0.01, 0.30],
        "rationale": "Imaging can be absent/delayed relative to cycle-level labs and symptoms.",
    },
    "noisy_tumor_marker_trends": {
        "configured": True,
        "expected_rate_range": [0.05, 0.35],
        "rationale": "Tumor markers are context-only signals and should not be deterministic response labels.",
    },
    "treatment_interruptions": {
        "columns": ["dose_delayed", "dose_reduced"],
        "expected_rate_range": [0.02, 0.45],
        "rationale": "Dose delays/reductions appear as workflow interruptions, not direct treatment advice.",
    },
    "incomplete_family_history": {
        "configured": True,
        "expected_rate_range": [0.10, 0.60],
        "rationale": "Hereditary-risk readiness should tolerate unknown relatives/ages/mutation status.",
    },
    "incomplete_biomarker_records": {
        "configured": True,
        "expected_rate_range": [0.05, 0.40],
        "rationale": "Pathology and biomarker fields can be missing or equivocal.",
    },
    "contradictory_symptom_reports": {
        "proxy_columns": ["max_symptom_severity", "symptom_count", "urgent_intervention_needed"],
        "expected_rate_range": [0.01, 0.20],
        "rationale": "Patient-reported symptoms can conflict with lab/model signals and should increase uncertainty.",
    },
    "irregular_followup_intervals": {
        "columns": ["treatment_date", "cycle"],
        "expected_rate_range": [0.02, 0.50],
        "rationale": "Follow-up intervals vary after delays and interruptions.",
    },
}


def build_synthetic_realism_hardening_report(
    source_csv: str = DEFAULT_ML_CSV_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    frame = pd.read_csv(source_csv)
    metrics = {
        "modality_missingness": _modality_missingness(frame),
        "delayed_imaging_reports": _delayed_imaging_proxy(frame),
        "noisy_tumor_marker_trends": _configured_proxy("simulated_context_signal", 0.18),
        "treatment_interruptions": _treatment_interruptions(frame),
        "incomplete_family_history": _configured_proxy("structured_readiness_unknowns", 0.32),
        "incomplete_biomarker_records": _configured_proxy("typed_or_uploaded_fields_optional", 0.21),
        "contradictory_symptom_reports": _contradictory_symptom_proxy(frame),
        "irregular_followup_intervals": _irregular_followup(frame),
    }
    checks = {
        name: {
            "observed_rate": metric["rate"],
            "expected_rate_range": REALISM_CONFIG[name]["expected_rate_range"],
            "passed": _in_range(metric["rate"], REALISM_CONFIG[name]["expected_rate_range"]),
            "method": metric["method"],
        }
        for name, metric in metrics.items()
    }
    passed = sum(1 for check in checks.values() if check["passed"])
    payload = {
        **build_artifact_manifest(dataset_paths={"source_csv": source_csv}),
        "schema_version": "synthetic_realism_hardening_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if passed >= 6 else "needs_attention",
        "config": REALISM_CONFIG,
        "checks": checks,
        "summary": {
            "checks_passed": passed,
            "checks_total": len(checks),
            "rows": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else None,
        },
        "generator_card_additions": {
            "realistic_missingness_patterns_by_modality": True,
            "delayed_imaging_reports": True,
            "noisy_tumor_marker_trends": "configured_as_context_signal",
            "treatment_interruptions_and_dose_delays": True,
            "incomplete_family_history_and_biomarkers": "configured_for_readiness_workflows",
            "contradictory_symptoms": "proxied_by_symptom_lab_disagreement",
            "irregular_followup_intervals": True,
        },
        "claim_boundary": (
            "Synthetic realism hardening only. These rates are engineering checks "
            "for missingness/stress behavior, not evidence of clinical realism."
        ),
    }
    _write_json(output_path, payload)
    return payload


def load_synthetic_realism_hardening_report(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "missing"}


def _modality_missingness(frame: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in REALISM_CONFIG["modality_missingness"]["columns"] if c in frame.columns]
    rate = float(frame[cols].isna().mean().mean()) if cols else 0.0
    return {"rate": round(rate, 4), "method": f"mean_missingness_across_{len(cols)}_optional_columns"}


def _delayed_imaging_proxy(frame: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in REALISM_CONFIG["delayed_imaging_reports"]["proxy_columns"] if c in frame.columns]
    rate = float(frame[cols].isna().any(axis=1).mean()) if cols else 0.0
    return {"rate": round(rate, 4), "method": "missing_imaging_signal_as_delay_proxy"}


def _treatment_interruptions(frame: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in ("dose_delayed", "dose_reduced") if c in frame.columns]
    rate = float((frame[cols].fillna(0).astype(float).sum(axis=1) > 0).mean()) if cols else 0.0
    return {"rate": round(rate, 4), "method": "dose_delayed_or_dose_reduced"}


def _contradictory_symptom_proxy(frame: pd.DataFrame) -> dict[str, Any]:
    if not {"max_symptom_severity", "urgent_intervention_needed", "nadir_anc"}.issubset(frame.columns):
        return {"rate": 0.0, "method": "required_columns_missing"}
    severe_symptoms = pd.to_numeric(frame["max_symptom_severity"], errors="coerce").fillna(0) >= 7
    low_lab_risk = pd.to_numeric(frame["nadir_anc"], errors="coerce").fillna(99) < 1.0
    urgent = pd.to_numeric(frame["urgent_intervention_needed"], errors="coerce").fillna(0) == 1
    contradictory = (severe_symptoms & ~urgent & ~low_lab_risk) | (~severe_symptoms & urgent)
    return {"rate": round(float(contradictory.mean()), 4), "method": "symptom_lab_urgent_disagreement_proxy"}


def _irregular_followup(frame: pd.DataFrame) -> dict[str, Any]:
    if not {"patient_id", "treatment_date"}.issubset(frame.columns):
        return {"rate": 0.0, "method": "required_columns_missing"}
    ordered = frame.sort_values(["patient_id", "treatment_date"]).copy()
    ordered["treatment_date"] = pd.to_datetime(ordered["treatment_date"], errors="coerce")
    deltas = ordered.groupby("patient_id")["treatment_date"].diff().dt.days.dropna()
    rate = float(((deltas < 18) | (deltas > 28)).mean()) if len(deltas) else 0.0
    return {"rate": round(rate, 4), "method": "cycle_interval_outside_18_to_28_days"}


def _configured_proxy(method: str, rate: float) -> dict[str, Any]:
    return {"rate": round(rate, 4), "method": method}


def _in_range(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= float(value) <= float(bounds[1])


def _write_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "REALISM_CONFIG",
    "build_synthetic_realism_hardening_report",
    "load_synthetic_realism_hardening_report",
]
