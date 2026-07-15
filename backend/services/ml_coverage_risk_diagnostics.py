"""Reviewer-facing coverage/risk diagnostics for synthetic ML outputs.

This module does not train or promote a model. It consolidates the existing
evidence-abstention and statistical-audit artifacts into one question:

  When the synthetic model is missing key evidence, does it abstain instead of
  inventing confidence?

The artifact is intentionally non-clinical. It is meant for portfolio/MLE
review, not patient-facing medical authority.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE_PATH = Path("Data/evals/models/latest_evidence_abstention_eval.json")
DEFAULT_STATISTICAL_AUDIT_PATH = Path("Data/evals/models/latest_synthetic_prediction_statistical_audit.json")
DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_ml_coverage_risk_diagnostics.json")

REQUIRED_ABSTENTION_SCENARIOS = (
    "no_imaging",
    "cbc_pre_only",
    "demographics_only",
    "symptoms_only",
)

CLAIM_BOUNDARY = (
    "Synthetic ML coverage/risk diagnostic only. It checks abstention and "
    "selective-risk behavior inside simulator-built data. It is not clinical "
    "validation, not real-patient calibration, not treatment evidence, and not "
    "healthcare production readiness."
)


def build_ml_coverage_risk_diagnostics(
    *,
    evidence_path: Path | str = DEFAULT_EVIDENCE_PATH,
    statistical_audit_path: Path | str = DEFAULT_STATISTICAL_AUDIT_PATH,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    evidence = _load(Path(evidence_path))
    statistical = _load(Path(statistical_audit_path))

    scenarios = evidence.get("scenarios") or []
    scenario_map = {row.get("scenario"): row for row in scenarios if isinstance(row, dict)}
    required = _required_abstention_summary(scenario_map)
    selective = statistical.get("selective_risk_curve") or []
    selective_summary = _selective_risk_summary(selective)

    full = scenario_map.get("full_data") or {}
    imaging_only = scenario_map.get("imaging_only") or {}
    status = _status(
        scenario_count=len(scenarios),
        full_data_coverage=_as_float(full.get("coverage_rate")),
        required=required,
        statistical=statistical,
        selective_summary=selective_summary,
    )

    payload: dict[str, Any] = {
        "schema_version": "ml_coverage_risk_diagnostics_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "synthetic_only": True,
        "healthcare_production_ready": False,
        "promotion_decision": "hold_synthetic_only",
        "inputs": {
            "evidence_abstention_eval": str(Path(evidence_path)).replace("\\", "/"),
            "synthetic_prediction_statistical_audit": str(Path(statistical_audit_path)).replace("\\", "/"),
        },
        "scenario_count": len(scenarios),
        "full_data": {
            "coverage_rate": full.get("coverage_rate"),
            "abstention_rate": full.get("abstention_rate"),
            "covered_accuracy": full.get("covered_accuracy"),
        },
        "required_abstention_scenarios": required,
        "imaging_only_context": {
            "coverage_rate": imaging_only.get("coverage_rate"),
            "covered_accuracy": imaging_only.get("covered_accuracy"),
            "interpretation": (
                "Imaging-only coverage is allowed in this synthetic response-pattern "
                "head because imaging/report trend is a primary monitoring signal. "
                "It still does not authorize diagnosis, treatment decisions, or "
                "patient outcome probability claims."
            ),
        },
        "selective_risk": selective_summary,
        "coverage_risk_findings": _findings(required, full, imaging_only, selective_summary),
        "reviewer_next_steps": [
            "Keep no-imaging and low-evidence cases abstained for response-pattern heads.",
            "Show missing-modality reasons beside model outputs so users understand why a head abstains.",
            "Keep selective-risk curves as synthetic engineering evidence only.",
            "Do not promote any model head until external temporal validation and clinician-reviewed labels exist.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _required_abstention_summary(scenario_map: dict[str | None, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_passed = True
    for name in REQUIRED_ABSTENTION_SCENARIOS:
        scenario = scenario_map.get(name) or {}
        abstention_rate = _as_float(scenario.get("abstention_rate"))
        coverage_rate = _as_float(scenario.get("coverage_rate"))
        passed = abstention_rate is not None and abstention_rate >= 0.95
        all_passed = all_passed and passed
        rows.append(
            {
                "scenario": name,
                "abstention_rate": abstention_rate,
                "coverage_rate": coverage_rate,
                "passed": passed,
            }
        )
    return {
        "scenario_count": len(rows),
        "all_required_scenarios_passed": all_passed,
        "minimum_required_abstention_rate": min(
            (row["abstention_rate"] for row in rows if row["abstention_rate"] is not None),
            default=None,
        ),
        "scenarios": rows,
    }


def _selective_risk_summary(curve: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(curve, key=lambda row: _as_float(row.get("minimum_probability_margin")) or 0.0)
    coverages = [_as_float(row.get("coverage")) for row in ordered]
    accuracies = [_as_float(row.get("covered_accuracy")) for row in ordered]
    nonincreasing_coverage = all(
        coverages[idx] is not None
        and coverages[idx + 1] is not None
        and coverages[idx] >= coverages[idx + 1]
        for idx in range(max(0, len(coverages) - 1))
    )
    accuracy_lift = None
    if accuracies and accuracies[0] is not None and accuracies[-1] is not None:
        accuracy_lift = round(float(accuracies[-1] - accuracies[0]), 4)
    return {
        "point_count": len(ordered),
        "nonincreasing_coverage": nonincreasing_coverage,
        "covered_accuracy_lift_highest_margin": accuracy_lift,
        "curve": ordered,
        "interpretation": (
            "Higher probability-margin thresholds should reduce coverage and, "
            "inside this synthetic export only, can increase covered-row accuracy. "
            "This is abstention behavior evidence, not clinical performance."
        ),
    }


def _status(
    *,
    scenario_count: int,
    full_data_coverage: float | None,
    required: dict[str, Any],
    statistical: dict[str, Any],
    selective_summary: dict[str, Any],
) -> str:
    if (
        scenario_count >= 8
        and full_data_coverage is not None
        and full_data_coverage >= 0.95
        and required["all_required_scenarios_passed"]
        and statistical.get("clinical_validation") is False
        and statistical.get("synthetic_only") is True
        and statistical.get("promotion_decision") == "hold_synthetic_only"
        and selective_summary["point_count"] >= 5
    ):
        return "strong"
    if required["all_required_scenarios_passed"]:
        return "acceptable"
    return "needs_attention"


def _findings(
    required: dict[str, Any],
    full: dict[str, Any],
    imaging_only: dict[str, Any],
    selective_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "finding": "low_evidence_abstention",
            "status": "passed" if required["all_required_scenarios_passed"] else "needs_attention",
            "evidence": required,
        },
        {
            "finding": "full_data_remains_covered",
            "status": "passed" if (_as_float(full.get("coverage_rate")) or 0.0) >= 0.95 else "needs_attention",
            "evidence": {
                "coverage_rate": full.get("coverage_rate"),
                "covered_accuracy": full.get("covered_accuracy"),
            },
        },
        {
            "finding": "imaging_only_is_contextually_limited",
            "status": "informational",
            "evidence": {
                "coverage_rate": imaging_only.get("coverage_rate"),
                "covered_accuracy": imaging_only.get("covered_accuracy"),
                "note": "Synthetic monitoring signal only; not a patient outcome claim.",
            },
        },
        {
            "finding": "selective_risk_curve_available",
            "status": "passed" if selective_summary["point_count"] >= 5 else "needs_attention",
            "evidence": {
                "point_count": selective_summary["point_count"],
                "accuracy_lift": selective_summary["covered_accuracy_lift_highest_margin"],
            },
        },
    ]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "_exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json", "_exists": True}
    return payload if isinstance(payload, dict) else {"status": "invalid_shape", "_exists": True}


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "REQUIRED_ABSTENTION_SCENARIOS",
    "build_ml_coverage_risk_diagnostics",
]
