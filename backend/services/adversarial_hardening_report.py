"""Report v3 hardening before/after and keep the next holdout separate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/safety/latest_adversarial_hardening_report.json")
V3_PATH = Path("Data/evals/safety/latest_adversarial_holdout_v3_baseline.json")
V4_PATH = Path("Data/evals/safety/latest_adversarial_holdout_v4_baseline.json")

V3_BEFORE = {
    "status": "needs_attention",
    "total_n": 122,
    "pass_count": 88,
    "fail_count": 34,
    "pass_rate": 0.721311,
    "unsafe_leakage_rate": 0.309091,
    "over_refusal_rate": 0.0,
    "source": "v3 baseline before generalized hardening, recorded before this pass",
}

CLAIM_BOUNDARY = (
    "Adversarial hardening reports internal safety-regression movement only. "
    "It does not prove clinical safety, real-world robustness, external review, "
    "or patient benefit."
)


def build_adversarial_hardening_report(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    v3_after = _read(V3_PATH)
    v4 = _read(V4_PATH)
    v4_summary = _summary(v4)
    weakest_categories = sorted(
        (
            {"category": category, "pass_rate": metrics.get("pass_rate"), "total_n": metrics.get("total_n")}
            for category, metrics in (v4.get("by_category") or {}).items()
        ),
        key=lambda row: float(row.get("pass_rate") or 0.0),
    )[:6]
    payload = {
        "schema_version": "adversarial_hardening_report_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention",
        "canonical_headline": {
            "artifact": str(V4_PATH).replace("\\", "/"),
            "evaluation_role": "fresh_internal_frozen_holdout_not_used_for_tuning",
            "pass_rate": v4_summary.get("pass_rate"),
            "unsafe_leakage_rate": v4_summary.get("unsafe_leakage_rate"),
            "not_solved": True,
            "why": (
                "The fresh frozen v4 holdout is the strongest current internal generalization signal. "
                "Tuning-used v3 movement is supporting evidence only."
            ),
        },
        "v3_before": V3_BEFORE,
        "v3_after": _summary(v3_after),
        "v4_fresh_holdout": v4_summary,
        "weakest_v4_categories": weakest_categories,
        "v3_delta": {
            "pass_rate_delta": round(float(v3_after.get("pass_rate", 0)) - V3_BEFORE["pass_rate"], 6),
            "unsafe_leakage_delta": round(float(v3_after.get("unsafe_leakage_rate", 0)) - V3_BEFORE["unsafe_leakage_rate"], 6),
            "over_refusal_delta": round(float(v3_after.get("over_refusal_rate", 0)) - V3_BEFORE["over_refusal_rate"], 6),
        },
        "generalized_fix_families": [
            "private identifier and external-forwarding requests",
            "prompt override / validator disabling / hidden prompt extraction",
            "cross-patient and cohort-level record access",
            "genetic-risk and VUS conclusion phrasing",
            "diagnosis/progression confirmation",
            "tumor-marker-to-recurrence/progression assumptions",
            "treatment start/stop/hold/restart/choice requests",
            "dosage schedule / mg / dose-adjustment requests",
            "patient-specific prognosis/survival estimates",
            "supplement or natural-cure replacement claims",
        ],
        "contamination_note": (
            "V3 was used for generalized hardening in this pass. V4 is the newer "
            "internal holdout baseline and should remain separate from future tuning."
        ),
        "reporting_policy": {
            "dashboard_primary": "v4_fresh_holdout",
            "v3_before_after_is_primary": False,
            "perfect_or_near_perfect_tuning_used_scores_are_independent_evidence": False,
            "external_author_eval_completed": False,
        },
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "total_n": payload.get("total_n"),
        "pass_count": payload.get("pass_count"),
        "fail_count": payload.get("fail_count"),
        "pass_rate": payload.get("pass_rate"),
        "unsafe_leakage_rate": payload.get("unsafe_leakage_rate"),
        "over_refusal_rate": payload.get("over_refusal_rate"),
        "safe_negative_control_pass_rate": payload.get("safe_negative_control_pass_rate"),
        "was_used_for_tuning": payload.get("was_used_for_tuning"),
    }


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["build_adversarial_hardening_report"]
