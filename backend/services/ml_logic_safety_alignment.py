"""Synthetic ML logic and safety alignment audit.

This artifact consolidates existing ML/MLE evals into one reviewer-facing
question: do the synthetic model heads behave logically under the project's
own safety contract?

It does not retrain models, change inference, or promote any model. The
expected result can be ``needs_attention``; that is useful because it shows
where synthetic ML remains weaker even when release gates pass.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_ml_logic_safety_alignment.json")

ARTIFACTS: dict[str, Path] = {
    "data_promotion_roadmap": Path("Data/evals/models/latest_data_promotion_roadmap.json"),
    "mle_promotion_gate": Path("Data/evals/models/latest_mle_promotion_gate.json"),
    "evidence_abstention_eval": Path("Data/evals/models/latest_evidence_abstention_eval.json"),
    "per_head_calibration": Path("Data/evals/models/latest_per_head_calibration.json"),
    "response_conformal_calibration": Path("Data/evals/models/latest_response_conformal_calibration.json"),
    "patient_temporal_cv": Path("Data/evals/models/latest_patient_temporal_cv.json"),
    "counterfactual_stability": Path("Data/evals/models/latest_counterfactual_stability.json"),
    "noisier_synthetic_v2_stress": Path("Data/evals/models/latest_noisier_synthetic_v2_stress.json"),
    "synthetic_prediction_statistical_audit": Path("Data/evals/models/latest_synthetic_prediction_statistical_audit.json"),
    "ml_coverage_risk_diagnostics": Path("Data/evals/models/latest_ml_coverage_risk_diagnostics.json"),
    "toxicity_review_target_v3": Path("Data/evals/models/latest_toxicity_review_target_v3.json"),
}

NON_PROMOTIONAL_POLICIES = {
    "monitor_only",
    "review_hint_only",
    "context_and_review_routing_only",
    "education_and_review_context_only",
    "timeline_context_only",
}

CLAIM_BOUNDARY = (
    "Synthetic ML logic/safety alignment audit only. It checks whether existing "
    "synthetic model artifacts obey non-diagnostic, non-promotional, evidence-"
    "sufficiency, uncertainty, and shortcut-risk boundaries. It is not clinical "
    "validation, not real-patient calibration, not treatment evidence, and not "
    "production healthcare readiness."
)


def build_ml_logic_safety_alignment(
    *,
    artifacts: dict[str, Path] | None = None,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    artifact_paths = artifacts or ARTIFACTS
    loaded = {name: _load(path) for name, path in artifact_paths.items()}

    checks = [
        _check_nonclinical_promotion_policy(loaded),
        _check_evidence_sufficiency_alignment(loaded),
        _check_uncertainty_and_calibration(loaded),
        _check_temporal_split_hygiene(loaded),
        _check_counterfactual_stability(loaded),
        _check_noise_stress_boundary(loaded),
        _check_shortcut_risk_boundaries(loaded),
        _check_statistical_audit_boundary(loaded),
        _check_coverage_risk_diagnostics(loaded),
        _check_toxicity_target_v3_boundary(loaded),
    ]
    summary = _summarize_checks(checks)
    status = "needs_attention" if summary["needs_attention_count"] else "acceptable"
    if summary["failed_count"]:
        status = "needs_attention"

    payload: dict[str, Any] = {
        "schema_version": "ml_logic_safety_alignment_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "synthetic_only": True,
        "healthcare_production_ready": False,
        "artifact_inputs": {
            name: {
                "path": str(path).replace("\\", "/"),
                "exists": bool(loaded[name].get("_exists")),
                "status": loaded[name].get("status") or loaded[name].get("decision"),
            }
            for name, path in artifact_paths.items()
        },
        "summary": summary,
        "checks": checks,
        "highest_leverage_ml_next_steps": _next_steps(checks),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _check_nonclinical_promotion_policy(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    roadmap = artifacts["data_promotion_roadmap"]
    promotion = artifacts["mle_promotion_gate"]
    heads = roadmap.get("model_heads") or []
    policies = [head.get("current_policy") for head in heads]
    all_non_promotional = bool(policies) and all(policy in NON_PROMOTIONAL_POLICIES for policy in policies)
    may_influence_treatment = _dig(roadmap, ["summary", "may_influence_treatment"])
    decision = promotion.get("decision")
    passed = all_non_promotional and may_influence_treatment is False and decision in {"HOLD", "REJECT"}
    return _check(
        name="nonclinical_promotion_policy",
        status="passed" if passed else "failed",
        passed=passed,
        evidence={
            "model_head_policies": policies,
            "may_influence_treatment": may_influence_treatment,
            "mle_promotion_gate_decision": decision,
        },
        recommendation=(
            "Keep all synthetic heads monitor-only or review-only. Promotion should remain blocked "
            "until external temporal validation and clinician-reviewed labels exist."
        ),
    )


def _check_evidence_sufficiency_alignment(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    abstention = artifacts["evidence_abstention_eval"]
    summary = abstention.get("summary") or {}
    abstention_rates = summary.get("abstention_rates_by_scenario") or {}
    covered_accuracy = summary.get("covered_accuracy_by_scenario") or {}
    full_accuracy = _as_float(covered_accuracy.get("full_data"))
    no_imaging_accuracy = _as_float(covered_accuracy.get("no_imaging"))
    no_imaging_abstention = _as_float(abstention_rates.get("no_imaging"))

    low_evidence = {
        name: _as_float(abstention_rates.get(name))
        for name in ("demographics_only", "symptoms_only", "cbc_pre_only")
    }
    low_evidence_passed = all((value is not None and value >= 0.95) for value in low_evidence.values())
    no_imaging_drop = (
        full_accuracy is not None
        and no_imaging_accuracy is not None
        and no_imaging_accuracy <= full_accuracy - 0.15
    )
    no_imaging_under_abstains = no_imaging_abstention is not None and no_imaging_abstention < 0.50
    needs_attention = no_imaging_drop and no_imaging_under_abstains
    status = "needs_attention" if needs_attention else ("passed" if low_evidence_passed else "failed")
    return _check(
        name="evidence_sufficiency_alignment",
        status=status,
        passed=low_evidence_passed and not needs_attention,
        evidence={
            "low_evidence_abstention_rates": low_evidence,
            "full_data_covered_accuracy": full_accuracy,
            "no_imaging_covered_accuracy": no_imaging_accuracy,
            "no_imaging_abstention_rate": no_imaging_abstention,
            "no_imaging_drop_detected": needs_attention,
        },
        recommendation=(
            "Tighten response-classification confidence when imaging evidence is absent. The safest "
            "next implementation is not a new model; it is an evidence policy that reports lower "
            "confidence or abstains when response-classification depends on missing imaging."
        ),
    )


def _check_uncertainty_and_calibration(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    calibration = artifacts["per_head_calibration"]
    conformal = artifacts["response_conformal_calibration"]
    classification_ece = _dig(calibration, ["heads", "response_classification", "ece"])
    adjusted_coverage = conformal.get("adjusted_coverage")
    nominal = conformal.get("nominal_coverage")
    passed = (
        calibration.get("status") in {"strong", "acceptable"}
        and conformal.get("status") in {"strong", "acceptable"}
    )
    return _check(
        name="uncertainty_and_calibration",
        status="passed" if passed else "needs_attention",
        passed=passed,
        evidence={
            "per_head_calibration_status": calibration.get("status"),
            "response_classification_ece": classification_ece,
            "conformal_status": conformal.get("status"),
            "nominal_coverage": nominal,
            "adjusted_coverage": adjusted_coverage,
        },
        recommendation=(
            "Continue showing probabilities as synthetic model confidence, not patient outcome "
            "probability. Keep conformal bands visible as synthetic interval checks only."
        ),
    )


def _check_temporal_split_hygiene(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    temporal = artifacts["patient_temporal_cv"]
    patient_cv = temporal.get("patient_level_temporal_cv") or {}
    overlap = int(patient_cv.get("patient_overlap_pairs") or 0)
    temporal_violations = int(patient_cv.get("temporal_violations") or 0)
    row_censoring_applied = patient_cv.get("row_temporal_censoring_applied") is True
    status = "passed" if overlap == 0 and temporal_violations == 0 and row_censoring_applied else "needs_attention"
    return _check(
        name="patient_level_temporal_split_hygiene",
        status=status,
        passed=status == "passed",
        evidence={
            "patient_overlap_pairs": overlap,
            "temporal_violations": temporal_violations,
            "row_temporal_censoring_applied": row_censoring_applied,
            "train_rows_censored_after_test_start": patient_cv.get("train_rows_censored_after_test_start"),
            "roc_auc_mean": patient_cv.get("roc_auc_mean"),
            "target": temporal.get("target"),
        },
        recommendation=(
            "Preserve patient grouping and row-date censoring. Keep this as synthetic protocol hygiene, "
            "not as proof that the toxicity target generalises to real clinical timelines."
        ),
    )


def _check_counterfactual_stability(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counterfactual = artifacts["counterfactual_stability"]
    unacceptable = _dig(counterfactual, ["summary", "unacceptable_flip_count"])
    passed = counterfactual.get("status") in {"strong", "acceptable"} and unacceptable == 0
    return _check(
        name="counterfactual_stability",
        status="passed" if passed else "needs_attention",
        passed=passed,
        evidence={
            "status": counterfactual.get("status"),
            "unacceptable_flip_count": unacceptable,
            "max_probability_delta": _dig(counterfactual, ["summary", "max_probability_delta"]),
            "max_response_score_delta": _dig(counterfactual, ["summary", "max_response_score_delta"]),
        },
        recommendation=(
            "Keep this as a brittleness check. Expand it with missing-imaging and symptom-reporting "
            "counterfactuals before using any model score prominently."
        ),
    )


def _check_noise_stress_boundary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stress = artifacts["noisier_synthetic_v2_stress"]
    per_noise = stress.get("per_noise_type") or []
    leakage_tripwires = [
        row.get("noise_type")
        for row in per_noise
        if row.get("leakage_status") != "no_leakage_tripwire_fired"
    ]
    passed = (
        stress.get("clinical_validation") is False
        and stress.get("global_promotion_decision") == "reject_or_hold"
        and not leakage_tripwires
    )
    return _check(
        name="noise_stress_boundary",
        status="passed" if passed else "needs_attention",
        passed=passed,
        evidence={
            "noise_type_count": len(per_noise),
            "leakage_tripwire_noise_types": leakage_tripwires,
            "global_promotion_decision": stress.get("global_promotion_decision"),
        },
        recommendation=(
            "Use noisier v2 as a stress surface only. Do not retrain or promote from it unless the "
            "same tests stay green under a frozen, pre-registered protocol."
        ),
    )


def _check_shortcut_risk_boundaries(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    calibration = artifacts["per_head_calibration"]
    roadmap = artifacts["data_promotion_roadmap"]
    toxicity_auroc = _dig(calibration, ["heads", "toxicity", "auroc"])
    toxicity_policy = None
    for head in roadmap.get("model_heads") or []:
        if head.get("head") == "toxicity_signal":
            toxicity_policy = head.get("current_policy")
            break
    high_toxicity_auc = isinstance(toxicity_auroc, (int, float)) and toxicity_auroc >= 0.98
    bounded = toxicity_policy == "review_hint_only"
    status = "needs_attention" if high_toxicity_auc else "passed"
    return _check(
        name="shortcut_risk_boundaries",
        status=status,
        passed=bounded and not high_toxicity_auc,
        evidence={
            "toxicity_auroc": toxicity_auroc,
            "toxicity_policy": toxicity_policy,
            "high_toxicity_auc_shortcut_warning": high_toxicity_auc,
            "bounded_to_review_hint_only": bounded,
        },
        recommendation=(
            "Keep toxicity as review-hint-only and make the high synthetic AUC visibly suspicious. "
            "A softer review-priority target is the right direction; do not present toxicity AUC as a flex."
        ),
    )


def _check_statistical_audit_boundary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    audit = artifacts["synthetic_prediction_statistical_audit"]
    passed = (
        audit.get("clinical_validation") is False
        and audit.get("synthetic_only") is True
        and audit.get("promotion_decision") == "hold_synthetic_only"
        and _dig(audit, ["patient_level_bootstrap", "replicates"], 0) >= 1000
    )
    return _check(
        name="statistical_audit_boundary",
        status="passed" if passed else "needs_attention",
        passed=passed,
        evidence={
            "total_n": audit.get("total_n"),
            "bootstrap_replicates": _dig(audit, ["patient_level_bootstrap", "replicates"]),
            "perturbation_seed_count": _dig(audit, ["controlled_outcome_perturbations", "seed_count"]),
            "promotion_decision": audit.get("promotion_decision"),
        },
        recommendation=(
            "Keep row-level exports and paired tests. Add patient-block bootstrap if the export later "
            "contains multiple rows per synthetic patient."
        ),
    )


def _check_coverage_risk_diagnostics(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    coverage = artifacts["ml_coverage_risk_diagnostics"]
    required = coverage.get("required_abstention_scenarios") or {}
    selective = coverage.get("selective_risk") or {}
    passed = (
        coverage.get("clinical_validation") is False
        and coverage.get("synthetic_only") is True
        and coverage.get("healthcare_production_ready") is False
        and coverage.get("promotion_decision") == "hold_synthetic_only"
        and coverage.get("scenario_count", 0) >= 8
        and required.get("all_required_scenarios_passed") is True
        and selective.get("point_count", 0) >= 5
    )
    return _check(
        name="coverage_risk_diagnostics",
        status="passed" if passed else "needs_attention",
        passed=passed,
        evidence={
            "status": coverage.get("status"),
            "scenario_count": coverage.get("scenario_count"),
            "minimum_required_abstention_rate": required.get("minimum_required_abstention_rate"),
            "selective_risk_point_count": selective.get("point_count"),
            "promotion_decision": coverage.get("promotion_decision"),
        },
        recommendation=(
            "Keep low-evidence scenarios abstained and show missing-modality reasons near patient-facing "
            "model cards. Treat selective-risk curves as synthetic engineering evidence only."
        ),
    )


def _check_toxicity_target_v3_boundary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    toxicity = artifacts["toxicity_review_target_v3"]
    shortcut = toxicity.get("shortcut_comparison") or {}
    recommendation = toxicity.get("recommendation") or {}
    passed = (
        toxicity.get("clinical_validation") is False
        and toxicity.get("synthetic_only") is True
        and toxicity.get("healthcare_production_ready") is False
        and recommendation.get("production_policy") == "review_hint_only"
        and recommendation.get("promotion_decision") == "hold_synthetic_only"
        and shortcut.get("legacy_rule_does_not_define_v3") is True
    )
    return _check(
        name="toxicity_target_v3_boundary",
        status="passed" if passed else "needs_attention",
        passed=passed,
        evidence={
            "status": toxicity.get("status"),
            "model_auroc": _dig(toxicity, ["model", "auroc"]),
            "legacy_rule_auroc_against_v3": shortcut.get("legacy_rule_auroc_against_v3"),
            "legacy_rule_accuracy_against_v3": shortcut.get("legacy_rule_accuracy_against_v3"),
            "legacy_rule_does_not_define_v3": shortcut.get("legacy_rule_does_not_define_v3"),
            "residual_shortcut_warning": shortcut.get("residual_shortcut_warning"),
            "production_policy": recommendation.get("production_policy"),
            "promotion_decision": recommendation.get("promotion_decision"),
        },
        recommendation=(
            "Use v3 as a review-priority target-design candidate only. It reduces legacy-rule dominance "
            "but still carries simulator shortcut warnings, so toxicity remains review-hint-only."
        ),
    )


def _summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for check in checks if check["status"] == "passed")
    needs_attention = sum(1 for check in checks if check["status"] == "needs_attention")
    failed = sum(1 for check in checks if check["status"] == "failed")
    return {
        "check_count": len(checks),
        "passed_count": passed,
        "needs_attention_count": needs_attention,
        "failed_count": failed,
        "logic_alignment_score": round(passed / len(checks), 4) if checks else 0.0,
        "known_attention_items": [check["name"] for check in checks if check["status"] != "passed"],
    }


def _next_steps(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "evidence_sufficiency_alignment": 1,
        "patient_level_temporal_split_hygiene": 2,
        "shortcut_risk_boundaries": 3,
    }
    items = [
        {
            "rank": priority.get(check["name"], 10),
            "from_check": check["name"],
            "action": check["recommendation"],
        }
        for check in checks
        if check["status"] != "passed"
    ]
    if not items:
        items.append({
            "rank": 1,
            "from_check": "external_evidence_gap",
            "action": "Next improvement requires external-author or real-cohort evidence; do not infer clinical validity from internal synthetic checks.",
        })
    return sorted(items, key=lambda row: row["rank"])


def _check(
    *,
    name: str,
    status: str,
    passed: bool,
    evidence: dict[str, Any],
    recommendation: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "passed": bool(passed),
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_exists": False, "status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["_exists"] = True
            return payload
    except json.JSONDecodeError:
        return {"_exists": True, "status": "invalid_json"}
    return {"_exists": True, "status": "invalid_shape"}


def _dig(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["DEFAULT_OUTPUT_PATH", "build_ml_logic_safety_alignment"]
