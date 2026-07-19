"""Compact reviewer-facing release decision surface over detailed artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/governance/latest_release_decision_surface.json")

CHECKS: tuple[dict[str, Any], ...] = (
    {"id": "critical_safety_regression", "tier": "hard_blocker", "owner": "AI safety", "path": "Data/evals/safety/latest_safety_benchmark.json", "status_path": ("summary", "status"), "accepted": {"passed"}},
    {"id": "medical_claim_boundary", "tier": "hard_blocker", "owner": "medical safety", "path": "Data/evals/safety/latest_medical_claim_boundary_eval.json", "status_path": ("status",), "accepted": {"strong", "acceptable"}},
    {"id": "training_leakage", "tier": "hard_blocker", "owner": "MLE", "path": "Data/evals/models/latest_leakage_audit.json", "status_path": ("status",), "accepted": {"passed", "strong", "acceptable"}},
    {"id": "portfolio_claim_safety", "tier": "hard_blocker", "owner": "governance", "path": "Data/evals/governance/latest_portfolio_claim_safety_check.json", "status_path": ("status",), "accepted": {"informational", "strong", "acceptable", "passed"}},
    {"id": "frozen_adversarial_v4", "tier": "warning", "owner": "AI safety", "path": "Data/evals/safety/latest_adversarial_holdout_v4_baseline.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "frozen_adversarial_v5", "tier": "warning", "owner": "AI safety", "path": "Data/evals/safety/latest_adversarial_holdout_v5_baseline.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "rag_governance_tradeoff", "tier": "warning", "owner": "RAG", "path": "Data/evals/rag/latest_rag_governance_tradeoff.json", "status_path": ("status",), "accepted": {"acceptable"}},
    {"id": "route_latency", "tier": "warning", "owner": "SWE/ops", "path": "Data/evals/ops/latest_route_latency_budget.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "simple_baseline_superiority", "tier": "warning", "owner": "MLE", "path": "Data/evals/models/latest_synthetic_simple_baseline_audit.json", "status_path": ("status",), "accepted": {"acceptable"}},
    {"id": "synthetic_v2_stability", "tier": "warning", "owner": "MLE", "path": "Data/evals/models/latest_synthetic_v2_model_stability.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "xai_fidelity", "tier": "warning", "owner": "XAI", "path": "Data/evals/models/latest_xai_fidelity_audit.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "dependency_reproducibility", "tier": "warning", "owner": "SWE/ops", "path": "Data/evals/ops/latest_dependency_lock_audit.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "external_rag_holdout", "tier": "informational", "owner": "external review", "path": "Data/evals/rag/latest_rag_holdout_baseline_comparison.json", "status_path": ("status",), "accepted": set()},
    {"id": "goldset_adjudication", "tier": "informational", "owner": "external review", "path": "Data/evals/rag/latest_goldset_adjudication_readiness.json", "status_path": ("status",), "accepted": set()},
)


def build_release_decision_surface(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    rows = [_evaluate(check) for check in CHECKS]
    blocker_failures = [row for row in rows if row["tier"] == "hard_blocker" and row["decision"] != "pass"]
    warnings = [row for row in rows if row["tier"] == "warning" and row["decision"] != "pass"]
    payload = {
        "schema_version": "release_decision_surface_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blocker_failures else "needs_attention" if warnings else "acceptable",
        "engineering_release_decision": "BLOCK" if blocker_failures else "PROCEED_WITH_WARNINGS" if warnings else "PROCEED",
        "hard_blocker_count": len(blocker_failures),
        "warning_count": len(warnings),
        "checks": rows,
        "headline_limitations": _headline_limitations(),
        "detailed_registry_remains_authoritative": True,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": "This is a compact engineering decision aid, not clinical approval, validation, or production healthcare readiness.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _evaluate(check: dict[str, Any]) -> dict[str, Any]:
    path = Path(check["path"])
    if not path.exists():
        return {**_public(check), "decision": "missing", "observed_status": None, "key_metrics": {}}
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {**_public(check), "decision": "invalid", "observed_status": None, "error": str(exc), "key_metrics": {}}
    status = _get(artifact, check["status_path"])
    accepted = check["accepted"]
    decision = "informational" if check["tier"] == "informational" else "pass" if status in accepted else "attention"
    return {
        **_public(check), "decision": decision, "observed_status": status,
        "generated_at": artifact.get("generated_at"), "key_metrics": _key_metrics(check["id"], artifact),
    }


def _public(check: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in check.items() if key not in {"accepted", "status_path"}}


def _get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for part in path:
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _key_metrics(check_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    summary = artifact.get("summary") or {}
    candidates = {
        "frozen_adversarial_v4": ("pass_rate", "unsafe_leakage_rate", "over_refusal_rate"),
        "frozen_adversarial_v5": ("pass_rate", "unsafe_leakage_rate", "over_refusal_rate"),
        "rag_governance_tradeoff": ("improvement_proven_vs_bm25", "full_stack_recall_at_10", "bm25_recall_at_10"),
        "route_latency": ("production_ready", "insufficient_sample_count", "highest_observed_p95_ms"),
        "xai_fidelity": ("additivity_verifiable", "prediction_present_rate", "multiple_one_hot_feature_patient_rate"),
        "synthetic_v2_stability": ("synthetic_only", "clinical_validation"),
    }.get(check_id, ())
    return {key: artifact.get(key, summary.get(key)) for key in candidates}


def _headline_limitations() -> list[str]:
    return [
        "No real patient data, IRB approval, clinician sign-off, or clinical validation.",
        "Frozen adversarial generalization remains below target and is internally authored.",
        "Raw retrieval superiority over BM25 is not proven on the internal goldset.",
        "Synthetic ML metrics do not establish clinical performance or transfer.",
        "Latency and dependency reproducibility do not establish production readiness.",
    ]


__all__ = ["build_release_decision_surface"]
