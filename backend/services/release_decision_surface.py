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
    {"id": "frozen_adversarial_v6", "tier": "warning", "owner": "AI safety", "path": "Data/evals/safety/latest_adversarial_holdout_v6_baseline.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "rag_governance_tradeoff", "tier": "warning", "owner": "RAG", "path": "Data/evals/rag/latest_rag_governance_tradeoff.json", "status_path": ("status",), "accepted": {"acceptable"}},
    {"id": "live_rag_internal", "tier": "warning", "owner": "RAG", "path": "Data/evals/rag/latest_live_rag_eval.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "route_latency", "tier": "warning", "owner": "SWE/ops", "path": "Data/evals/ops/latest_route_latency_budget.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "simple_baseline_superiority", "tier": "warning", "owner": "MLE", "path": "Data/evals/models/latest_synthetic_simple_baseline_audit.json", "status_path": ("status",), "accepted": {"acceptable"}},
    {"id": "synthetic_v2_stability", "tier": "warning", "owner": "MLE", "path": "Data/evals/models/latest_synthetic_v2_model_stability.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "xai_fidelity", "tier": "warning", "owner": "XAI", "path": "Data/evals/models/latest_xai_fidelity_audit.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "dependency_reproducibility", "tier": "warning", "owner": "SWE/ops", "path": "Data/evals/ops/latest_dependency_lock_audit.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "durable_automation_worker", "tier": "warning", "owner": "automation", "path": "Data/evals/ops/latest_durable_automation_worker_eval.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "automation_reliability", "tier": "warning", "owner": "automation", "path": "Data/evals/ops/latest_automation_reliability_dossier.json", "status_path": ("status",), "accepted": {"strong"}},
    {"id": "deployment_profile", "tier": "warning", "owner": "deployment", "path": "Data/evals/ops/latest_deployment_profile_validation.json", "status_path": ("status",), "accepted": {"strong"}},
    {"id": "deployment_recovery_local", "tier": "warning", "owner": "deployment", "path": "Data/evals/ops/latest_deployment_recovery_drill.json", "status_path": ("status",), "accepted": {"strong_local_only"}},
    {"id": "benchmark_freshness", "tier": "warning", "owner": "SWE/ops", "path": "Data/evals/benchmark/latest_benchmark_summary.json", "status_path": ("status",), "accepted": {"acceptable", "strong"}},
    {"id": "finetune_governance", "tier": "informational", "owner": "fine-tuning", "path": "Data/evals/models/latest_finetune_governance.json", "status_path": ("status",), "accepted": set()},
    {"id": "finetune_promotion", "tier": "informational", "owner": "fine-tuning", "path": "Data/evals/models/latest_finetune_promotion_gate.json", "status_path": ("status",), "accepted": set()},
    {"id": "ispy2_external_stress", "tier": "informational", "owner": "MLE", "path": "Data/evals/models/latest_ispy2_tcia_external_stress.json", "status_path": ("status",), "accepted": set()},
    {"id": "external_rag_holdout", "tier": "informational", "owner": "external review", "path": "Data/evals/rag/latest_rag_holdout_baseline_comparison.json", "status_path": ("status",), "accepted": set()},
    {"id": "goldset_adjudication", "tier": "informational", "owner": "external review", "path": "Data/evals/rag/latest_goldset_adjudication_readiness.json", "status_path": ("status",), "accepted": set()},
    {"id": "external_review_execution", "tier": "informational", "owner": "external review", "path": "Data/evals/governance/latest_external_review_execution_readiness.json", "status_path": ("status",), "accepted": set()},
)


def build_release_decision_surface(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    rows = [_evaluate(check) for check in CHECKS]
    blocker_failures = [row for row in rows if row["tier"] == "hard_blocker" and row["decision"] != "pass"]
    warnings = [row for row in rows if row["tier"] == "warning" and row["decision"] != "pass"]
    domains = _domain_summary(rows)
    payload = {
        "schema_version": "release_decision_surface_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blocker_failures else "needs_attention" if warnings else "acceptable",
        "engineering_release_decision": "BLOCK" if blocker_failures else "PROCEED_WITH_WARNINGS" if warnings else "PROCEED",
        "hard_blocker_count": len(blocker_failures),
        "warning_count": len(warnings),
        "checks": rows,
        "domains": domains,
        "domain_count": len(domains),
        "evidence_state_counts": _count_states(rows),
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
        return {
            **_public(check), "domain": _domain(check), "decision": "missing",
            "evidence_state": "missing", "observed_status": None, "key_metrics": {},
        }
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            **_public(check), "domain": _domain(check), "decision": "invalid",
            "evidence_state": "invalid", "observed_status": None, "error": str(exc), "key_metrics": {},
        }
    status = _get(artifact, check["status_path"])
    accepted = check["accepted"]
    decision = "informational" if check["tier"] == "informational" else "pass" if status in accepted else "attention"
    return {
        **_public(check), "domain": _domain(check), "decision": decision,
        "evidence_state": _evidence_state(check["id"], decision, artifact), "observed_status": status,
        "generated_at": artifact.get("generated_at"), "key_metrics": _key_metrics(check["id"], artifact),
    }


def _public(check: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in check.items() if key not in {"accepted", "status_path"}}


def _get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for part in path:
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _domain(check: dict[str, Any]) -> str:
    if check.get("domain"):
        return str(check["domain"])
    owner = str(check.get("owner") or "").lower()
    check_id = str(check.get("id") or "")
    if "fine" in owner:
        return "fine_tuning"
    if "automation" in owner:
        return "automation"
    if "deployment" in owner or check_id == "deployment_profile":
        return "deployment"
    if "medical" in owner or "external review" in owner:
        return "medical"
    if "mle" in owner or check_id.startswith("synthetic_") or check_id == "simple_baseline_superiority":
        return "mle"
    if "rag" in owner or "ai safety" in owner:
        return "aie"
    return "swe"


def _evidence_state(check_id: str, decision: str, artifact: dict[str, Any]) -> str:
    if decision in {"missing", "invalid"}:
        return decision
    if check_id in {"external_rag_holdout", "goldset_adjudication", "external_review_execution"}:
        completed = artifact.get("completed") is True or artifact.get("external_author_eval_completed") is True
        return "externally_completed" if completed else "external_blocked"
    if check_id in {"finetune_governance", "finetune_promotion"}:
        if artifact.get("model_trained") is True and artifact.get("decision") in {"HOLD", "REJECT", "PROMOTE"}:
            return "candidate_evaluated"
        return "scaffolded"
    if decision == "pass":
        return "verified_internal"
    if decision == "attention":
        return "needs_attention"
    return "informational"


def _count_states(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row["evidence_state"])
        counts[state] = counts.get(state, 0) + 1
    return counts


def _domain_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    domain_order = ("aie", "swe", "mle", "medical", "automation", "fine_tuning", "deployment")
    summary = []
    for domain in domain_order:
        members = [row for row in rows if row["domain"] == domain]
        if not members:
            state = "no_evidence_selected"
        elif any(row["tier"] == "hard_blocker" and row["decision"] != "pass" for row in members):
            state = "blocked"
        elif any(row["decision"] in {"attention", "missing", "invalid"} for row in members):
            state = "needs_attention"
        elif any(row["evidence_state"] == "external_blocked" for row in members):
            state = "external_evidence_incomplete"
        elif members and all(row["evidence_state"] in {"scaffolded", "external_blocked", "informational"} for row in members):
            state = "prepared_not_proven"
        else:
            state = "verified_internal_only"
        summary.append({
            "domain": domain,
            "state": state,
            "check_count": len(members),
            "verified_internal_count": sum(row["evidence_state"] == "verified_internal" for row in members),
            "needs_attention_count": sum(row["evidence_state"] == "needs_attention" for row in members),
            "scaffolded_count": sum(row["evidence_state"] == "scaffolded" for row in members),
            "external_blocked_count": sum(row["evidence_state"] == "external_blocked" for row in members),
        })
    return summary


def _key_metrics(check_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    summary = artifact.get("summary") or {}
    candidates = {
        "frozen_adversarial_v4": ("pass_rate", "unsafe_leakage_rate", "over_refusal_rate"),
        "frozen_adversarial_v5": ("pass_rate", "unsafe_leakage_rate", "over_refusal_rate"),
        "frozen_adversarial_v6": ("pass_rate", "unsafe_leakage_rate", "over_refusal_rate"),
        "rag_governance_tradeoff": ("improvement_proven_vs_bm25", "full_stack_recall_at_10", "bm25_recall_at_10"),
        "route_latency": ("production_ready", "insufficient_sample_count", "highest_observed_p95_ms"),
        "xai_fidelity": ("additivity_verifiable", "prediction_present_rate", "multiple_one_hot_feature_patient_rate"),
        "synthetic_v2_stability": ("synthetic_only", "clinical_validation"),
        "durable_automation_worker": ("control_pass_rate", "live_n8n_delivery_enabled", "live_delivery_test_completed"),
        "automation_reliability": ("check_count", "passed_count", "external_delivery_enabled_by_default", "real_emergency_coverage_claim"),
        "deployment_profile": ("profile", "strict_profile", "n_failed", "non_demo_auth_implemented", "deployment_capability"),
        "benchmark_freshness": ("critical_status", "issue_count", "benchmark_count"),
        "finetune_governance": ("model_trained", "readiness_state", "training_ready", "promotion_ready"),
        "finetune_promotion": ("decision", "promotion_scope", "behavior_improvement_statistically_proven"),
        "external_review_execution": ("completed", "clinical_validation"),
    }.get(check_id, ())
    return {key: artifact.get(key, summary.get(key)) for key in candidates}


def _headline_limitations() -> list[str]:
    return [
        "No real patient data, IRB approval, clinician sign-off, or clinical validation.",
        "Frozen adversarial generalization remains below target and is internally authored.",
        "Raw retrieval superiority over BM25 is not proven on the internal goldset.",
        "Synthetic ML metrics do not establish clinical performance or transfer.",
        "Fine-tuning is scaffolded only; no adapter or candidate generations exist.",
        "OIDC bearer-token verification exists, but no live provider, browser PKCE flow, or provider logout has been demonstrated.",
        "Automation delivery evidence is not human acknowledgement or emergency coverage.",
        "Latency and dependency reproducibility do not establish production readiness.",
    ]


__all__ = ["build_release_decision_surface"]
