"""Small reviewer-facing release summary built from canonical artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.run_release_gate import run_release_gate


OUTPUT_PATH = Path("Data/evals/governance/latest_focused_release_summary.json")


def _load(path: str) -> dict[str, Any]:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}


def _metric(payload: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def build_report() -> dict[str, Any]:
    gate = run_release_gate()
    write_contract = _load("Data/evals/agentic_tool_use/latest_confirmed_write_contract_eval.json")
    rag = _load("Data/evals/rag/latest_rag_baseline_comparison.json")
    route_policy = _load("Data/evals/rag/latest_route_aware_rag_policy_eval.json")
    adversarial = _load("Data/evals/safety/latest_adversarial_safety_regression.json")
    holdout = _load("Data/evals/safety/latest_adversarial_holdout_v4_baseline.json")
    holdout_v5 = _load("Data/evals/safety/latest_adversarial_holdout_v5_baseline.json")
    holdout_v6 = _load("Data/evals/safety/latest_adversarial_holdout_v6_baseline.json")
    load = _load("Data/evals/ops/latest_load_test_report.json")
    runtime = _load("Data/evals/ops/latest_runtime_quality_sentinel.json")
    deployment = _load("Data/evals/ops/latest_deployment_profile_validation.json")
    holdout_rag = _load("Data/evals/rag/latest_rag_holdout_baseline_comparison.json")
    route_latency = _load("Data/evals/ops/latest_route_latency_budget.json")
    xai = _load("Data/evals/models/latest_xai_fidelity_audit.json")
    dependencies = _load("Data/evals/ops/latest_dependency_lock_audit.json")
    automation = _load("Data/evals/ops/latest_durable_automation_worker_eval.json")
    live_rag = _load("Data/evals/rag/latest_live_rag_eval.json")
    ispy2 = _load("Data/evals/models/latest_ispy2_tcia_external_stress.json")
    finetune = _load("Data/evals/models/latest_finetune_governance.json")
    recovery = _load("Data/evals/ops/latest_deployment_recovery_drill.json")
    registry = _load("Data/evals/benchmark/latest_benchmark_summary.json")

    baseline_summary = rag.get("summary") or {}
    required_count = sum(1 for row in gate["artifacts"] if row["required"])
    optional_count = len(gate["artifacts"]) - required_count
    optional_issue_count = sum(1 for row in gate["artifacts"] if not row["required"] and row["issues"])
    original_pass_rate = (
        round(float(adversarial.get("pass_count", 0)) / float(adversarial.get("total_n", 1)), 4)
        if adversarial.get("total_n") else None
    )
    return {
        "schema_version": "focused_release_summary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "release_gate": {
            "status": gate["status"],
            "artifact_count": gate["artifact_count"],
            "hard_blocker_failure_count": gate["failure_count"],
            "required_artifact_count": required_count,
            "optional_artifact_count": optional_count,
            "optional_artifact_ratio": round(optional_count / max(1, len(gate["artifacts"])), 4),
            "optional_issue_count": optional_issue_count,
            "failures": [row for row in gate["artifacts"] if row["required"] and row["issues"]],
        },
        "core_evidence": {
            "confirmed_write_contract": {
                "status": write_contract.get("status"),
                "case_count": write_contract.get("case_count"),
                "pass_rate": write_contract.get("pass_rate"),
            },
            "rag": {
                "improvement_proven_vs_bm25": baseline_summary.get("improvement_proven_vs_bm25", False),
                "full_stack_recall_at_10": baseline_summary.get("full_stack_recall_at_10"),
                "bm25_recall_at_10": baseline_summary.get("bm25_recall_at_10"),
                "route_policy_promotion": route_policy.get("promotion_decision"),
                "route_policy_source_tier_correctness": _metric(route_policy, "route_aware_summary", "source_tier_correct"),
                "external_holdout_completed": holdout_rag.get("completed", False),
                "live_internal_case_count": _metric(live_rag, "summary", "case_count"),
                "live_internal_citation_precision": _metric(live_rag, "summary", "citation_precision"),
                "live_internal_claim_support_rate": _metric(live_rag, "summary", "claim_support_rate"),
                "live_internal_unsafe_answer_rate": _metric(live_rag, "summary", "unsafe_answer_rate"),
            },
            "adversarial_safety": {
                "original_internal_pass_rate": original_pass_rate,
                "holdout_v4_pass_rate": holdout.get("pass_rate"),
                "holdout_v4_unsafe_leakage_rate": holdout.get("unsafe_leakage_rate"),
                "holdout_v5_pass_rate": holdout_v5.get("pass_rate"),
                "holdout_v5_unsafe_leakage_rate": holdout_v5.get("unsafe_leakage_rate"),
                "holdout_v6_pass_rate": holdout_v6.get("pass_rate"),
                "holdout_v6_unsafe_leakage_rate": holdout_v6.get("unsafe_leakage_rate"),
                "holdout_v6_over_refusal_rate": holdout_v6.get("over_refusal_rate"),
                "canonical_headline": "Frozen internal v6 is the newest one-pass warning; it is not a tuning set.",
                "not_solved": True,
            },
            "xai_mechanics": {
                "status": xai.get("status"),
                "patient_explanation_n": xai.get("patient_explanation_n"),
                "additivity_verifiable": xai.get("additivity_verifiable"),
                "additivity_pass_rate": xai.get("additivity_pass_rate"),
                "causal_interpretation_allowed": xai.get("causal_interpretation_allowed", False),
            },
            "dependency_reproducibility": {
                "status": dependencies.get("status"),
                "transitive_lock_complete": dependencies.get("transitive_lock_complete"),
                "environment_matches_transitive_lock": dependencies.get("environment_matches_transitive_lock"),
                "portable_across_platforms": dependencies.get("portable_across_platforms", False),
                "vulnerability_scan_included": dependencies.get("vulnerability_scan_included", False),
            },
            "automation_durability": {
                "status": automation.get("status"),
                "control_pass_rate": automation.get("control_pass_rate"),
                "live_delivery_test_completed": automation.get("live_delivery_test_completed", False),
                "delivery_receipt_is_human_acknowledgement": False,
            },
            "latency": {
                "load_success_rate": _metric(load, "summary", "success_rate"),
                "load_p95_ms": _metric(load, "summary", "latency_ms", "p95"),
                "runtime_p95_ms": _metric(runtime, "summary", "latency_p95_ms"),
                "route_percentile_minimum_samples": route_latency.get("minimum_samples_for_percentile_credibility", 30),
                "routes_with_insufficient_samples": _metric(route_latency, "summary", "insufficient_sample_count"),
                "production_ready": False,
            },
            "deployment_profile": {
                "status": deployment.get("status"),
                "profile": deployment.get("profile"),
                "healthcare_production_ready": False,
            },
            "external_data_stress": {
                "status": ispy2.get("status"),
                "dataset": _metric(ispy2, "source", "dataset"),
                "joined_row_count": _metric(ispy2, "source", "joined_row_count"),
                "target_match_to_nlcare": _metric(ispy2, "task_boundary", "nlcare_target_match", default=False),
                "used_for_nlcare_training": ispy2.get("used_for_nlcare_training", False),
                "promotion_allowed": ispy2.get("promotion_allowed", False),
            },
            "finetuning": {
                "status": finetune.get("status"),
                "accepted_examples": _metric(finetune, "dataset", "accepted"),
                "training_ready": finetune.get("training_ready", False),
                "promotion_ready": finetune.get("promotion_ready", False),
                "failed_checks": _metric(finetune, "execution_readiness", "failed_checks", default=[]),
            },
            "recovery": {
                "status": recovery.get("status"),
                "local_restore_passed": recovery.get("passed", False),
                "postgres_restore_tested": recovery.get("postgres_restore_tested", False),
                "strict_profile_validated": recovery.get("strict_profile_validated", False),
            },
        },
        "canonical_evidence_paths": [
            "Data/evals/governance/latest_release_decision_surface.json",
            "Data/evals/rag/latest_rag_baseline_comparison.json",
            "Data/evals/rag/latest_live_rag_eval.json",
            "Data/evals/safety/latest_adversarial_holdout_v4_baseline.json",
            "Data/evals/safety/latest_adversarial_holdout_v5_baseline.json",
            "Data/evals/safety/latest_adversarial_holdout_v6_baseline.json",
            "Data/evals/models/latest_synthetic_simple_baseline_audit.json",
            "Data/evals/models/latest_synthetic_v2_model_stability.json",
            "Data/evals/models/latest_ispy2_tcia_external_stress.json",
            "Data/evals/models/latest_finetune_governance.json",
            "Data/evals/ops/latest_deployment_profile_validation.json",
            "Data/evals/ops/latest_deployment_recovery_drill.json",
        ],
        "registry_health": {
            "status": registry.get("status"),
            "issue_count": registry.get("issue_count"),
            "reviewer_instruction": "Use canonical_evidence_paths first; treat the remaining registry as supporting appendix evidence.",
        },
        "active_warnings": [
            "Full source-governed retrieval has not proven raw Recall@10 improvement over BM25.",
            "The post-hoc route-aware policy is held because it reduces source-tier correctness.",
            "The no-read external-author RAG holdout remains incomplete.",
            "All ML results remain synthetic-only engineering self-tests.",
            "The I-SPY2 pCR benchmark is target-mismatched and cannot validate or promote NLCare model heads.",
            "The 5,000-prompt perfect result is an internal tuning-used regression bank, not independent robustness evidence.",
            "No clinician, genetic counselor, or external-author review has been completed.",
            "Frozen internal v6 scored 0.5185 with unsafe leakage 0.5606; it is the newest safety warning and must not be tuned against in this pass.",
            "The exact transitive dependency lock covers CPython 3.14/Windows only and has no vulnerability scan.",
            "Automation durability controls are code-tested, but live n8n/channel reliability is untested.",
            "Route percentile labels require at least 30 samples; smaller samples are insufficient, not ideal.",
            "Most gate entries are supporting or informational, so the focused core view is the reviewer headline.",
        ],
        "what_this_release_can_claim": [
            "patient-confirmed, provenance-stamped, undoable record writes",
            "source-governed retrieval and explicit negative-result reporting",
            "synthetic-only temporal ML/MLE evaluation and promotion boundaries",
            "deployment-shaped engineering configuration with runtime validation",
            "mechanically verifiable synthetic-model log-odds explanation additivity",
            "database-leased redacted engineering automation with signed receipt persistence",
        ],
        "what_this_release_cannot_claim": [
            "clinical validation or clinician approval",
            "real-world patient safety or benefit",
            "retrieval superiority over BM25",
            "production healthcare readiness or compliance certification",
            "generalisation from synthetic ML to real patients",
        ],
        "claim_boundary": (
            "Concise engineering release view only. A passing gate means configured engineering "
            "checks passed; it does not establish clinical validation, patient benefit, clinician "
            "approval, or production healthcare readiness."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = ["OUTPUT_PATH", "build_report", "write_report"]
