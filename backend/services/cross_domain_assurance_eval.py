"""Disposable cross-domain assurance drills for NLCare.

The individual controls already have focused tests. This module composes them
to catch disagreements at service boundaries without performing a patient
write, network delivery, managed-cloud request, or clinical action.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from backend.services.agent_execution_policy import (
    build_confirmation_contract,
    enforce_agent_execution_policy,
)
from backend.services.automation_fault_injection_eval import (
    build_automation_fault_injection_eval,
)
from backend.services.bounded_agentic_workflow import plan_patient_agent_workflow
from backend.services.data_platform_reliability_eval import (
    build_data_platform_reliability_eval,
)
from backend.services.deployment_recovery_drill import run_local_recovery_drill
from backend.services.rag_degradation_resilience_eval import (
    build_rag_degradation_resilience_eval,
)
from backend.services.trace_envelope_v2 import (
    build_trace_envelope_v2,
    validate_trace_envelope_v2,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = Path("Data/evals/governance/latest_cross_domain_assurance_eval.json")
RAG_BASELINE_PATH = Path("Data/evals/rag/latest_rag_baseline_comparison.json")
ML_AUDIT_PATH = Path("Data/evals/models/latest_synthetic_prediction_statistical_audit.json")
FOCUSED_RELEASE_PATH = Path("Data/evals/governance/latest_focused_release_summary.json")


def build_cross_domain_assurance_eval(
    *,
    root_dir: str | Path = ROOT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Run composed, offline engineering assurance scenarios."""

    root = Path(root_dir)
    scenarios: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="nlcare-cross-domain-") as temporary:
        temp = Path(temporary)
        issued = datetime.now(timezone.utc)
        write_plan = plan_patient_agent_workflow("Log nausea severity 6/10 today")
        action_payload = {"symptom": "nausea", "severity": 6}
        contract = build_confirmation_contract(
            write_plan,
            patient_scope_id="synthetic-assurance-patient-a",
            action_payload=action_payload,
            now=issued,
            ttl_seconds=60,
            confirmation_id="cross-domain-confirmation",
        )

        valid_execution = enforce_agent_execution_policy(
            write_plan,
            confirmed_by_user=True,
            patient_scope_id="synthetic-assurance-patient-a",
            action_payload=action_payload,
            confirmation_contract=contract,
            require_bound_confirmation=True,
            now=issued + timedelta(seconds=1),
        )
        scenarios.append(
            _scenario(
                "patient_scoped_write_confirmation_allows_only_bound_action",
                valid_execution["decision"] == "allow"
                and valid_execution["confirmation_validation"]["valid"] is True
                and valid_execution["clinical_authority_allowed"] is False,
                {
                    "decision": valid_execution["decision"],
                    "effective_tools": valid_execution["effective_tools"],
                    "clinical_authority_allowed": False,
                },
            )
        )

        substituted_execution = enforce_agent_execution_policy(
            write_plan,
            confirmed_by_user=True,
            patient_scope_id="synthetic-assurance-patient-a",
            action_payload={"symptom": "nausea", "severity": 9},
            confirmation_contract=contract,
            require_bound_confirmation=True,
            now=issued + timedelta(seconds=1),
        )
        scenarios.append(
            _scenario(
                "confirmation_payload_substitution_fails_closed",
                substituted_execution["decision"] == "block"
                and "confirmation_payload_mismatch"
                in substituted_execution["confirmation_validation"]["issues"],
                {
                    "decision": substituted_execution["decision"],
                    "issues": substituted_execution["confirmation_validation"]["issues"],
                },
            )
        )

        replayed_execution = enforce_agent_execution_policy(
            write_plan,
            confirmed_by_user=True,
            patient_scope_id="synthetic-assurance-patient-a",
            action_payload=action_payload,
            confirmation_contract=contract,
            consumed_confirmation_ids={"cross-domain-confirmation"},
            require_bound_confirmation=True,
            now=issued + timedelta(seconds=1),
        )
        scenarios.append(
            _scenario(
                "consumed_confirmation_replay_fails_closed",
                replayed_execution["decision"] == "block"
                and "confirmation_replayed"
                in replayed_execution["confirmation_validation"]["issues"],
                {
                    "decision": replayed_execution["decision"],
                    "issues": replayed_execution["confirmation_validation"]["issues"],
                },
            )
        )

        trace = build_trace_envelope_v2(
            {
                "intent": "structured_update",
                "safety": {"level": "low", "scope": "record_capture"},
                "pipeline_trace": {"terminal_step": "verified_write"},
                "cache": {"status": "not_cacheable", "cacheable": False},
            },
            patient_id="synthetic-assurance-patient-a",
            route="patient_chat",
            latency_ms={"total": 12.0},
            correlation_id="cross-domain-assurance-trace",
        )
        trace_valid, trace_issues = validate_trace_envelope_v2(trace)
        poisoned_trace = {
            **trace,
            "patient_id": "synthetic-assurance-patient-a",
            "private_chain_of_thought": "must never persist",
        }
        poison_valid, poison_issues = validate_trace_envelope_v2(poisoned_trace)
        scenarios.append(
            _scenario(
                "trace_is_redacted_and_poisoned_trace_is_rejected",
                trace_valid
                and not trace_issues
                and not poison_valid
                and "forbidden_key:patient_id" in poison_issues
                and "forbidden_key:private_chain_of_thought" in poison_issues,
                {
                    "clean_trace_valid": trace_valid,
                    "poison_issues": poison_issues,
                    "correlation_id_preserved": trace["correlation_id"]
                    == "cross-domain-assurance-trace",
                },
            )
        )

        automation = build_automation_fault_injection_eval(
            temp / "automation_fault_injection.json"
        )
        scenarios.append(
            _report_scenario(
                "automation_idempotency_leases_signatures_and_receipts",
                automation,
                automation.get("passed_count") == automation.get("scenario_count")
                and automation.get("external_delivery_performed") is False
                and automation.get("human_acknowledgement_proven") is False,
            )
        )

        rag = build_rag_degradation_resilience_eval(
            temp / "rag_degradation_resilience.json"
        )
        scenarios.append(
            _report_scenario(
                "rag_corruption_staleness_and_sparse_fallback",
                rag,
                rag.get("failed_count") == 0
                and rag.get("managed_network_request_performed") is False
                and rag.get("retrieval_improvement_proven") is False,
            )
        )

        data = build_data_platform_reliability_eval(
            root_dir=root,
            output_path=temp / "data_platform_reliability.json",
        )
        scenarios.append(
            _report_scenario(
                "data_replay_quarantine_migration_delete_and_scale",
                data,
                data.get("failed") == 0
                and data.get("patient_data_processed") is False
                and _partition_scale_replay_is_consistent(data),
            )
        )

        recovery = run_local_recovery_drill(temp / "deployment_recovery.json")
        scenarios.append(
            _report_scenario(
                "local_backup_restore_preserves_synthetic_content",
                recovery,
                recovery.get("passed") is True
                and recovery.get("contains_patient_data") is False
                and recovery.get("postgres_restore_tested") is False,
            )
        )

        promotion = _promotion_boundaries(root)
        scenarios.append(
            _scenario(
                "weak_evidence_cannot_promote_rag_or_synthetic_ml",
                promotion["passed"],
                promotion,
            )
        )

        warning_visibility = _warning_visibility(root)
        scenarios.append(
            _scenario(
                "negative_results_remain_visible_on_release_surface",
                warning_visibility["passed"],
                warning_visibility,
            )
        )

    passed = sum(int(row["passed"]) for row in scenarios)
    report = {
        "schema_version": "nlcare_cross_domain_assurance_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong_internal_assurance"
        if passed == len(scenarios)
        else "needs_attention",
        "scenario_count": len(scenarios),
        "passed_count": passed,
        "failed_count": len(scenarios) - passed,
        "pass_rate": round(passed / len(scenarios), 6) if scenarios else 0.0,
        "scenarios": scenarios,
        "patient_write_performed": False,
        "external_network_request_performed": False,
        "managed_cloud_operation_performed": False,
        "clinical_action_automated": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "independent_reviewer_completed": False,
        "claim_boundary": (
            "This artifact composes disposable internal software drills. It improves "
            "change-safety evidence but is not independent validation, clinical "
            "evidence, managed-cloud reliability proof, or production healthcare readiness."
        ),
    }
    output = _resolve(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _promotion_boundaries(root: Path) -> dict[str, Any]:
    rag = _read_json(_resolve(root, RAG_BASELINE_PATH))
    ml = _read_json(_resolve(root, ML_AUDIT_PATH))
    rag_proven = rag.get("summary", {}).get("improvement_proven_vs_bm25")
    ml_promotion = ml.get("promotion_decision")
    passed = (
        rag_proven is False
        and ml_promotion == "hold_synthetic_only"
        and rag.get("clinical_validation") is False
        and ml.get("clinical_validation") is False
    )
    return {
        "passed": passed,
        "rag_improvement_proven_vs_bm25": rag_proven,
        "ml_promotion_decision": ml_promotion,
        "clinical_validation": False,
    }


def _warning_visibility(root: Path) -> dict[str, Any]:
    release = _read_json(_resolve(root, FOCUSED_RELEASE_PATH))
    warnings = [str(item) for item in release.get("active_warnings") or []]
    joined = " ".join(warnings).lower()
    passed = (
        "recall@10 improvement over bm25" in joined
        and "frozen internal v7 scored 0.6761" in joined
        and "all ml results remain synthetic-only" in joined
    )
    return {
        "passed": passed,
        "warning_count": len(warnings),
        "rag_negative_visible": "recall@10 improvement over bm25" in joined,
        "frozen_adversarial_warning_visible": "frozen internal v7 scored 0.6761"
        in joined,
        "synthetic_ml_boundary_visible": "all ml results remain synthetic-only"
        in joined,
    }


def _partition_scale_replay_is_consistent(report: dict[str, Any]) -> bool:
    replay = report.get("partition_scale_replay") or {}
    base_count = replay.get("base_record_count")
    multiplier = replay.get("scale_multiplier")
    replayed_count = replay.get("replayed_record_count")
    unique_count = replay.get("unique_record_ids")
    return bool(
        replay.get("deterministic") is True
        and isinstance(base_count, int)
        and base_count > 0
        and base_count == report.get("gold_record_count")
        and isinstance(multiplier, int)
        and multiplier > 0
        and replayed_count == base_count * multiplier
        and unique_count == replayed_count
    )


def _report_scenario(
    scenario_id: str,
    report: dict[str, Any],
    condition: bool,
) -> dict[str, Any]:
    return _scenario(
        scenario_id,
        bool(condition) and report.get("clinical_validation") is False,
        {
            "source_schema_version": report.get("schema_version"),
            "source_status": report.get("status"),
            "clinical_validation": report.get("clinical_validation"),
            "healthcare_production_ready": report.get(
                "healthcare_production_ready"
            ),
        },
    )


def _scenario(
    scenario_id: str,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "passed": bool(passed),
        "evidence": evidence,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


__all__ = ["build_cross_domain_assurance_eval"]
