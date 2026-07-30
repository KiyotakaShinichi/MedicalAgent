"""Cross-platform NLCare ship gate.

Runs the same checks as ``make ship`` without requiring GNU Make. The script
stops on the first failed command and returns that command's exit code.
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend-react"
SHIP_MANIFEST = ROOT / "Data" / "evals" / "ops" / "latest_ship_run.json"
DEFAULT_STEP_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    cwd: Path = ROOT
    env: dict[str, str] | None = None
    timeout_seconds: int | None = None


def _npm_cmd(*args: str) -> list[str]:
    executable = "npm.cmd" if os.name == "nt" else "npm"
    return [executable, *args]


def _effective_timeout(step: Step) -> int:
    if step.timeout_seconds is not None:
        return max(30, int(step.timeout_seconds))
    configured = os.getenv("NLCARE_SHIP_STEP_TIMEOUT_SECONDS")
    if configured:
        try:
            return max(30, int(configured))
        except ValueError:
            pass
    return DEFAULT_STEP_TIMEOUT_SECONDS


def _run(step: Step) -> dict[str, object]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if step.env:
        env.update(step.env)
    print(f"\n[ship] {step.name}", flush=True)
    print(f"[ship] cwd={step.cwd}", flush=True)
    print(f"[ship] cmd={' '.join(step.command)}", flush=True)
    timeout_seconds = _effective_timeout(step)
    started = time.perf_counter()
    subprocess.run(
        step.command,
        cwd=step.cwd,
        env=env,
        check=True,
        timeout=timeout_seconds,
    )
    elapsed = round(time.perf_counter() - started, 3)
    print(f"[ship] passed in {elapsed}s", flush=True)
    return {
        "name": step.name,
        "status": "passed",
        "duration_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "cwd": str(step.cwd.relative_to(ROOT) if step.cwd != ROOT else "."),
        "command": step.command,
    }


def _write_manifest(
    *,
    status: str,
    step_results: list[dict[str, object]],
    failed_step: str | None = None,
    failure_kind: str | None = None,
) -> None:
    SHIP_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "nlcare_ship_run_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "completed_step_count": len(step_results),
        "failed_step": failed_step,
        "failure_kind": failure_kind,
        "steps": step_results,
        "claim_boundary": (
            "This manifest records local engineering gate execution only. A passing ship run "
            "does not establish clinical validation, real-world safety, compliance, or "
            "production healthcare readiness."
        ),
    }
    SHIP_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    steps = [
        Step(
            name="Backend breast-monitoring integration tests",
            command=[sys.executable, "-m", "pytest", "tests/test_breast_monitoring.py", "-q"],
            env={
                "RAG_FORCE_SPARSE": "true",
                "ONCOTRACK_FAST_MODE": "true",
            },
        ),
        Step(
            name="Backend progressive-loading and notification reliability tests",
            command=[
                sys.executable,
                "-m",
                "pytest",
                "tests/test_patient_progressive_report.py",
                "tests/test_patient_report_enrichment_jobs.py",
                "tests/test_high_risk_conversation_alerts.py",
                "tests/test_n8n_webhook_dispatcher.py",
                "tests/test_n8n_automation_templates.py",
                "-q",
            ],
        ),
        Step(
            name="Cloud, data-platform, and managed-vector contract tests",
            command=[
                sys.executable,
                "-m",
                "pytest",
                "tests/test_managed_vector_store.py",
                "tests/test_data_platform_pipeline.py",
                "tests/test_cloud_infrastructure_readiness.py",
                "tests/test_vector_store_contract_eval.py",
                "tests/test_azure_search_index_admin.py",
                "tests/test_managed_vector_shadow_sync.py",
                "tests/test_managed_vector_shadow_comparison.py",
                "tests/test_data_platform_reliability_eval.py",
                "tests/test_ops_health_snapshot.py",
                "tests/test_release_decision_surface.py",
                "tests/test_constraint_aware_improvement_program.py",
                "tests/test_oidc_pkce.py",
                "tests/test_dependency_security_scan.py",
                "tests/test_software_supply_chain_evidence.py",
                "tests/test_synthetic_automation_staging_readiness.py",
                "tests/test_rag_degradation_resilience_eval.py",
                "tests/test_agent_execution_policy_eval.py",
                "tests/test_agent_execution_policy.py",
                "tests/test_agentic_orchestrator_and_verifier.py",
                "tests/test_synthetic_prediction_statistical_audit.py",
                "tests/test_patient_xai_readability_dossier.py",
                "tests/test_ship_runner.py",
                "tests/test_xai_retraining_stability_audit.py",
                "tests/test_xai_rank_stability_audit.py",
                "tests/test_credible_route_latency_sample.py",
                "tests/test_automation_channel_drill.py",
                "tests/test_adversarial_v6_tuning_regression.py",
                "tests/test_unsafe_intent_mutation_dev_eval.py",
                "tests/test_cross_domain_assurance_eval.py",
                "tests/test_senior_engineering_evidence.py",
                "tests/test_llm_usage_telemetry.py",
                "tests/test_finetune_promotion.py",
                "tests/test_finetune_semantic_contamination.py",
                "tests/test_finetune_hardening_assurance.py",
                "tests/test_rag_paired_statistical_comparison.py",
                "tests/test_xai_reliability_gate.py",
                "tests/test_patient_xai_envelope.py",
                "tests/test_evidence_maturity_matrix.py",
                "tests/test_credibility_gap_registry.py",
                "tests/test_rag_vector_runtime_cache.py",
                "tests/test_retrieval_runtime_cache_eval.py",
                "tests/test_provider_usage_reconciliation.py",
                "tests/test_finetune_contamination_adjudication.py",
                "tests/test_synthetic_model_perturbation_retrain_eval.py",
                "tests/test_synthetic_staging_resilience_dossier.py",
                "-q",
            ],
        ),
        Step(
            name="Frontend Vitest unit tests",
            command=_npm_cmd("run", "test"),
            cwd=FRONTEND,
        ),
        Step(
            name="Frontend Playwright smoke",
            command=_npm_cmd("run", "test:e2e", "--", "tests/e2e/smoke.spec.ts"),
            cwd=FRONTEND,
        ),
        Step(
            name="Frontend lint",
            command=_npm_cmd("run", "lint"),
            cwd=FRONTEND,
        ),
        Step(
            name="Frontend production build",
            command=_npm_cmd("run", "build"),
            cwd=FRONTEND,
        ),
        Step(
            name="Reproducible knowledge-base chunk materialization",
            command=[sys.executable, "scripts/ingest_knowledge_base.py", "--skip-index"],
        ),
        Step(
            name="Non-patient data-platform pipeline",
            command=[sys.executable, "scripts/run_data_platform_pipeline.py"],
        ),
        Step(
            name="Managed-vector contract evaluation",
            command=[sys.executable, "scripts/run_vector_store_contract_eval.py"],
        ),
        Step(
            name="Data-platform reliability drills",
            command=[sys.executable, "scripts/run_data_platform_reliability_eval.py"],
        ),
        Step(
            name="Azure AI Search index readiness",
            command=[sys.executable, "scripts/run_azure_search_index_readiness.py"],
        ),
        Step(
            name="Managed-vector shadow sync readiness",
            command=[sys.executable, "scripts/run_managed_vector_shadow_sync.py"],
        ),
        Step(
            name="Managed-vector frozen shadow comparison readiness",
            command=[sys.executable, "scripts/run_managed_vector_shadow_comparison.py"],
        ),
        Step(
            name="Azure compiled reference-infrastructure readiness",
            command=[sys.executable, "scripts/run_cloud_infrastructure_readiness.py"],
        ),
        Step(
            name="Patient enrichment background eval",
            command=[sys.executable, "scripts/run_patient_enrichment_background_eval.py"],
        ),
        Step(
            name="High-risk conversation alert eval",
            command=[sys.executable, "scripts/run_high_risk_conversation_alert_eval.py"],
        ),
        Step(
            name="ML logic/safety alignment audit",
            command=[sys.executable, "scripts/run_ml_logic_safety_alignment.py"],
        ),
        Step(
            name="Synthetic statistical sensitivity audit",
            command=[
                sys.executable,
                "scripts/run_synthetic_prediction_statistical_audit.py",
            ],
        ),
        Step(
            name="Patient XAI implementation-readability audit",
            command=[
                sys.executable,
                "scripts/run_patient_xai_readability_dossier.py",
            ],
        ),
        Step(
            name="OIDC browser PKCE readiness",
            command=[sys.executable, "scripts/run_oidc_browser_pkce_readiness.py"],
        ),
        Step(
            name="Deployment profile and fail-closed matrix",
            command=[sys.executable, "scripts/validate_deployment_env.py"],
        ),
        Step(
            name="Dependency security scan",
            command=[sys.executable, "scripts/run_dependency_security_scan.py"],
        ),
        Step(
            name="Lock-derived SBOM and repository secret scan",
            command=[sys.executable, "scripts/run_software_supply_chain_evidence.py"],
        ),
        Step(
            name="Development unsafe-intent mutation regression",
            command=[sys.executable, "scripts/run_unsafe_intent_mutation_dev_eval.py"],
        ),
        Step(
            name="Tuning-informed adversarial v6 regression",
            command=[sys.executable, "scripts/run_adversarial_v6_tuning_regression.py"],
        ),
        Step(
            name="Adversarial v6 contamination retrospective",
            command=[sys.executable, "scripts/run_adversarial_v6_retrospective.py"],
        ),
        Step(
            name="Synthetic XAI rank-stability audit",
            command=[sys.executable, "scripts/run_xai_rank_stability_audit.py"],
        ),
        Step(
            name="Synthetic XAI retraining-stability audit",
            command=[sys.executable, "scripts/run_xai_retraining_stability_audit.py"],
        ),
        Step(
            name="Synthetic XAI mechanical fidelity audit",
            command=[sys.executable, "scripts/run_xai_fidelity_audit.py"],
        ),
        Step(
            name="Fail-closed synthetic XAI presentation policy",
            command=[sys.executable, "scripts/run_xai_reliability_gate.py"],
        ),
        Step(
            name="Bounded agent execution-policy eval",
            command=[sys.executable, "scripts/run_agent_execution_policy_eval.py"],
        ),
        Step(
            name="Local RAG degradation resilience drill",
            command=[sys.executable, "scripts/run_rag_degradation_resilience_eval.py"],
        ),
        Step(
            name="Credible local route-latency sample",
            command=[sys.executable, "scripts/run_credible_route_latency_sample.py"],
        ),
        Step(
            name="Route-latency budget refresh",
            command=[sys.executable, "scripts/run_route_latency_budget.py"],
        ),
        Step(
            name="Token, cost, and stage-latency observability refresh",
            command=[sys.executable, "scripts/run_cost_latency_report.py"],
        ),
        Step(
            name="Retrieval runtime-cache regression evidence",
            command=[sys.executable, "scripts/run_retrieval_runtime_cache_eval.py"],
        ),
        Step(
            name="Provider-token reconciliation",
            command=[sys.executable, "scripts/run_provider_usage_reconciliation.py"],
        ),
        Step(
            name="Paired RAG statistical comparison",
            command=[sys.executable, "scripts/run_rag_paired_statistical_comparison.py"],
        ),
        Step(
            name="Signed localhost automation channel drill",
            command=[sys.executable, "scripts/run_automation_channel_drill.py"],
        ),
        Step(
            name="Synthetic n8n and MailHog staging readiness",
            command=[sys.executable, "scripts/run_synthetic_automation_staging_readiness.py"],
        ),
        Step(
            name="Fine-tune promotion evidence gate",
            command=[sys.executable, "scripts/run_finetune_promotion_gate.py"],
        ),
        Step(
            name="Fine-tune semantic contamination screen",
            command=[sys.executable, "scripts/run_finetune_semantic_contamination.py"],
        ),
        Step(
            name="Fine-tune contamination adjudication readiness",
            command=[
                sys.executable,
                "scripts/run_finetune_contamination_adjudication.py",
            ],
        ),
        Step(
            name="Synthetic ML perturbation and retraining stress",
            command=[
                sys.executable,
                "scripts/run_synthetic_model_perturbation_retrain_eval.py",
            ],
            timeout_seconds=900,
        ),
        Step(
            name="Synthetic staging resilience dossier",
            command=[
                sys.executable,
                "scripts/run_synthetic_staging_resilience_dossier.py",
            ],
        ),
        Step(
            name="Fine-tune hardening assurance",
            command=[sys.executable, "scripts/run_finetune_hardening_assurance.py"],
        ),
        Step(
            name="Cross-domain evidence maturity matrix",
            command=[sys.executable, "scripts/run_evidence_maturity_matrix.py"],
        ),
        Step(
            name="Canonical credibility-gap registry",
            command=[sys.executable, "scripts/run_credibility_gap_registry.py"],
        ),
        Step(
            name="Cross-domain composed assurance drill",
            command=[sys.executable, "scripts/run_cross_domain_assurance_eval.py"],
        ),
        Step(
            name="Senior engineering evidence dossier",
            command=[sys.executable, "scripts/run_senior_engineering_evidence.py"],
        ),
        Step(
            name="Canonical release decision surface",
            command=[sys.executable, "scripts/run_release_decision_surface.py"],
        ),
        Step(
            name="Constraint-aware cross-domain improvement program",
            command=[sys.executable, "scripts/run_constraint_aware_improvement_program.py"],
        ),
        Step(
            name="Consolidated benchmark registry",
            command=[sys.executable, "scripts/generate_benchmark_report.py"],
        ),
        Step(
            name="Evidence-backed service health snapshot",
            command=[sys.executable, "scripts/run_ops_health_snapshot.py"],
        ),
        Step(
            name="Focused release summary",
            command=[sys.executable, "scripts/run_focused_release_summary.py"],
        ),
        Step(
            name="Release artifact gate",
            command=[sys.executable, "scripts/run_release_gate.py"],
        ),
    ]
    step_results: list[dict[str, object]] = []
    for step in steps:
        try:
            step_results.append(_run(step))
        except subprocess.TimeoutExpired:
            timeout_seconds = _effective_timeout(step)
            step_results.append(
                {
                    "name": step.name,
                    "status": "timed_out",
                    "duration_seconds": timeout_seconds,
                    "timeout_seconds": timeout_seconds,
                    "cwd": str(step.cwd.relative_to(ROOT) if step.cwd != ROOT else "."),
                    "command": step.command,
                }
            )
            _write_manifest(
                status="failed",
                step_results=step_results,
                failed_step=step.name,
                failure_kind="timeout",
            )
            print(
                f"\n[ship] FAILED: {step.name} timed out after {timeout_seconds}s",
                file=sys.stderr,
                flush=True,
            )
            return 124
        except subprocess.CalledProcessError as exc:
            step_results.append(
                {
                    "name": step.name,
                    "status": "failed",
                    "duration_seconds": None,
                    "timeout_seconds": _effective_timeout(step),
                    "cwd": str(step.cwd.relative_to(ROOT) if step.cwd != ROOT else "."),
                    "command": step.command,
                    "exit_code": int(exc.returncode or 1),
                }
            )
            _write_manifest(
                status="failed",
                step_results=step_results,
                failed_step=step.name,
                failure_kind="nonzero_exit",
            )
            print(f"\n[ship] FAILED: {step.name} exited {exc.returncode}", file=sys.stderr, flush=True)
            return int(exc.returncode or 1)
    _write_manifest(status="passed", step_results=step_results)
    print("\n[ship] PASSED: all gates green", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
