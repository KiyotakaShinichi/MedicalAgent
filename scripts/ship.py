"""Cross-platform NLCare ship gate.

Runs the same checks as ``make ship`` without requiring GNU Make. The script
stops on the first failed command and returns that command's exit code.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend-react"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    cwd: Path = ROOT
    env: dict[str, str] | None = None


def _npm_cmd(*args: str) -> list[str]:
    executable = "npm.cmd" if os.name == "nt" else "npm"
    return [executable, *args]


def _run(step: Step) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if step.env:
        env.update(step.env)
    print(f"\n[ship] {step.name}", flush=True)
    print(f"[ship] cwd={step.cwd}", flush=True)
    print(f"[ship] cmd={' '.join(step.command)}", flush=True)
    subprocess.run(step.command, cwd=step.cwd, env=env, check=True)


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
                "tests/test_xai_retraining_stability_audit.py",
                "tests/test_xai_rank_stability_audit.py",
                "tests/test_credible_route_latency_sample.py",
                "tests/test_automation_channel_drill.py",
                "tests/test_adversarial_v6_tuning_regression.py",
                "tests/test_unsafe_intent_mutation_dev_eval.py",
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
            name="OIDC browser PKCE readiness",
            command=[sys.executable, "scripts/run_oidc_browser_pkce_readiness.py"],
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
            name="Signed localhost automation channel drill",
            command=[sys.executable, "scripts/run_automation_channel_drill.py"],
        ),
        Step(
            name="Synthetic n8n and MailHog staging readiness",
            command=[sys.executable, "scripts/run_synthetic_automation_staging_readiness.py"],
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
    for step in steps:
        try:
            _run(step)
        except subprocess.CalledProcessError as exc:
            print(f"\n[ship] FAILED: {step.name} exited {exc.returncode}", file=sys.stderr, flush=True)
            return int(exc.returncode or 1)
    print("\n[ship] PASSED: all gates green", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
