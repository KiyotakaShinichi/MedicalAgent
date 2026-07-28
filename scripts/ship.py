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
            name="Focused release summary",
            command=[sys.executable, "scripts/run_focused_release_summary.py"],
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
            name="Azure reference-infrastructure readiness",
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
