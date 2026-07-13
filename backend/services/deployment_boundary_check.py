"""Deployment-boundary check.

Reports the current state of "production-shaped engineering polish"
without claiming healthcare production readiness, HIPAA compliance,
or clinical deployment readiness.

The artifact's ``status_label`` is **always**
``production_shaped_not_healthcare_production_ready`` regardless of
how many checks pass — that is the project's hard ceiling under the
constraints, and the test suite enforces it.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_PATH = Path("Data/evals/ops/latest_deployment_boundary_check.json")

FIXED_STATUS_LABEL = "production_shaped_not_healthcare_production_ready"


@dataclass(frozen=True)
class Check:
    name: str
    description: str

    def evaluate(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class FileExistsCheck(Check):
    path: str

    def evaluate(self) -> dict[str, Any]:
        exists = Path(self.path).exists()
        return {
            "name": self.name,
            "description": self.description,
            "ok": exists,
            "evidence_path": self.path,
        }


@dataclass(frozen=True)
class EnvVarCheck(Check):
    var: str
    must_equal: str | None = None
    must_not_equal: str | None = None

    def evaluate(self) -> dict[str, Any]:
        value = os.environ.get(self.var)
        ok = True
        reason = None
        if value is None:
            ok = False
            reason = "unset"
        if self.must_equal is not None and value != self.must_equal:
            ok = False
            reason = f"expected {self.must_equal!r}; got {value!r}"
        if self.must_not_equal is not None and value == self.must_not_equal:
            ok = False
            reason = f"must not equal {self.must_not_equal!r}"
        return {
            "name": self.name,
            "description": self.description,
            "ok": ok,
            "var": self.var,
            "value": value,
            "reason": reason,
        }


@dataclass(frozen=True)
class DemoCredentialsGatedCheck(Check):
    """Demo credentials should only be allowed when ENVIRONMENT != 'production'
    OR ALLOW_DEMO_AUTH explicitly == 'true'.  In production, demo auth
    must be off unless the operator has flagged it on."""

    def evaluate(self) -> dict[str, Any]:
        env = os.environ.get("ENVIRONMENT", "development")
        allow_demo = os.environ.get("ALLOW_DEMO_AUTH", "false").lower()
        ok = True
        reason = None
        if env == "production" and allow_demo != "true":
            # Production with demo auth off is the safe state.
            ok = True
        elif env == "production" and allow_demo == "true":
            # Production with demo auth on: explicitly flagged; warn.
            ok = False
            reason = "ENVIRONMENT=production AND ALLOW_DEMO_AUTH=true — demo creds will be live"
        return {
            "name": self.name,
            "description": self.description,
            "ok": ok,
            "environment": env,
            "allow_demo_auth": allow_demo,
            "reason": reason,
        }


CHECKS: tuple[Check, ...] = (
    FileExistsCheck(
        name="env_example_present",
        description=".env.example exists at the repo root",
        path=".env.example",
    ),
    FileExistsCheck(
        name="docker_compose_present",
        description="docker-compose.yml present for local startup",
        path="docker-compose.yml",
    ),
    FileExistsCheck(
        name="deployment_readiness_doc_present",
        description="docs/deployment_readiness.md exists",
        path="docs/deployment_readiness.md",
    ),
    FileExistsCheck(
        name="deployment_boundary_doc_present",
        description="docs/deployment_boundary.md exists (explicit production-shaped-not-healthcare framing)",
        path="docs/deployment_boundary.md",
    ),
    FileExistsCheck(
        name="health_check_doc_present",
        description="A health-check route or endpoint is documented somewhere",
        path="docs/monitoring.md",
    ),
    FileExistsCheck(
        name="local_smoke_command_present",
        description="A local-smoke / ship script is wired into the repo",
        path="scripts/ship.py",
    ),
    FileExistsCheck(
        name="security_controls_doc_present",
        description="docs/security_controls.md exists",
        path="docs/security_controls.md",
    ),
    DemoCredentialsGatedCheck(
        name="demo_credentials_gated_off_in_production",
        description="Demo credentials must be off when ENVIRONMENT=production unless explicitly flagged",
    ),
)


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    results = [check.evaluate() for check in CHECKS]
    passed = sum(1 for r in results if r["ok"])
    failed = [r for r in results if not r["ok"]]

    overall_status = (
        "strong" if not failed else
        "acceptable" if len(failed) <= 1 else "needs_attention"
    )

    return {
        "schema_version": "deployment_boundary_check_v1",
        "status": overall_status,
        "status_label": FIXED_STATUS_LABEL,
        "label": "deployment_boundary_check",
        "clinical_validation": False,
        "no_hipaa_compliance_claim": True,
        "no_clinical_deployment_claim": True,
        "claim_boundary": (
            f"Deployment boundary status is permanently fixed at "
            f"``{FIXED_STATUS_LABEL}``.  Engineering polish only; no HIPAA "
            "compliance claim, no clinical deployment claim, no production "
            "healthcare readiness, no clinical validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "n_checks": len(results),
        "n_passed": passed,
        "n_failed": len(failed),
        "checks": results,
        "failed_checks": failed,
        "what_this_does_not_certify": [
            "HIPAA compliance",
            "SOC 2 / HITRUST",
            "FDA / CE / regulatory clearance",
            "clinical safety",
            "real-world patient privacy",
            "production healthcare deployment",
        ],
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "CHECKS",
    "FIXED_STATUS_LABEL",
    "OUTPUT_PATH",
    "build_report",
    "write_report",
]
