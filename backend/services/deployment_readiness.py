"""Deployment preflight checks for the engineering prototype.

These checks make deployment posture visible without claiming healthcare
production readiness. They focus on environment hygiene, container assets,
release-gate availability, and explicit clinical boundaries.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_deployment_readiness.json")
ROOT = Path(__file__).resolve().parents[2]


def build_deployment_readiness(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    checks = _checks(env_map)
    blockers = [check for check in checks if check["severity"] == "blocker" and not check["passed"]]
    warnings = [check for check in checks if check["severity"] == "warning" and not check["passed"]]
    payload = {
        "schema_version": "deployment_readiness_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if blockers or warnings else "acceptable",
        "headline_metric": f"{len(blockers)} blockers, {len(warnings)} warnings",
        "total_n": len(checks),
        "pass_count": sum(1 for check in checks if check["passed"]),
        "fail_count": len(blockers) + len(warnings),
        "skipped_count": 0,
        "checks": checks,
        "deployment_shaped": True,
        "healthcare_production_ready": False,
        "clinical_validation": False,
        "phi_compliance_review_completed": False,
        "external_review_completed": False,
        "allowed_claim": (
            "Deployment-shaped engineering prototype with preflight checks, "
            "health/readiness probes, Docker assets, and explicit boundaries."
        ),
        "blocked_claims": [
            "production healthcare deployment",
            "PHI-ready deployment",
            "clinically validated system",
            "safe for real patient care",
            "hospital/EHR integration-ready",
        ],
        "claim_boundary": (
            "Deployment readiness here covers software packaging and environment "
            "hygiene only. It does not establish clinical validation, PHI "
            "compliance, hospital readiness, or real-world safety."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _checks(env: dict[str, str]) -> list[dict[str, Any]]:
    environment = (env.get("ENVIRONMENT") or env.get("APP_ENV") or "development").strip().lower()
    allow_demo = _truthy(env.get("ALLOW_DEMO_AUTH", ""))
    cors = [origin.strip() for origin in env.get("ONCOTRACK_CORS_ORIGINS", "").split(",") if origin.strip()]
    database_url = env.get("DATABASE_URL", "")
    groq_key = env.get("GROQ_API_KEY", "")

    return [
        _check(
            "environment_declared",
            environment in {"development", "docker", "staging", "production", "prod", "test"},
            "warning",
            f"ENVIRONMENT/APP_ENV resolved to {environment!r}.",
            "Set ENVIRONMENT explicitly for non-local deployments.",
        ),
        _check(
            "demo_auth_disabled_outside_development",
            not (environment in {"staging", "production", "prod"} and allow_demo),
            "blocker",
            "Demo credentials must not be enabled in staging/production.",
            "Use real authentication or keep ENVIRONMENT=development for demos.",
        ),
        _check(
            "database_url_configured",
            bool(database_url) or environment in {"development", "test"},
            "blocker",
            "DATABASE_URL is required outside local development.",
            "Set DATABASE_URL to Postgres for deployment-shaped runs.",
        ),
        _check(
            "cors_origin_explicit_for_non_dev",
            bool(cors) or environment in {"development", "docker", "test"},
            "warning",
            "ONCOTRACK_CORS_ORIGINS should list exact frontend origins outside local demos.",
            "Set ONCOTRACK_CORS_ORIGINS=https://your-frontend.example.",
        ),
        _check(
            "no_placeholder_llm_key_for_remote_mode",
            bool(groq_key.strip()) and not groq_key.lower().startswith("replace_with"),
            "warning",
            "GROQ_API_KEY appears unset or placeholder; deterministic/local paths may still work.",
            "Set a real key for live API RAG demos, or document deterministic-only mode.",
        ),
        _check(
            "dockerfile_present",
            (ROOT / "Dockerfile").exists(),
            "blocker",
            "Backend Dockerfile exists.",
            "Restore Dockerfile before container deployment.",
        ),
        _check(
            "frontend_production_dockerfile_present",
            (ROOT / "frontend-react" / "Dockerfile").exists(),
            "warning",
            "Production-style frontend Dockerfile exists.",
            "Add frontend-react/Dockerfile for static Vite serving.",
        ),
        _check(
            "prod_compose_present",
            (ROOT / "docker-compose.prod.yml").exists(),
            "warning",
            "Production-shaped Compose file exists.",
            "Add docker-compose.prod.yml for non-dev container smoke testing.",
        ),
        _check(
            "release_gate_artifact_present",
            (ROOT / "Data" / "evals" / "governance" / "latest_release_gate_explanation.json").exists(),
            "blocker",
            "Release-gate explanation artifact exists.",
            "Run python scripts/run_release_gate.py.",
        ),
        _check(
            "production_boundary_artifact_present",
            (ROOT / "Data" / "evals" / "governance" / "latest_production_readiness_boundary.json").exists(),
            "blocker",
            "Production boundary artifact exists and blocks healthcare-production claims.",
            "Run python scripts/run_production_readiness_boundary.py.",
        ),
    ]


def _check(name: str, passed: bool, severity: str, evidence: str, remediation: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
        "remediation": remediation,
    }


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["build_deployment_readiness"]
