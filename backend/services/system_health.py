from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import DATABASE_URL
from backend.services.artifact_manifest import freshness_status


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = "Data/evals/system/latest_system_health.json"


ARTIFACTS = {
    "benchmark_registry": "Data/evals/benchmark/latest_benchmark_summary.json",
    "rag_eval": "Data/evals/rag/latest_rag_benchmark.json",
    "safety_red_team": "Data/evals/safety/latest_safety_benchmark.json",
    "mle_readiness": "Data/mle_monitoring/latest_mle_readiness.json",
    "biomarker_feature_benchmark": "Data/mle_monitoring/biomarker_feature_benchmark.json",
    "public_biomarker_mapping": "Data/mle_monitoring/public_biomarker_mapping_readiness.json",
    "cbioportal_mapping": "Data/mle_monitoring/cbioportal_biomarker_schema_mapping.json",
    "clinical_safety_checklist": "Data/evals/safety/clinical_safety_review_checklist.json",
}


def build_system_health_report(
    *,
    db: Session | None = None,
    output_path: str = DEFAULT_OUTPUT_PATH,
    freshness_ttl_seconds: int = 24 * 60 * 60,
) -> dict[str, Any]:
    db_status = _database_status(db)
    artifact_rows = [_artifact_status(name, path, freshness_ttl_seconds) for name, path in ARTIFACTS.items()]
    dependency_rows = _dependency_status()
    frontend_status = _frontend_status()
    issue_rows = _collect_issues(db_status, artifact_rows, dependency_rows, frontend_status)
    report = {
        "schema_version": "system_health_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "healthy" if not issue_rows else "needs_attention",
        "backend": {
            "python_version": sys.version.split()[0],
            "database_url_kind": _database_kind(DATABASE_URL),
            "database": db_status,
        },
        "environment": {
            "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
            "rag_judge_enabled": str(os.environ.get("ONCOTRACK_RAG_JUDGE", "")).lower() in {"on", "true", "1"},
            "cors_origins_configured": bool(
                os.environ.get("NLCARE_CORS_ORIGINS") or os.environ.get("ONCOTRACK_CORS_ORIGINS")
            ),
        },
        "dependencies": dependency_rows,
        "artifacts": artifact_rows,
        "frontend": frontend_status,
        "issues": issue_rows,
        "next_actions": _next_actions(issue_rows),
        "claim_boundary": (
            "System health checks operational readiness for the engineering demo. Passing health checks do not "
            "establish clinical safety, HIPAA readiness, or production availability."
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_system_health_report(db: Session | None = None, output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["database_live"] = _database_status(db)
            return payload
        except Exception:
            pass
    return build_system_health_report(db=db, output_path=output_path)


def _database_status(db: Session | None) -> dict[str, Any]:
    if db is None:
        return {"status": "unknown", "message": "No database session provided."}
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "Database responded to SELECT 1."}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:220]}


def _dependency_status() -> list[dict[str, Any]]:
    packages = {
        "fastapi": "API serving",
        "sqlalchemy": "database ORM",
        "sentence_transformers": "dense retrieval",
        "faiss": "dense vector index",
        "rank_bm25": "sparse BM25 retrieval",
        "sklearn": "classical ML baselines",
        "torch": "optional imaging/deep-learning baselines",
        "shap": "optional explainability",
    }
    rows = []
    for package, purpose in packages.items():
        rows.append({
            "package": package,
            "purpose": purpose,
            "available": importlib.util.find_spec(package) is not None,
        })
    return rows


def _artifact_status(name: str, path: str, ttl_seconds: int) -> dict[str, Any]:
    file_path = ROOT_DIR / path
    if not file_path.exists():
        return {"name": name, "path": path, "exists": False, "freshness": "missing", "status": "missing"}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"name": name, "path": path, "exists": True, "freshness": "unknown", "status": "error", "error": str(exc)[:160]}
    generated_at = payload.get("generated_at") or (payload.get("artifact_freshness") or {}).get("generated_at")
    artifact_ttl = (payload.get("artifact_freshness") or {}).get("ttl_seconds") or ttl_seconds
    return {
        "name": name,
        "path": path,
        "exists": True,
        "status": str(payload.get("status") or (payload.get("summary") or {}).get("status") or "available"),
        "freshness": freshness_status(generated_at, int(artifact_ttl)) if generated_at else "unknown",
        "generated_at": generated_at,
    }


def _frontend_status() -> dict[str, Any]:
    dist_index = ROOT_DIR / "frontend-react" / "dist" / "index.html"
    package_json = ROOT_DIR / "frontend-react" / "package.json"
    return {
        "react_app_present": package_json.exists(),
        "production_build_present": dist_index.exists(),
        "dist_index": str(dist_index.relative_to(ROOT_DIR)).replace("\\", "/") if dist_index.exists() else None,
    }


def _collect_issues(
    db_status: dict[str, Any],
    artifact_rows: list[dict[str, Any]],
    dependency_rows: list[dict[str, Any]],
    frontend_status: dict[str, Any],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if db_status.get("status") != "ok":
        issues.append({"area": "database", "severity": "critical", "message": db_status.get("message", "Database unavailable.")})
    for artifact in artifact_rows:
        if artifact.get("status") in {"missing", "error"}:
            issues.append({"area": "artifact", "severity": "warning", "message": f"{artifact['name']} is {artifact['status']}."})
        elif artifact.get("freshness") == "stale":
            issues.append({"area": "artifact", "severity": "info", "message": f"{artifact['name']} is stale."})
    for dependency in dependency_rows:
        if dependency["package"] in {"sentence_transformers", "faiss", "rank_bm25"} and not dependency["available"]:
            issues.append({"area": "dependency", "severity": "warning", "message": f"{dependency['package']} missing; RAG falls back or degrades."})
    if not frontend_status.get("production_build_present"):
        issues.append({"area": "frontend", "severity": "info", "message": "React production build is not present yet; run npm run build."})
    return issues


def _next_actions(issues: list[dict[str, str]]) -> list[str]:
    if not issues:
        return ["Keep benchmark artifacts fresh and run Playwright smoke tests before demos."]
    actions = []
    if any(issue["area"] == "database" for issue in issues):
        actions.append("Start FastAPI with the configured database and rerun /admin/system-health.")
    if any(issue["area"] == "artifact" for issue in issues):
        actions.append("Regenerate stale/missing eval artifacts with scripts/run_quality_gate.py or the Admin dashboard run buttons.")
    if any(issue["area"] == "dependency" for issue in issues):
        actions.append("Install optional dense retrieval/imaging dependencies only if the machine can support them.")
    if any(issue["area"] == "frontend" for issue in issues):
        actions.append("Run npm run build inside frontend-react before a polished demo.")
    return actions


def _database_kind(url: str) -> str:
    if url.startswith("sqlite"):
        return "sqlite_demo"
    if url.startswith("postgres"):
        return "postgres"
    return "configured"
