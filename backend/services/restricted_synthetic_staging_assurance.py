"""Release evidence for restricted, synthetic-only staging boundaries."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "Data/evals/ops/latest_restricted_synthetic_staging_assurance.json"
DEPENDENCY_ARTIFACT = ROOT / "Data/evals/ops/latest_dependency_security_scan.json"
MINIMUM_TESTS = 20
TEST_PATHS = (
    "tests/test_restricted_synthetic_staging_hardening.py",
    "tests/test_deployment_profile_validation.py",
    "tests/test_container_runtime_hardening.py",
)
SOURCE_PATHS = (
    "backend/services/auth.py",
    "backend/services/oidc_pkce.py",
    "backend/services/deployment_profile_validation.py",
    "backend/services/synthetic_data_boundary.py",
    "backend/services/upload_security.py",
    "backend/services/patient_uploads.py",
    "backend/api/main.py",
    "Dockerfile",
    "scripts/container_entrypoint.py",
    "frontend-react/src/api/client.ts",
    "frontend-react/src/context/AuthContext.tsx",
    "frontend-react/src/context/authContextCore.ts",
    "docker-compose.prod.yml",
    "docker-compose.synthetic-staging.yml",
    *TEST_PATHS,
)


def _count(output: str, label: str) -> int:
    values = re.findall(rf"(?<!\w)(\d+)\s+{re.escape(label)}\b", output.lower())
    return max((int(value) for value in values), default=0)


def _dependency_state(root: Path) -> dict[str, Any]:
    path = root / DEPENDENCY_ARTIFACT.relative_to(ROOT)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"available": False, "acceptable": False, "status": "missing_or_invalid"}
    summary = payload.get("summary") or {}
    acceptable = (
        payload.get("status") in {"acceptable", "strong"}
        and int(summary.get("high_or_critical_count") or 0) == 0
        and int(summary.get("unaccepted_known_vulnerability_count") or 0) == 0
    )
    return {
        "available": True,
        "acceptable": acceptable,
        "status": payload.get("status"),
        "high_or_critical_count": int(summary.get("high_or_critical_count") or 0),
        "unaccepted_known_vulnerability_count": int(
            summary.get("unaccepted_known_vulnerability_count") or 0
        ),
        "artifact_path": str(path.relative_to(root)).replace("\\", "/"),
    }


def _digests(root: Path) -> dict[str, str]:
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
        if (root / relative).is_file()
    }


def build_report(
    *,
    return_code: int,
    output: str,
    duration_seconds: float,
    root: Path = ROOT,
    timed_out: bool = False,
) -> dict[str, Any]:
    passed = _count(output, "passed")
    failed = _count(output, "failed")
    errors = _count(output, "error") + _count(output, "errors")
    dependencies = _dependency_state(root)
    suite_passed = (
        return_code == 0
        and not timed_out
        and passed >= MINIMUM_TESTS
        and failed == 0
        and errors == 0
        and dependencies["acceptable"]
    )
    return {
        "schema_version": "restricted_synthetic_staging_assurance_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if suite_passed else "failed",
        "invariant": (
            "Restricted staging accepts synthetic data only, stores no raw bearer token, "
            "and releases no unscanned upload."
        ),
        "summary": {
            "passed_tests": passed,
            "failed_tests": failed,
            "error_tests": errors,
            "minimum_required_tests": MINIMUM_TESTS,
            "pytest_exit_code": return_code,
            "timed_out": timed_out,
            "duration_seconds": round(duration_seconds, 3),
        },
        "controls": {
            "browser_oidc_authorization_code_pkce_required": True,
            "raw_bearer_tokens_persisted": False,
            "browser_token_storage": "sessionStorage",
            "synthetic_data_boundary_at_api": True,
            "synthetic_patient_namespace_enforced": True,
            "uploads_disabled_by_default_in_strict_profiles": True,
            "strict_base64_and_magic_type_alignment": True,
            "quarantine_before_promotion": True,
            "external_scanner_required_when_strict_uploads_enabled": True,
            "scanner_failure_is_fail_closed": True,
            "leased_worker_wired_in_disposable_staging": True,
            "distroless_nonroot_backend_runtime": True,
            "shell_free_container_entrypoint": True,
        },
        "dependency_security": dependencies,
        "provenance": {
            "test_command": [
                sys.executable,
                "-m",
                "pytest",
                *TEST_PATHS,
                "-q",
            ],
            "source_sha256": _digests(root),
            "patient_content_retained": False,
        },
        "deployment_scope": "restricted_synthetic_staging_only",
        "clinical_validation": False,
        "patient_data_processed": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "This artifact verifies repeatable engineering controls for a restricted "
            "synthetic staging profile. It is not a penetration test, privacy or security "
            "certification, clinical validation, or production healthcare readiness."
        ),
    }


def run_assurance(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    root: Path = ROOT,
    timeout_seconds: int = 180,
    executor: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *TEST_PATHS, "-q"]
    started = time.perf_counter()
    timed_out = False
    try:
        completed = executor(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        return_code = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        return_code = 124
    report = build_report(
        return_code=return_code,
        output=output,
        duration_seconds=time.perf_counter() - started,
        root=root,
        timed_out=timed_out,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["DEFAULT_OUTPUT_PATH", "MINIMUM_TESTS", "build_report", "run_assurance"]
