"""Build bounded container vulnerability evidence from a Trivy JSON result."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "nlcare-synthetic-staging-backend:latest"
DEFAULT_RAW_PATH = ROOT / "Data/evals/ops/nlcare_backend_container_scan.json"
DEFAULT_OUTPUT_PATH = ROOT / "Data/evals/ops/latest_container_security_scan.json"
DEFAULT_SBOM_PATH = ROOT / "Data/evals/ops/nlcare_backend_sbom.cdx.json"
SEVERITIES = ("UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL")


def summarize_trivy_result(
    raw: dict[str, Any],
    *,
    image_inspect: dict[str, Any] | None = None,
    scanner_version: str | None = None,
    supply_chain: dict[str, Any] | None = None,
    sbom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inspect = image_inspect or {}
    findings = [
        finding
        for result in raw.get("Results") or []
        for finding in result.get("Vulnerabilities") or []
        if isinstance(finding, dict)
    ]
    counts = Counter(str(row.get("Severity") or "UNKNOWN").upper() for row in findings)
    severity_counts = {severity.lower(): counts.get(severity, 0) for severity in SEVERITIES}
    fixable = [
        row for row in findings
        if str(row.get("Severity") or "").upper() in {"HIGH", "CRITICAL"}
        and bool(str(row.get("FixedVersion") or "").strip())
    ]
    high_or_critical = severity_counts["high"] + severity_counts["critical"]
    configured_user = str(
        ((inspect.get("Config") or {}).get("User")) or ""
    ).strip()
    runs_as_nonroot = bool(configured_user) and configured_user not in {"0", "root", "0:0"}
    scanned_image_id = str((raw.get("Metadata") or {}).get("ImageID") or "")
    current_image_id = str(inspect.get("Id") or "")
    image_identity_matches = bool(scanned_image_id) and (
        not current_image_id or scanned_image_id == current_image_id
    )

    supply = supply_chain or {}
    sbom_summary = sbom or {}
    supply_chain_blocked = bool(supply) and not bool(
        supply.get("all_base_images_digest_pinned")
    )
    sbom_blocked = bool(sbom_summary) and not bool(
        sbom_summary.get("available")
    )
    if not image_identity_matches:
        status = "stale_image_mismatch"
    elif not runs_as_nonroot:
        status = "blocked"
    elif severity_counts["critical"] or fixable:
        status = "blocked"
    elif supply_chain_blocked or sbom_blocked:
        status = "blocked"
    elif severity_counts["high"]:
        status = "needs_attention"
    else:
        status = "acceptable"

    ranked = sorted(
        findings,
        key=lambda row: (
            -SEVERITIES.index(str(row.get("Severity") or "UNKNOWN").upper()),
            not bool(str(row.get("FixedVersion") or "").strip()),
            str(row.get("VulnerabilityID") or ""),
        ),
    )
    top_findings = [
        {
            "vulnerability_id": row.get("VulnerabilityID"),
            "severity": str(row.get("Severity") or "UNKNOWN").upper(),
            "package": row.get("PkgName"),
            "installed_version": row.get("InstalledVersion"),
            "fixed_version": row.get("FixedVersion") or None,
            "status": row.get("Status"),
            "primary_url": row.get("PrimaryURL"),
        }
        for row in ranked[:25]
    ]

    return {
        "schema_version": "container_security_scan_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scanner": {
            "name": "trivy",
            "version": scanner_version,
            "executed": True,
            "raw_schema_version": raw.get("SchemaVersion"),
        },
        "image": {
            "reference": raw.get("ArtifactName"),
            "artifact_type": raw.get("ArtifactType"),
            "scanned_image_id": scanned_image_id,
            "current_image_id": current_image_id or None,
            "identity_matches_current_image": image_identity_matches,
            "configured_user": configured_user or None,
            "runs_as_nonroot": runs_as_nonroot,
            "distroless_runtime_declared": True,
            "repo_digests": inspect.get("RepoDigests") or [],
            "oci_labels": {
                key: value
                for key, value in (((inspect.get("Config") or {}).get("Labels")) or {}).items()
                if key.startswith("org.opencontainers.image.")
                or key.startswith("ai.nlcare.")
            },
        },
        "supply_chain": supply,
        "sbom": sbom_summary,
        "summary": {
            "total_finding_count": len(findings),
            "severity_counts": severity_counts,
            "high_or_critical_count": high_or_critical,
            "fixable_high_or_critical_count": len(fixable),
            "unfixed_high_or_critical_count": high_or_critical - len(fixable),
            "public_deployment_blocked": status != "acceptable",
            "base_images_digest_pinned": supply.get(
                "all_base_images_digest_pinned"
            ),
            "sbom_available": sbom_summary.get("available"),
        },
        "top_findings": top_findings,
        "deployment_decision": (
            "BLOCK_PUBLIC_DEPLOYMENT" if status != "acceptable" else "ELIGIBLE_FOR_FURTHER_REVIEW"
        ),
        "scope": "restricted_synthetic_staging_container",
        "patient_data_processed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "This is a point-in-time container vulnerability scan for a synthetic-only "
            "engineering image. It is not a penetration test, security certification, "
            "clinical validation, compliance proof, or production healthcare readiness."
        ),
    }


def build_container_security_scan(
    *,
    image: str = DEFAULT_IMAGE,
    raw_path: Path = DEFAULT_RAW_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sbom_path: Path = DEFAULT_SBOM_PATH,
    execute_scan: bool = False,
    executor: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    trivy = _find_trivy()
    scanner_version = _tool_version(trivy, executor) if trivy else None
    if execute_scan:
        if not trivy:
            report = _unavailable_report(image, "trivy executable not found")
            return _write(report, output_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        completed = executor(
            [trivy, "image", "--scanners", "vuln", "--format", "json", "--output", str(raw_path), image],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if completed.returncode != 0:
            report = _unavailable_report(image, f"trivy exited {completed.returncode}")
            return _write(report, output_path)
        sbom_completed = executor(
            [
                trivy,
                "image",
                "--format",
                "cyclonedx",
                "--output",
                str(sbom_path),
                image,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if sbom_completed.returncode != 0:
            report = _unavailable_report(
                image,
                f"trivy SBOM generation exited {sbom_completed.returncode}",
            )
            return _write(report, output_path)

    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _write(_unavailable_report(image, f"raw result unavailable: {exc}"), output_path)

    inspect = _docker_inspect(image, executor)
    supply_chain = _dockerfile_supply_chain(ROOT / "Dockerfile")
    sbom_summary = _sbom_summary(sbom_path)
    report = summarize_trivy_result(
        raw,
        image_inspect=inspect,
        scanner_version=scanner_version,
        supply_chain=supply_chain,
        sbom=sbom_summary,
    )
    report["raw_artifact_path"] = str(raw_path.relative_to(ROOT)).replace("\\", "/")
    report["raw_artifact_sha256"] = _sha256(raw_path)
    return _write(report, output_path)


def _dockerfile_supply_chain(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {
            "dockerfile_available": False,
            "all_base_images_digest_pinned": False,
            "base_images": [],
        }
    base_images = [
        match.group(1)
        for match in re.finditer(r"(?im)^FROM\s+([^\s]+)", text)
    ]
    digest_pattern = re.compile(r"@sha256:[0-9a-f]{64}$", re.IGNORECASE)
    return {
        "dockerfile_available": True,
        "dockerfile_path": _display_path(path),
        "dockerfile_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "base_images": base_images,
        "all_base_images_digest_pinned": bool(base_images)
        and all(digest_pattern.search(image) for image in base_images),
        "mutable_tag_only_base_count": sum(
            not bool(digest_pattern.search(image)) for image in base_images
        ),
    }


def _sbom_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "path": _display_path(path),
            "reason": type(exc).__name__,
        }
    components = payload.get("components") or []
    return {
        "available": True,
        "format": payload.get("bomFormat"),
        "spec_version": payload.get("specVersion"),
        "component_count": len(components),
        "serial_number": payload.get("serialNumber"),
        "path": _display_path(path),
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        value = path.relative_to(ROOT)
    except ValueError:
        value = path
    return str(value).replace("\\", "/")


def _docker_inspect(
    image: str,
    executor: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    docker = shutil.which("docker")
    if not docker:
        return {}
    completed = executor(
        [docker, "image", "inspect", image],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        return {}
    try:
        rows = json.loads(completed.stdout)
    except (ValueError, json.JSONDecodeError):
        return {}
    return rows[0] if rows else {}


def _find_trivy() -> str | None:
    direct = shutil.which("trivy")
    if direct:
        return direct
    local = Path(os.environ.get("LOCALAPPDATA") or "")
    candidates = sorted(
        local.glob("Microsoft/WinGet/Packages/AquaSecurity.Trivy_*/trivy.exe")
    ) if local else []
    return str(candidates[-1]) if candidates else None


def _tool_version(
    executable: str,
    executor: Callable[..., subprocess.CompletedProcess[str]],
) -> str | None:
    try:
        completed = executor(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = (completed.stdout or completed.stderr or "").strip().splitlines()
    return first[0] if completed.returncode == 0 and first else None


def _unavailable_report(image: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "container_security_scan_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "tool_unavailable",
        "scanner": {"name": "trivy", "executed": False, "reason": reason},
        "image": {"reference": image, "runs_as_nonroot": None},
        "summary": {
            "total_finding_count": None,
            "high_or_critical_count": None,
            "fixable_high_or_critical_count": None,
            "public_deployment_blocked": True,
        },
        "deployment_decision": "BLOCK_PUBLIC_DEPLOYMENT",
        "scope": "restricted_synthetic_staging_container",
        "patient_data_processed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": "No container-security assurance is claimed without an executed scan.",
    }


def _write(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "build_container_security_scan",
    "summarize_trivy_result",
    "_dockerfile_supply_chain",
]
