"""Generate bounded software supply-chain evidence from repository lockfiles."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.ci_secret_scan import scan_secret_findings


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("Data/evals/ops/latest_software_supply_chain_evidence.json")
DEFAULT_SBOM_DIR = Path("Data/evals/ops/sbom")
DEFAULT_CONTAINER_SCAN = Path("Data/evals/ops/latest_container_security_scan.json")
_PYTHON_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")


def build_software_supply_chain_evidence(
    *,
    root: str | Path = ROOT,
    python_lock: str | Path = "requirements-lock-py314-win.txt",
    frontend_lock: str | Path = "frontend-react/package-lock.json",
    output_path: str | Path = DEFAULT_OUTPUT,
    sbom_dir: str | Path = DEFAULT_SBOM_DIR,
    container_scan_path: str | Path = DEFAULT_CONTAINER_SCAN,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    python_path = _resolve(repo, python_lock)
    frontend_path = _resolve(repo, frontend_lock)
    output = _resolve(repo, output_path)
    sbom_root = _resolve(repo, sbom_dir)
    container_scan = _load_container_scan(_resolve(repo, container_scan_path))

    python_components = _python_components(python_path)
    frontend_components = _frontend_components(frontend_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    python_sbom = _cyclonedx(
        "nlcare-python-runtime",
        python_components,
        generated_at,
        _sha256(python_path),
    )
    frontend_sbom = _cyclonedx(
        "nlcare-frontend-runtime",
        frontend_components,
        generated_at,
        _sha256(frontend_path),
    )

    sbom_root.mkdir(parents=True, exist_ok=True)
    python_sbom_path = sbom_root / "python.cdx.json"
    frontend_sbom_path = sbom_root / "frontend.cdx.json"
    python_sbom_path.write_text(json.dumps(python_sbom, indent=2), encoding="utf-8")
    frontend_sbom_path.write_text(json.dumps(frontend_sbom, indent=2), encoding="utf-8")

    secret_findings = scan_secret_findings(repo)
    scanners = {
        name: {
            "available": shutil.which(name) is not None,
            "executed": False,
            "reason": (
                "tool detected; execution remains a deployment/CI action"
                if shutil.which(name)
                else "tool unavailable in this environment"
            ),
        }
        for name in ("trivy", "syft", "grype")
    }
    scanners["docker"] = {
        "available": shutil.which("docker") is not None,
        "executed": False,
        "reason": "image build and scan are separate deployment actions",
    }
    if (container_scan.get("scanner") or {}).get("executed"):
        scanners["trivy"] = {
            "available": True,
            "executed": True,
            "reason": "canonical container scan artifact was executed",
        }

    container_attention = bool((container_scan.get("scanner") or {}).get("executed")) and (
        container_scan.get("status") != "acceptable"
    )

    payload = {
        "schema_version": "software_supply_chain_evidence_v1",
        "generated_at": generated_at,
        "status": (
            "acceptable"
            if not secret_findings and not container_attention
            else "needs_attention"
        ),
        "lockfiles": {
            "python": _lock_summary(repo, python_path, python_components),
            "frontend": _lock_summary(repo, frontend_path, frontend_components),
        },
        "sbom": {
            "format": "CycloneDX",
            "spec_version": "1.5",
            "python_path": str(python_sbom_path.relative_to(repo)).replace("\\", "/"),
            "frontend_path": str(frontend_sbom_path.relative_to(repo)).replace("\\", "/"),
            "component_count": len(python_components) + len(frontend_components),
        },
        "secret_scan": {
            "executed": True,
            "finding_count": len(secret_findings),
            "findings": secret_findings,
            "secret_values_included": False,
        },
        "container_scan": {
            "executed": bool((container_scan.get("scanner") or {}).get("executed")),
            "tools": scanners,
            "status": container_scan.get("status"),
            "image": (container_scan.get("image") or {}).get("reference"),
            "image_identity_matches_current_image": (
                container_scan.get("image") or {}
            ).get("identity_matches_current_image"),
            "high_or_critical_count": (
                container_scan.get("summary") or {}
            ).get("high_or_critical_count"),
            "fixable_high_or_critical_count": (
                container_scan.get("summary") or {}
            ).get("fixable_high_or_critical_count"),
            "deployment_decision": container_scan.get("deployment_decision"),
            "artifact_path": str(
                _resolve(repo, container_scan_path).relative_to(repo)
            ).replace("\\", "/"),
            "gap": (
                "Container findings remain an explicit deployment blocker."
                if container_scan.get("status") not in {"acceptable"}
                else "Point-in-time scan passed its current engineering policy."
            ),
        },
        "dependency_vulnerability_artifact": "Data/evals/ops/latest_dependency_security_scan.json",
        "clinical_validation": False,
        "production_healthcare_ready": False,
        "claim_boundary": (
            "Lock-derived SBOMs and a repository secret scan are software-engineering "
            "evidence only. They are not a security certification, compliance proof, "
            "clinical validation, or production healthcare readiness."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _python_components(path: Path) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = _PYTHON_PIN.match(raw.strip())
        if not match:
            continue
        name, version = match.groups()
        normalized = name.lower().replace("_", "-")
        components.append({
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{normalized}@{version}",
        })
    return sorted(components, key=lambda item: item["name"].lower())


def _frontend_components(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    components: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for package_path, package in (payload.get("packages") or {}).items():
        if not package_path or not isinstance(package, dict):
            continue
        name = str(package.get("name") or package_path.rsplit("node_modules/", 1)[-1])
        version = str(package.get("version") or "")
        if not name or not version or (name, version) in seen:
            continue
        seen.add((name, version))
        encoded_name = name.replace("@", "%40", 1) if name.startswith("@") else name
        components.append({
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:npm/{encoded_name}@{version}",
        })
    return sorted(components, key=lambda item: (item["name"].lower(), item["version"]))


def _cyclonedx(
    name: str,
    components: list[dict[str, Any]],
    generated_at: str,
    lock_sha256: str,
) -> dict[str, Any]:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{lock_sha256[:8]}-{lock_sha256[8:12]}-"
        f"{lock_sha256[12:16]}-{lock_sha256[16:20]}-{lock_sha256[20:32]}",
        "version": 1,
        "metadata": {
            "timestamp": generated_at,
            "component": {"type": "application", "name": name, "version": "prototype"},
            "properties": [{"name": "nlcare:lock_sha256", "value": lock_sha256}],
        },
        "components": components,
    }


def _lock_summary(repo: Path, path: Path, components: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(repo)).replace("\\", "/"),
        "exists": path.exists(),
        "sha256": _sha256(path),
        "component_count": len(components),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_container_scan(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = ["build_software_supply_chain_evidence"]
