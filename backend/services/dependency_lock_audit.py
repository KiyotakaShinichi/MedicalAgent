"""Audit direct and environment-scoped transitive Python dependency locks."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIREMENTS = Path("requirements.txt")
LOCK = Path("requirements-lock.txt")
TRANSITIVE_LOCK = Path("requirements-lock-py314-win.txt")
OUTPUT = Path("Data/evals/ops/latest_dependency_lock_audit.json")


def build_dependency_lock_audit(
    requirements_path: str | Path = REQUIREMENTS,
    lock_path: str | Path = LOCK,
    transitive_lock_path: str | Path = TRANSITIVE_LOCK,
) -> dict[str, Any]:
    requested = _requirements(Path(requirements_path), locked=False)
    locked = _requirements(Path(lock_path), locked=True)
    missing_from_lock = sorted(set(requested) - set(locked))
    unexpected_in_lock = sorted(set(locked) - set(requested))
    installed_missing: list[str] = []
    installed_drift: list[dict[str, str]] = []
    for name, locked_version in locked.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed_missing.append(name)
            continue
        if installed != locked_version:
            installed_drift.append({"package": name, "locked": locked_version, "installed": installed})

    lock_complete = not missing_from_lock and all(bool(version) for version in locked.values())
    environment_matches_lock = not installed_missing and not installed_drift
    transitive_path = Path(transitive_lock_path)
    transitive_locked, transitive_metadata = _transitive_lock(transitive_path)
    installed_all = _installed_distributions()
    transitive_missing_from_lock = sorted(set(installed_all) - set(transitive_locked))
    transitive_missing_from_environment = sorted(set(transitive_locked) - set(installed_all))
    transitive_version_drift = [
        {"package": name, "locked": transitive_locked[name], "installed": installed_all[name]}
        for name in sorted(set(transitive_locked) & set(installed_all))
        if transitive_locked[name] != installed_all[name]
    ]
    fingerprint = _environment_fingerprint()
    fingerprint_matches = bool(transitive_metadata) and all(
        transitive_metadata.get(key) == value for key, value in fingerprint.items()
    )
    direct_covered_by_transitive = not (set(requested) - set(transitive_locked))
    transitive_lock_complete = bool(transitive_locked) and direct_covered_by_transitive and not (
        transitive_missing_from_lock
        or transitive_missing_from_environment
        or transitive_version_drift
    )
    environment_matches_transitive_lock = transitive_lock_complete and fingerprint_matches
    status = (
        "acceptable"
        if lock_complete and environment_matches_lock and environment_matches_transitive_lock
        else "needs_attention"
    )
    return {
        "schema_version": "dependency_lock_audit_v2_2026_07",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "direct_requirement_count": len(requested),
        "locked_direct_requirement_count": len(locked),
        "lock_complete": lock_complete,
        "missing_from_lock": missing_from_lock,
        "unexpected_in_lock": unexpected_in_lock,
        "environment_matches_lock": environment_matches_lock,
        "installed_missing": installed_missing,
        "installed_version_drift": installed_drift,
        "transitive_lock_path": str(transitive_path),
        "transitive_lock_sha256": _sha256(transitive_path),
        "transitive_locked_distribution_count": len(transitive_locked),
        "installed_distribution_count": len(installed_all),
        "direct_requirements_covered_by_transitive_lock": direct_covered_by_transitive,
        "transitive_lock_complete": transitive_lock_complete,
        "environment_matches_transitive_lock": environment_matches_transitive_lock,
        "transitive_missing_from_lock": transitive_missing_from_lock,
        "transitive_missing_from_environment": transitive_missing_from_environment,
        "transitive_version_drift": transitive_version_drift,
        "environment_fingerprint": fingerprint,
        "locked_environment_fingerprint": transitive_metadata,
        "portable_across_platforms": False,
        "vulnerability_scan_included": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "The exact transitive lock reproduces the declared Python/Windows environment only. It is not a "
            "cross-platform lock, vulnerability assessment, compliance certification, clinical validation, or "
            "healthcare production proof."
        ),
    }


def write_dependency_lock_audit(path: str | Path = OUTPUT) -> dict[str, Any]:
    payload = build_dependency_lock_audit()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_environment_transitive_lock(path: str | Path = TRANSITIVE_LOCK) -> Path:
    output = Path(path)
    fingerprint = _environment_fingerprint()
    lines = [
        "# Generated environment-specific transitive lock. Do not edit manually.",
        "# Scope: exact local engineering environment; not portable or a security certification.",
        *[f"# {key}: {value}" for key, value in fingerprint.items()],
        "",
    ]
    lines.extend(f"{name}=={version}" for name, version in _installed_distributions().items())
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _requirements(path: Path, *, locked: bool) -> dict[str, str | None]:
    parsed: dict[str, str | None] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?(?:==([^;\s]+))?", line)
        if not match:
            continue
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        version = match.group(2)
        if locked and not version:
            version = ""
        parsed[name] = version
    return parsed


def _installed_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[_canonical_name(name)] = distribution.version
    return dict(sorted(installed.items()))


def _transitive_lock(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.exists():
        return {}, {}
    locked = _requirements(path, locked=True)
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#\s*([a-z_]+):\s*(.+)$", line.strip())
        if match:
            metadata[match.group(1)] = match.group(2)
    return {name: str(version) for name, version in locked.items()}, metadata


def _environment_fingerprint() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "build_dependency_lock_audit",
    "write_dependency_lock_audit",
    "write_environment_transitive_lock",
]
