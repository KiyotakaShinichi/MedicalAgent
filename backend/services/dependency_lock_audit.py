"""Audit direct Python dependency lock coverage and local environment drift."""

from __future__ import annotations

import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIREMENTS = Path("requirements.txt")
LOCK = Path("requirements-lock.txt")
OUTPUT = Path("Data/evals/ops/latest_dependency_lock_audit.json")


def build_dependency_lock_audit(
    requirements_path: str | Path = REQUIREMENTS,
    lock_path: str | Path = LOCK,
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
    return {
        "schema_version": "dependency_lock_audit_v1_2026_07",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if lock_complete and environment_matches_lock else "needs_attention",
        "direct_requirement_count": len(requested),
        "locked_direct_requirement_count": len(locked),
        "lock_complete": lock_complete,
        "missing_from_lock": missing_from_lock,
        "unexpected_in_lock": unexpected_in_lock,
        "environment_matches_lock": environment_matches_lock,
        "installed_missing": installed_missing,
        "installed_version_drift": installed_drift,
        "transitive_lock_complete": False,
        "vulnerability_scan_included": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "Direct dependency pinning improves engineering reproducibility only. It is not a complete transitive "
            "lock, vulnerability assessment, compliance certification, or healthcare production proof."
        ),
    }


def write_dependency_lock_audit(path: str | Path = OUTPUT) -> dict[str, Any]:
    payload = build_dependency_lock_audit()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


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


__all__ = ["build_dependency_lock_audit", "write_dependency_lock_audit"]
