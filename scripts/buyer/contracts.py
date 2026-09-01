"""Machine-readable buyer-candidate contracts and integrity checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ASSET_STATUSES = {
    "INCLUDED",
    "GENERATED",
    "DEPENDENCY",
    "OPTIONAL",
    "BUYER_MUST_PROVISION",
    "NOT_INCLUDED",
    "REVIEW_REQUIRED",
}
TRANSFER_STATUSES = {
    "CLEAR",
    "DEPENDENCY_ONLY",
    "BUYER_MUST_OBTAIN",
    "REVIEW_REQUIRED",
    "NOT_INCLUDED",
}
MATURITY_STATUSES = {"READY", "PARTIAL", "RESEARCH_ONLY", "RELEASE_BLOCKED", "OPTIONAL"}


class ContractError(ValueError):
    """Raised when a buyer-facing machine contract is invalid."""


def load_json(path: str | Path) -> Any:
    with (ROOT / path).open(encoding="utf-8") if not Path(path).is_absolute() else Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_file_bytes(path: str | Path) -> bytes:
    """Return the committed bytes for a tracked, repository-relative file."""
    normalized = Path(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ContractError(f"Tracked path must be repository-relative: {path}")
    relative = normalized.as_posix()
    try:
        return subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"Unable to read tracked file at HEAD: {relative}") from exc


def sha256_tracked_file(path: str | Path) -> str:
    """Hash committed bytes so checkout line-ending settings cannot change identity."""
    return hashlib.sha256(tracked_file_bytes(path)).hexdigest()


def combined_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_tracked_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def tracked_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    return sorted(path for path in output.split("\0") if path)


def parse_env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key.replace("_", "").isalnum() and key.upper() == key:
            values[key] = value
    return values


def validate_asset_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "nlcare_buyer_asset_manifest_v1":
        raise ContractError("Unexpected asset-manifest schema_version")
    required = {
        "asset_id",
        "path_or_category",
        "description",
        "ownership_status",
        "included_in_transfer",
        "generated",
        "requires_external_provider",
        "buyer_action_required",
        "notes",
    }
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ContractError("Asset manifest must contain assets")
    ids: set[str] = set()
    for asset in assets:
        missing = required - set(asset)
        if missing:
            raise ContractError(f"Asset is missing fields: {sorted(missing)}")
        if asset["asset_id"] in ids:
            raise ContractError(f"Duplicate asset_id: {asset['asset_id']}")
        ids.add(asset["asset_id"])
        if asset["ownership_status"] not in ASSET_STATUSES:
            raise ContractError(f"Invalid asset status: {asset['ownership_status']}")
        if not isinstance(asset["included_in_transfer"], bool):
            raise ContractError("included_in_transfer must be boolean")


def validate_license_inventory(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "nlcare_technical_license_inventory_v1":
        raise ContractError("Unexpected license-inventory schema_version")
    required = {
        "component",
        "version_or_source",
        "license",
        "commercial_use_status",
        "redistribution_status",
        "transfer_status",
        "evidence_reference",
        "buyer_action",
        "notes",
    }
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ContractError("License inventory must contain components")
    names: set[str] = set()
    for component in components:
        missing = required - set(component)
        if missing:
            raise ContractError(f"License component is missing fields: {sorted(missing)}")
        if component["component"] in names:
            raise ContractError(f"Duplicate license component: {component['component']}")
        names.add(component["component"])
        if component["transfer_status"] not in TRANSFER_STATUSES:
            raise ContractError(f"Invalid transfer status: {component['transfer_status']}")
        if component["license"] == "UNKNOWN" and component["transfer_status"] != "REVIEW_REQUIRED":
            raise ContractError("Unknown licenses must be REVIEW_REQUIRED")


def validate_configuration_matrix(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "nlcare_configuration_matrix_v1":
        raise ContractError("Unexpected configuration-matrix schema_version")
    entries = payload.get("variables")
    if not isinstance(entries, list):
        raise ContractError("Configuration matrix variables must be a list")
    required = {
        "variable",
        "category",
        "required",
        "secret",
        "default",
        "offline_behavior",
        "demo_behavior",
        "production_significance",
        "external_provider",
        "notes",
    }
    names: list[str] = []
    for entry in entries:
        missing = required - set(entry)
        if missing:
            raise ContractError(f"Configuration entry is missing fields: {sorted(missing)}")
        names.append(entry["variable"])
    if len(names) != len(set(names)):
        raise ContractError("Configuration matrix contains duplicate variables")
    expected = set(parse_env_example())
    actual = set(names)
    if expected != actual:
        raise ContractError(
            f"Configuration matrix drift: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def validate_candidate_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "nlcare_buyer_candidate_v1":
        raise ContractError("Unexpected buyer-candidate schema_version")
    if payload.get("candidate_type") != "BUYER_CANDIDATE":
        raise ContractError("candidate_type must be BUYER_CANDIDATE")
    if payload.get("source_sha") != "CURRENT_COMMIT":
        raise ContractError("Tracked candidate must resolve source_sha at verification time")
    if payload.get("clinical_release") is not False:
        raise ContractError("Buyer candidate must not claim clinical release")
    if payload.get("production_release") is not False:
        raise ContractError("Buyer candidate must not claim production release")
    if payload.get("dep001_status") not in {"BLOCKED_BEHAVIORAL", "RELEASE_BLOCKED"}:
        raise ContractError("DEP-001 blocker must remain explicit")
    blockers = " ".join(payload.get("known_blockers", [])).lower()
    for phrase in ("dep-001", "clinical", "license"):
        if phrase not in blockers:
            raise ContractError(f"Buyer candidate must disclose {phrase}")


def verify_protected_evidence(payload: dict[str, Any]) -> tuple[int, list[str]]:
    if payload.get("schema_version") != "nlcare_protected_evidence_manifest_v1":
        raise ContractError("Unexpected protected-evidence schema_version")
    entries = payload.get("files", [])
    failures: list[str] = []
    for entry in entries:
        path = entry.get("path", "")
        normalized = Path(path)
        if normalized.is_absolute() or ".." in normalized.parts:
            failures.append(f"invalid protected path: {path}")
            continue
        resolved = ROOT / path
        if not resolved.is_file():
            failures.append(f"missing: {path}")
        elif sha256_file(resolved) != entry.get("sha256"):
            failures.append(f"changed: {path}")
    if payload.get("file_count") != len(entries):
        failures.append("protected evidence file_count does not match entries")
    return len(entries), failures
