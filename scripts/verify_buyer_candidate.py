"""Canonical verification command for an NLCare buyer candidate.

The default mode validates transfer contracts, protected evidence, repository
contents, offline prerequisites, the synthetic demo lifecycle, and package
selection. ``--full`` additionally provisions derived artifacts and runs the
repository's declared fresh-clone consumer tests. It does not reinterpret or
regenerate scientific evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.buyer.contracts import (  # noqa: E402
    combined_hash,
    git_sha,
    load_json,
    sha256_file,
    validate_asset_manifest,
    validate_candidate_manifest,
    validate_configuration_matrix,
    validate_license_inventory,
    verify_protected_evidence,
)
from scripts.buyer.demo import reset_demo, seed_demo  # noqa: E402
from scripts.buyer.package import selected_files  # noqa: E402


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _capture(command: list[str]) -> tuple[bool, str]:
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (process.stdout, process.stderr) if part.strip())
    return process.returncode == 0, output[-2000:] or f"exit={process.returncode}"


def _contract_check(name: str, path: str, validator: Callable[[dict], None]) -> Check:
    try:
        validator(load_json(path))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return Check(name, False, str(exc))
    return Check(name, True, path)


def _repository_clean(allow_dirty: bool) -> Check:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()
    if output and not allow_dirty:
        return Check("repository_clean", False, "tracked working-tree changes are present")
    return Check("repository_clean", True, "dirty state explicitly allowed" if output else "clean")


def _protected_evidence() -> Check:
    try:
        count, failures = verify_protected_evidence(
            load_json("config/buyer/protected_evidence_manifest.json")
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return Check("protected_evidence", False, str(exc))
    return Check(
        "protected_evidence",
        not failures and count == 757,
        f"{count}/757 byte-identical" if not failures else "; ".join(failures[:10]),
    )


def _demo_lifecycle() -> Check:
    try:
        first = seed_demo()
        second = seed_demo()
        if first["logical_fingerprint"] != second["logical_fingerprint"]:
            return Check("synthetic_demo", False, "logical fingerprint changed across reset/seed")
        reset_demo()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return Check("synthetic_demo", False, str(exc))
    return Check("synthetic_demo", True, f"deterministic {first['logical_fingerprint']}")


def _package_dry_run() -> Check:
    try:
        files = selected_files()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return Check("buyer_package", False, str(exc))
    return Check("buyer_package", True, f"{len(files) + 1} manifest-bound files")


def _personal_path_check() -> Check:
    windows_profile_prefix = "C:" + "\\Users"
    personal_cv_marker = "newest" + "CV"
    process = subprocess.run(
        ["git", "grep", "-IlF", "-e", windows_profile_prefix, "-e", personal_cv_marker],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if process.returncode not in {0, 1}:
        return Check("owner_machine_paths", False, process.stderr.strip())
    tracked = process.stdout.splitlines()
    allowed = {
        "backend/services/dep001b_overlap_audit.py",
        "backend/services/dep001c_blind_bank.py",
        "Data/evals/safety/latest_adversarial_generalization_label_audit.json",
    }
    unexpected = sorted(set(path.replace("\\", "/") for path in tracked) - allowed)
    return Check(
        "owner_machine_paths",
        not unexpected,
        "only documented consumed-holdout compatibility paths remain"
        if not unexpected
        else f"unexpected paths: {unexpected}",
    )


def build_report(*, full: bool, allow_dirty: bool) -> dict:
    checks = [
        _repository_clean(allow_dirty),
        _contract_check(
            "asset_manifest", "config/buyer/asset_manifest.json", validate_asset_manifest
        ),
        _contract_check(
            "license_inventory", "config/buyer/license_inventory.json", validate_license_inventory
        ),
        _contract_check(
            "configuration_matrix",
            "config/buyer/configuration_matrix.json",
            validate_configuration_matrix,
        ),
        _contract_check(
            "candidate_manifest", "config/buyer/candidate.json", validate_candidate_manifest
        ),
        _protected_evidence(),
        _personal_path_check(),
    ]
    commands = [
        ("dependency_contract", [sys.executable, "scripts/check_dependency_contract.py"]),
        ("environment_documentation", [sys.executable, "scripts/check_env_documentation.py"]),
        ("secret_scan", [sys.executable, "scripts/ci_secret_scan.py"]),
        ("fresh_clone_structure", [sys.executable, "scripts/check_fresh_clone_offline.py"]),
    ]
    if full:
        commands.append(
            (
                "fresh_clone_provisioned_consumers",
                [
                    sys.executable,
                    "scripts/check_fresh_clone_offline.py",
                    "--provision",
                    "--run-tests",
                ],
            )
        )
    for name, command in commands:
        passed, detail = _capture(command)
        checks.append(Check(name, passed, detail))
    checks.extend([_demo_lifecycle(), _package_dry_run()])

    resolved_hashes = {
        "asset_manifest_hash": sha256_file("config/buyer/asset_manifest.json"),
        "license_inventory_hash": sha256_file("config/buyer/license_inventory.json"),
        "dependency_lock_hash": combined_hash(["uv.lock", "frontend-react/package-lock.json"]),
        "evidence_index_hash": sha256_file("docs/buyer/EVIDENCE_AND_LIMITATIONS.md"),
        "protected_evidence_hash": sha256_file("config/buyer/protected_evidence_manifest.json"),
    }
    return {
        "schema_version": "nlcare_buyer_candidate_verification_v1",
        "candidate_type": "BUYER_CANDIDATE",
        "source_sha": git_sha(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "production_release": False,
        "clinical_release": False,
        "dep001_status": "BLOCKED_BEHAVIORAL",
        "mode": "full" if full else "standard",
        "status": "passed" if all(check.passed for check in checks) else "failed",
        "checks": [asdict(check) for check in checks],
        "resolved_hashes": resolved_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = build_report(full=args.full, allow_dirty=args.allow_dirty)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        output = args.json_output
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
