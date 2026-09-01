from __future__ import annotations

import json
import subprocess

import pytest

from scripts.buyer.contracts import (
    ROOT,
    load_json,
    validate_asset_manifest,
    validate_candidate_manifest,
    validate_configuration_matrix,
    validate_license_inventory,
    verify_protected_evidence,
)
from scripts.buyer.demo import (
    DEFAULT_DATABASE,
    DemoSafetyError,
    backup_demo,
    marker_path,
    reset_demo,
    restore_demo,
    seed_demo,
)
from scripts.buyer.package import archive_bytes, selected_files, verify_archive


def test_buyer_manifests_validate() -> None:
    validate_asset_manifest(load_json("config/buyer/asset_manifest.json"))
    validate_license_inventory(load_json("config/buyer/license_inventory.json"))
    validate_candidate_manifest(load_json("config/buyer/candidate.json"))
    validate_configuration_matrix(load_json("config/buyer/configuration_matrix.json"))


def test_unknown_license_is_never_treated_as_clear() -> None:
    payload = load_json("config/buyer/license_inventory.json")
    unknown = [item for item in payload["components"] if item["license"] == "UNKNOWN"]
    assert unknown
    assert {item["transfer_status"] for item in unknown} == {"REVIEW_REQUIRED"}


def test_absent_root_license_is_disclosed() -> None:
    assert not (ROOT / "LICENSE").exists()
    inventory = load_json("config/buyer/license_inventory.json")
    source = next(
        item for item in inventory["components"] if item["component"].startswith("NLCare first-party")
    )
    assert source["transfer_status"] == "REVIEW_REQUIRED"


def test_protected_evidence_is_complete_and_unchanged() -> None:
    count, failures = verify_protected_evidence(
        load_json("config/buyer/protected_evidence_manifest.json")
    )
    assert count == 757
    assert failures == []


def test_package_selection_includes_every_protected_file() -> None:
    selected = set(selected_files())
    protected = load_json("config/buyer/protected_evidence_manifest.json")["files"]
    assert {entry["path"] for entry in protected} <= selected


def test_package_selection_excludes_runtime_and_external_source_data() -> None:
    selected = selected_files()
    assert not any(path.endswith((".db", ".sqlite", ".sqlite3", ".log")) for path in selected)
    assert not any(path.startswith(("tmp/", "logs/", "Data/external_bridge/")) for path in selected)
    assert ".env.example" in selected


def test_small_archive_is_deterministic_and_manifest_bound() -> None:
    paths = ["README.md", "config/buyer/candidate.json"]
    first, first_manifest = archive_bytes(paths)
    second, second_manifest = archive_bytes(paths)
    assert first == second
    assert first_manifest == second_manifest
    assert verify_archive(first) == first_manifest


def test_demo_path_rejects_non_demo_database() -> None:
    with pytest.raises(DemoSafetyError, match="must stay under"):
        seed_demo(ROOT / "medical_agent.db")


def test_demo_reset_refuses_unmarked_database() -> None:
    database = DEFAULT_DATABASE.with_name("unmarked_test.db")
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b"not a demo")
    try:
        with pytest.raises(DemoSafetyError, match="unmarked"):
            reset_demo(database)
    finally:
        database.unlink(missing_ok=True)


def test_demo_seed_is_deterministic_and_backup_restores() -> None:
    database = DEFAULT_DATABASE.with_name("buyer_contract_test.db")
    backup = DEFAULT_DATABASE.with_name("buyer_contract_backup.sqlite")
    for candidate in (database, backup, marker_path(database), marker_path(backup)):
        candidate.unlink(missing_ok=True)
    try:
        first = seed_demo(database)
        second = seed_demo(database)
        assert first["logical_fingerprint"] == second["logical_fingerprint"]
        backup_demo(backup, database)
        reset_demo(database)
        restored = restore_demo(backup, database)
        assert restored["logical_fingerprint"] == first["logical_fingerprint"]
    finally:
        if marker_path(database).exists():
            reset_demo(database)
        backup.unlink(missing_ok=True)
        marker_path(backup).unlink(missing_ok=True)


def test_current_tree_has_no_tracked_personal_cv_or_runtime_logs() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    assert not any("newestcv" in path.lower() for path in tracked)
    assert not any(path.endswith(".log") for path in tracked if not path.startswith("Data/evals/"))


def test_candidate_does_not_claim_release_or_hide_dep001() -> None:
    candidate = load_json("config/buyer/candidate.json")
    assert candidate["production_release"] is False
    assert candidate["clinical_release"] is False
    assert candidate["dep001_status"] == "BLOCKED_BEHAVIORAL"
    assert "DEP-001" in " ".join(candidate["known_blockers"])


def test_buyer_data_room_entry_points_exist() -> None:
    required = load_json("config/buyer/package_policy.json")["required_paths"]
    assert all((ROOT / path).is_file() for path in required)
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Acquisition / Technical Diligence" in text


def test_docker_metadata_points_to_current_repository() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "KiyotakaShinichi/NLCare" in dockerfile
    assert "KiyotakaShinichi/MedicalAgent" not in dockerfile


def test_configuration_renderer_is_stable() -> None:
    before = (ROOT / "config/buyer/configuration_matrix.json").read_text(encoding="utf-8")
    process = subprocess.run(
        ["python", "scripts/render_buyer_configuration_matrix.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = (ROOT / "config/buyer/configuration_matrix.json").read_text(encoding="utf-8")
    assert json.loads(before) == json.loads(after)
    assert "wrote config" in process.stdout
