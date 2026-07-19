import importlib.metadata

from backend.services.dependency_lock_audit import (
    build_dependency_lock_audit,
    write_environment_transitive_lock,
)


def test_lock_audit_detects_complete_direct_coverage(tmp_path) -> None:
    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements-lock.txt"
    requirements.write_text("pytest\nfastapi\n", encoding="utf-8")
    lock.write_text("pytest==9.0.3\nfastapi==0.136.1\n", encoding="utf-8")
    artifact = build_dependency_lock_audit(requirements, lock, tmp_path / "missing-transitive-lock.txt")
    assert artifact["lock_complete"] is True
    assert artifact["missing_from_lock"] == []
    assert artifact["transitive_lock_complete"] is False
    assert artifact["vulnerability_scan_included"] is False
    assert artifact["clinical_validation"] is False


def test_environment_transitive_lock_matches_writer_interpreter(tmp_path) -> None:
    requirements = tmp_path / "requirements.txt"
    direct_lock = tmp_path / "requirements-lock.txt"
    transitive_lock = tmp_path / "requirements-lock-env.txt"
    pytest_version = importlib.metadata.version("pytest")
    requirements.write_text("pytest\n", encoding="utf-8")
    direct_lock.write_text(f"pytest=={pytest_version}\n", encoding="utf-8")
    write_environment_transitive_lock(transitive_lock)

    artifact = build_dependency_lock_audit(requirements, direct_lock, transitive_lock)

    assert artifact["status"] == "acceptable"
    assert artifact["transitive_lock_complete"] is True
    assert artifact["environment_matches_transitive_lock"] is True
    assert artifact["portable_across_platforms"] is False
