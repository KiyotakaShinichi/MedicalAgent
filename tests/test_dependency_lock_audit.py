from backend.services.dependency_lock_audit import build_dependency_lock_audit


def test_lock_audit_detects_complete_direct_coverage(tmp_path) -> None:
    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements-lock.txt"
    requirements.write_text("pytest\nfastapi\n", encoding="utf-8")
    lock.write_text("pytest==9.0.3\nfastapi==0.136.1\n", encoding="utf-8")
    artifact = build_dependency_lock_audit(requirements, lock)
    assert artifact["lock_complete"] is True
    assert artifact["missing_from_lock"] == []
    assert artifact["transitive_lock_complete"] is False
    assert artifact["vulnerability_scan_included"] is False
    assert artifact["clinical_validation"] is False
