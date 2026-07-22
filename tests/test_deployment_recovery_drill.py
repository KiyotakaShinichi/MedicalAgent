from pathlib import Path

from backend.services.deployment_recovery_drill import run_local_recovery_drill


def test_local_recovery_drill_restores_exact_synthetic_content(tmp_path: Path):
    report = run_local_recovery_drill(tmp_path / "recovery.json")
    assert report["status"] == "strong_local_only"
    assert report["passed"] is True
    assert report["restore"]["content_hash_match"] is True
    assert report["restore"]["integrity_check"] == "ok"
    assert report["contains_patient_data"] is False
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["strict_profile_validated"] is False
    assert report["postgres_restore_tested"] is False
