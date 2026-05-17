import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.clinical_safety_checklist import build_clinical_safety_review_checklist
from backend.services.system_health import build_system_health_report


def test_system_health_reports_artifacts_and_dependencies(tmp_path):
    output_path = tmp_path / "system_health.json"
    report = build_system_health_report(db=None, output_path=str(output_path))

    assert output_path.exists()
    assert report["schema_version"] == "system_health_v1"
    assert report["backend"]["database"]["status"] == "unknown"
    assert any(row["package"] == "fastapi" for row in report["dependencies"])
    assert any(row["name"] == "benchmark_registry" for row in report["artifacts"])
    assert report["claim_boundary"]


def test_clinical_safety_checklist_has_review_sections(tmp_path):
    output_path = tmp_path / "checklist.json"
    report = build_clinical_safety_review_checklist(output_path=str(output_path))

    assert output_path.exists()
    assert report["status"] == "ready_for_review"
    section_ids = {section["id"] for section in report["sections"]}
    assert "genetics_and_biomarkers" in section_ids
    assert "urgent_symptom_escalation" in section_ids
    assert report["sign_off_fields"]["decision"] == "pending"
