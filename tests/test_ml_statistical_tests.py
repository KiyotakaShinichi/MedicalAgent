from __future__ import annotations

import json
from pathlib import Path

from backend.services.ml_statistical_tests import build_ml_statistical_evidence


def test_ml_statistical_evidence_writes_sections_and_boundaries(tmp_path: Path):
    output = tmp_path / "ml_statistical_evidence.json"
    report = build_ml_statistical_evidence(output_path=output)

    assert output.exists()
    assert report["schema_version"].startswith("ml_statistical_evidence_v1")
    assert "clinical validity" in report["claim_boundary"]
    assert set(report["sections"]) >= {
        "per_head_calibration",
        "modality_robustness_comparison",
        "deep_learning_candidate_comparison",
        "subgroup_statistical_screen",
        "patient_temporal_cv",
    }

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recommended_next_raw_prediction_exports"]
    assert payload["sections"]["modality_robustness_comparison"]["claim_boundary"]


def test_ml_statistical_evidence_reports_missing_artifacts(tmp_path: Path):
    output = tmp_path / "missing_evidence.json"
    missing = {"per_head_calibration": tmp_path / "does_not_exist.json"}

    report = build_ml_statistical_evidence(artifacts=missing, output_path=output)

    assert report["status"] == "needs_attention"
    assert report["missing_artifacts"] == [str(missing["per_head_calibration"])]
    assert output.exists()
