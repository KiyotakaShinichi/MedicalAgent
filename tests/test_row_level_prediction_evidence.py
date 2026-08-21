from __future__ import annotations

import csv

from backend.services.row_level_prediction_export import run_row_level_prediction_evidence


def test_row_level_prediction_export_and_paired_stats(tmp_path):
    export_csv = tmp_path / "rows.csv"
    manifest_json = tmp_path / "manifest.json"
    paired_json = tmp_path / "paired.json"
    calibration_json = tmp_path / "calibration.json"

    payload = run_row_level_prediction_evidence(
        export_csv=export_csv,
        manifest_json=manifest_json,
        paired_json=paired_json,
        calibration_json=calibration_json,
    )

    assert payload["manifest"]["status"] == "strong"
    assert payload["manifest"]["total_n"] > 0
    assert payload["manifest"]["patient_id_unique"] is True
    assert payload["paired"]["classification"]
    assert payload["paired"]["regression"]
    assert payload["paired"]["promotion_allowed"] is False
    assert payload["calibration"]["models"]
    assert payload["calibration"]["clinical_validation"] is False
    assert export_csv.exists()

    with export_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "gradient_boosting_calibrated_probability" in rows[0]
    assert "random_forest_regressor_absolute_error" in rows[0]
