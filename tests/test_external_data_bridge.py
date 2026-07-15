from __future__ import annotations

import csv
import json

from backend.services.external_data_bridge import build_external_data_bridge


def test_external_data_bridge_maps_breastdcedl_into_canonical_rows(tmp_path):
    features = tmp_path / "features.csv"
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.csv"
    canonical = tmp_path / "canonical.csv"
    output = tmp_path / "bridge.json"
    gallery = tmp_path / "gallery.json"

    _write_csv(features, [
        {
            "patient_id": "ISPY1_TEST_01",
            "pcr_label": "1",
            "age": "52.5",
            "baseline_longest_diameter_mm": "41.2",
            "molecular_subtype": "HER2pos",
            "tumor_voxel_count": "1200",
            "early_enhancement_mean": "1.1",
            "delayed_enhancement_mean": "1.0",
            "washout_mean": "-0.1",
        },
        {
            "patient_id": "ISPY1_TEST_02",
            "pcr_label": "0",
            "age": "61",
            "baseline_longest_diameter_mm": "35.0",
            "molecular_subtype": "TripleNeg",
            "tumor_voxel_count": "900",
            "early_enhancement_mean": "0.8",
            "delayed_enhancement_mean": "0.7",
            "washout_mean": "-0.2",
        },
    ])
    metrics.write_text(json.dumps({
        "rows": 2,
        "positive_pcr": 1,
        "negative_pcr": 1,
        "model_type": "test_baseline",
        "best_model_by_roc_auc": "logistic_regression",
        "models": {"logistic_regression": {"roc_auc": 0.61}},
        "warning": "Exploratory PoC only. Not clinically validated.",
    }), encoding="utf-8")
    _write_csv(predictions, [
        {
            "patient_id": "ISPY1_TEST_01",
            "pcr_label": "1",
            "molecular_subtype": "HER2pos",
            "best_model_pcr_probability": "0.21",
            "best_model_predicted_label": "0",
        },
        {
            "patient_id": "ISPY1_TEST_02",
            "pcr_label": "0",
            "molecular_subtype": "TripleNeg",
            "best_model_pcr_probability": "0.85",
            "best_model_predicted_label": "1",
        },
    ])

    report = build_external_data_bridge(
        features_csv=str(features),
        metrics_json=str(metrics),
        predictions_csv=str(predictions),
        canonical_csv=str(canonical),
        output_path=str(output),
        failure_gallery_path=str(gallery),
    )

    assert output.exists()
    assert canonical.exists()
    assert gallery.exists()
    assert report["status"] == "strong"
    assert report["validation"]["status"] == "passed"
    assert report["coverage"]["roles_supported"]["external_pcr_imaging_response_benchmark"] is True
    assert report["coverage"]["roles_supported"]["full_oncotrack_timeline_training"] is False
    assert "not a full NLCare longitudinal" in report["claim_boundary"]

    canonical_rows = list(csv.DictReader(canonical.open(encoding="utf-8")))
    assert canonical_rows[0]["patient_id"] == "BREASTDCEDL:ISPY1_TEST_01"
    assert canonical_rows[0]["her2_status"] == "positive"
    assert canonical_rows[1]["er_status"] == "negative"

    gallery_payload = json.loads(gallery.read_text(encoding="utf-8"))
    assert gallery_payload["summary"]["false_positive_count"] == 1
    assert gallery_payload["summary"]["false_negative_count"] == 1


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
