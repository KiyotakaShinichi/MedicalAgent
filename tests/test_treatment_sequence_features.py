from __future__ import annotations

import csv
import json

from backend.services.treatment_sequence_features import build_treatment_sequence_feature_eval


def test_treatment_sequence_features_capture_combo_patterns_without_recommendations(tmp_path):
    source = tmp_path / "temporal_rows.csv"
    sequence_csv = tmp_path / "sequences.csv"
    output = tmp_path / "treatment_sequence_eval.json"
    rows = []
    for cycle in range(1, 4):
        rows.append({
            "patient_id": "P-HER2",
            "cycle": cycle,
            "age": 49,
            "stage": "IIIA",
            "molecular_subtype": "HR+/HER2+",
            "regimen": "TCHP then endocrine therapy",
            "dose_delayed": "0",
            "dose_reduced": "0",
            "intervention_count": "0",
        })
    for cycle in range(1, 4):
        rows.append({
            "patient_id": "P-TNBC",
            "cycle": cycle,
            "age": 44,
            "stage": "IIB",
            "molecular_subtype": "triple-negative",
            "regimen": "paclitaxel + carboplatin then AC",
            "dose_delayed": "1" if cycle == 2 else "0",
            "dose_reduced": "0",
            "intervention_count": "1" if cycle == 2 else "0",
        })
    _write_csv(source, rows)

    report = build_treatment_sequence_feature_eval(
        source_csv=str(source),
        sequence_csv=str(sequence_csv),
        output_path=str(output),
    )

    assert output.exists()
    assert sequence_csv.exists()
    assert report["status"] in {"strong", "acceptable"}
    assert report["patient_count"] == 2
    assert "does not compare real treatment efficacy" in report["claim_boundary"]
    assert report["modality_counts"]["chemotherapy"] == 2
    assert report["modality_counts"]["surgery_planned"] == 2
    assert report["modality_counts"]["radiation_planned"] == 2
    assert report["modality_counts"]["targeted_anti_her2"] == 1
    assert report["modality_counts"]["endocrine"] == 1
    assert report["modality_counts"]["supportive_care"] == 1

    sequence_rows = list(csv.DictReader(sequence_csv.open(encoding="utf-8")))
    patterns = {row["patient_id"]: row["treatment_combination_pattern"] for row in sequence_rows}
    assert "targeted_anti_her2" in patterns["P-HER2"]
    assert "supportive_care" in patterns["P-TNBC"]

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pattern_count"] == 2


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
