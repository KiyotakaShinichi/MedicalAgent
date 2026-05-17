from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.public_biomarker_mapping import build_public_biomarker_mapping_readiness


def test_public_biomarker_mapping_readiness_maps_breastdcedl_and_boundaries(tmp_path):
    features = tmp_path / "breastdcedl_features.csv"
    pd.DataFrame([
        {
            "patient_id": "ISPY1_1001",
            "pcr_label": 0,
            "age": 52,
            "molecular_subtype": "HRposHER2neg",
            "baseline_longest_diameter_mm": 42,
            "early_enhancement_mean": 0.8,
            "washout_mean": 0.1,
        }
    ]).to_csv(features, index=False)
    output = tmp_path / "mapping.json"

    report = build_public_biomarker_mapping_readiness(
        breastdcedl_features_path=str(features),
        output_path=str(output),
    )

    assert report["status"] == "ready"
    assert report["datasets"]["breastdcedl"]["status"] == "mapped"
    assert report["datasets"]["aacr_genie_bpc_brca"]["status"] == "future_access_candidate"
    assert "standalone recurrence" in report["tumor_marker_boundary"]
    assert "clinical_plus_biomarkers_plus_imaging" in report["three_stage_ablation_plan"]
    assert output.exists()
