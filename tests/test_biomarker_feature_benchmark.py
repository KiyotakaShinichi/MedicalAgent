from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.biomarker_feature_benchmark import run_biomarker_feature_benchmark


def test_biomarker_feature_benchmark_exports_lineage_and_leakage_audit(tmp_path):
    rows = []
    subtypes = ["HR+/HER2-", "HER2+", "triple-negative", "HR+/HER2+"]
    stages = ["IIA", "IIB", "IIIA", "IV"]
    regimens = ["AC-T", "TCHP", "carboplatin/paclitaxel", "TCHP then endocrine therapy"]
    for patient_idx in range(32):
        patient_id = f"TEST-BRCA-{patient_idx:03d}"
        label = 1 if patient_idx % 2 == 0 else 0
        subtype = subtypes[patient_idx % len(subtypes)]
        stage = stages[patient_idx % len(stages)]
        regimen = regimens[patient_idx % len(regimens)]
        for cycle in range(1, 4):
            response = 70 - cycle * 4 if label else 25 + cycle * 3
            rows.append(
                {
                    "patient_id": patient_id,
                    "cycle": cycle,
                    "treatment_date": f"2026-01-{cycle:02d}",
                    "age": 42 + patient_idx % 20,
                    "stage": stage,
                    "molecular_subtype": subtype,
                    "regimen": regimen,
                    "pre_wbc": 6.0 - cycle * 0.2,
                    "pre_anc": 3.2 - cycle * 0.1,
                    "pre_hemoglobin": 12.5 - cycle * 0.1,
                    "pre_platelets": 240 - cycle,
                    "nadir_wbc": 3.1 - cycle * 0.2,
                    "nadir_anc": 1.6 - cycle * 0.1,
                    "nadir_hemoglobin": 11.6 - cycle * 0.1,
                    "nadir_platelets": 180 - cycle,
                    "recovery_wbc": 5.1 - cycle * 0.1,
                    "recovery_hemoglobin": 12.0 - cycle * 0.1,
                    "recovery_platelets": 220 - cycle,
                    "mri_tumor_size_cm": 4.0 - (response / 100) * cycle * 0.1,
                    "mri_percent_change_from_baseline": -response / max(cycle, 1),
                    "response_score_percent": response,
                    "max_symptom_severity": 3 + cycle,
                    "symptom_count": cycle,
                    "intervention_count": 1 if cycle == 2 else 0,
                    "dose_delayed": 1 if cycle == 3 and not label else 0,
                    "dose_reduced": 0,
                    "latent_response_strength": response / 100,
                    "toxicity_risk_binary": 0,
                    "urgent_intervention_needed": 0,
                    "support_intervention_needed": 0,
                    "cycle_response_trend_class": "partial_response" if label else "stable",
                    "final_response_category": "complete_response_signal" if label else "stable_disease",
                    "final_cancer_status": "no_evidence_of_disease" if label else "active_disease_needs_review",
                    "treatment_success_binary": label,
                    "maintenance_needed": 1,
                    "final_response_multiclass": "complete_response_signal" if label else "stable_disease",
                }
            )
    source = tmp_path / "rows.csv"
    output = tmp_path / "benchmark.json"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    report = run_biomarker_feature_benchmark(
        source_csv=str(source),
        output_path=str(output),
        predictions_csv_path=str(predictions),
        seed=7,
    )

    assert report["status"] in {"passed", "needs_attention"}
    assert report["leakage_audit"]["forbidden_target_columns_used"] == []
    assert report["feature_lineage"]["tumor_marker_values_synthetic"]["leakage_risk"] == "medium"
    assert report["source_alignment"]["breastdcedl"]["status"] == "mapped_locally"
    assert report["tumor_marker_policy"]["role"] == "monitoring_context_only"
    assert "clinical_timeline_only" in report["classification"]
    assert "clinical_plus_biomarkers" in report["classification"]
    assert "clinical_plus_biomarkers_plus_imaging" in report["classification"]
    assert "enhanced_biomarker_tumor_marker" in report["classification"]
    assert "biomarker_imaging_vs_clinical_auroc_delta" in report["deltas"]
    assert Path(report["artifacts"]["predictions_csv"]).exists()
