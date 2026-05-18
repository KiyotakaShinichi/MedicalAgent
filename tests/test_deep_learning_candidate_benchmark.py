from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.services.deep_learning_candidate_benchmark import run_deep_learning_candidate_benchmark


def test_deep_learning_candidate_benchmark_outputs_classification_regression_and_boundaries(tmp_path):
    rows = []
    subtypes = ["HR+/HER2-", "HER2+", "triple-negative", "HR+/HER2+"]
    stages = ["IIA", "IIB", "IIIA", "IV"]
    regimens = ["AC-T", "TCHP", "carboplatin/paclitaxel", "endocrine therapy"]
    for patient_idx in range(48):
        patient_id = f"DLTEST-{patient_idx:03d}"
        label = int(patient_idx % 2 == 0)
        for cycle in range(1, 7):
            response = (12 + cycle * 11 + patient_idx % 4) if label else (8 + cycle * 3 + patient_idx % 5)
            rows.append({
                "patient_id": patient_id,
                "cycle": cycle,
                "treatment_date": f"2026-01-{cycle:02d}",
                "age": 38 + patient_idx % 30,
                "stage": stages[patient_idx % len(stages)],
                "molecular_subtype": subtypes[patient_idx % len(subtypes)],
                "regimen": regimens[patient_idx % len(regimens)],
                "pre_wbc": 6.0 - cycle * 0.12,
                "pre_anc": 3.2 - cycle * 0.08,
                "pre_hemoglobin": 12.4 - cycle * 0.07,
                "pre_platelets": 250 - cycle,
                "nadir_wbc": 3.4 - cycle * 0.12,
                "nadir_anc": 1.7 - cycle * 0.08,
                "nadir_hemoglobin": 11.7 - cycle * 0.06,
                "nadir_platelets": 190 - cycle,
                "recovery_wbc": 5.2 - cycle * 0.05,
                "recovery_hemoglobin": 12.0 - cycle * 0.04,
                "recovery_platelets": 230 - cycle,
                "mri_tumor_size_cm": 5.0 - response * 0.03,
                "mri_percent_change_from_baseline": -response,
                "response_score_percent": response,
                "max_symptom_severity": 3 + (cycle % 3),
                "symptom_count": cycle % 4,
                "intervention_count": int(cycle == 4 and not label),
                "dose_delayed": int(cycle == 5 and not label),
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
            })

    source = tmp_path / "rows.csv"
    output = tmp_path / "dl_report.json"
    pd.DataFrame(rows).to_csv(source, index=False)

    report = run_deep_learning_candidate_benchmark(
        source_csv=str(source),
        output_path=str(output),
        model_dir=str(tmp_path / "models"),
        epochs=3,
        seed=11,
    )

    assert output.exists()
    assert report["status"] in {"strong", "acceptable", "needs_attention"}
    assert "with_genetic_context" in report["models"]
    assert "without_genetic_context" in report["models"]
    assert "with_treatment_context" in report["models"]
    assert "with_genetic_and_treatment_context" in report["models"]
    assert "classification" in report["best_models"]
    assert "regression" in report["best_models"]
    assert "classification" in report["models"]["with_genetic_context"]["sequence_mlp"]
    assert "regression" in report["models"]["with_genetic_context"]["sequence_mlp"]
    assert report["genetic_context_ablation"]["decision"] in {
        "context_only_no_promotion",
        "candidate_for_external_validation_only",
    }
    assert report["treatment_context_ablation"]["decision"] == "context_only_no_treatment_recommendation"
    assert "not treatment recommendations" in report["treatment_feature_boundary"]
    assert "not real patient response" in report["claim_boundary"]
    assert Path(report["best_model"]["artifact_path"]).exists()
