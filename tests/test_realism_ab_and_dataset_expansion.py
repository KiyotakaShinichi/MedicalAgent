from __future__ import annotations

import json

import pandas as pd

from backend.services.dataset_expansion_deep_search import build_dataset_expansion_deep_search
from backend.services.realism_candidate_ab_gate import run_realism_candidate_ab_gate


def test_realism_candidate_ab_gate_blocks_production_replacement(tmp_path):
    current = tmp_path / "current.csv"
    candidate = tmp_path / "candidate.csv"
    candidate_artifact = tmp_path / "candidate_artifact.json"
    _write_rows(current, candidate, n_patients=48)
    candidate_artifact.write_text(json.dumps({
        "status": "candidate",
        "before_after_gaps": {
            "age": {"gap_improved": True},
            "baseline_tumor_size_mm_proxy": {"gap_improved": True},
        },
        "realism_candidate_decision": {
            "production_replacement_allowed": False,
        },
    }), encoding="utf-8")

    report = run_realism_candidate_ab_gate(
        current_csv=str(current),
        candidate_csv=str(candidate),
        candidate_artifact_path=str(candidate_artifact),
        output_path=str(tmp_path / "ab_gate.json"),
        legacy_output_path=str(tmp_path / "legacy.json"),
    )

    assert report["status"] in {"candidate", "needs_attention"}
    assert report["current"]["leakage"]["status"] == "passed"
    assert report["candidate"]["leakage"]["status"] == "passed"
    assert report["recommendation"]["decision"] == "keep_current_default"
    assert report["recommendation"]["production_replacement_allowed"] is False
    assert "not clinical validation" in report["claim_boundary"]
    assert (tmp_path / "legacy.json").exists()


def test_dataset_expansion_deep_search_prioritizes_treatment_and_imaging_sources(tmp_path):
    report = build_dataset_expansion_deep_search(
        output_path=str(tmp_path / "datasets.json"),
        doc_path=str(tmp_path / "datasets.md"),
    )

    assert report["status"] == "strong"
    assert report["dataset_count"] >= 8
    ids = {row["id"] for row in report["candidates"]}
    assert {"genie_bpc_brca_public", "duke_breast_mri", "tcga_brca_gdc"} <= ids
    assert report["highest_priority"][0]["id"] == "genie_bpc_brca_public"
    assert "patient-facing treatment recommendation" in report["blocked_claims"]
    assert (tmp_path / "datasets.md").exists()


def _write_rows(current_path, candidate_path, *, n_patients: int) -> None:
    rows = []
    subtypes = ["HR+/HER2-", "HER2+", "triple-negative"]
    regimens = ["dose-dense AC then paclitaxel", "TCHP", "endocrine therapy"]
    for patient_idx in range(n_patients):
        label = int(patient_idx % 2 == 0)
        for cycle in range(1, 5):
            response = 18 + label * 32 + cycle * 4 - (patient_idx % 5)
            tumor_cm = max(0.6, 5.0 - response / 35 + (patient_idx % 3) * 0.15)
            rows.append({
                "patient_id": f"P-{patient_idx:03d}",
                "cycle": cycle,
                "treatment_date": f"2026-01-{cycle:02d}",
                "age": 35 + patient_idx % 32,
                "stage": "IIA" if patient_idx % 3 else "IIIA",
                "molecular_subtype": subtypes[patient_idx % len(subtypes)],
                "regimen": regimens[patient_idx % len(regimens)],
                "pre_wbc": 5.0 + label * 0.4,
                "pre_anc": 2.8 + label * 0.2,
                "pre_hemoglobin": 12.0 + label * 0.4,
                "pre_platelets": 230 + label * 20,
                "nadir_wbc": 2.8 + label * 0.4,
                "nadir_anc": 1.3 + label * 0.3,
                "nadir_hemoglobin": 10.8 + label * 0.6,
                "nadir_platelets": 145 + label * 25,
                "recovery_wbc": 4.6 + label * 0.4,
                "recovery_hemoglobin": 11.5 + label * 0.4,
                "recovery_platelets": 195 + label * 20,
                "mri_tumor_size_cm": tumor_cm,
                "mri_percent_change_from_baseline": -response,
                "response_score_percent": response,
                "max_symptom_severity": 5 - label + patient_idx % 2,
                "symptom_count": 2 + patient_idx % 3,
                "intervention_count": int(not label),
                "dose_delayed": int(not label and cycle % 2 == 0),
                "dose_reduced": int(not label and cycle % 3 == 0),
                "latent_response_strength": 0.7 if label else 0.3,
                "toxicity_risk_binary": int(not label),
                "urgent_intervention_needed": int(not label and cycle == 3),
                "support_intervention_needed": int(not label),
                "cycle_response_trend_class": "improving" if label else "worsening",
                "final_response_category": "response_signal" if label else "limited_response_signal",
                "final_cancer_status": "monitor_only",
                "treatment_success_binary": label,
                "maintenance_needed": label,
                "final_response_multiclass": "response_signal" if label else "limited_response_signal",
            })
    current = pd.DataFrame(rows)
    candidate = current.copy()
    candidate["age"] = candidate["age"] + 8
    candidate["mri_tumor_size_cm"] = candidate["mri_tumor_size_cm"] * 1.4
    current.to_csv(current_path, index=False)
    candidate.to_csv(candidate_path, index=False)
