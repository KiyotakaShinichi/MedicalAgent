from backend.services.realtime_ood_gate import assess_realtime_ood, run_realtime_ood_eval


NORMAL_ROW = {
    "age": 52,
    "cycle": 3,
    "stage": "II",
    "molecular_subtype": "HR+/HER2-",
    "regimen": "synthetic regimen",
    "pre_wbc": 5.2,
    "pre_anc": 2.4,
    "pre_hemoglobin": 12.0,
    "pre_platelets": 220,
    "nadir_wbc": 2.1,
    "nadir_anc": 1.1,
    "nadir_hemoglobin": 10.8,
    "nadir_platelets": 160,
    "recovery_wbc": 4.8,
    "recovery_hemoglobin": 11.7,
    "recovery_platelets": 210,
    "mri_tumor_size_cm": 2.4,
    "mri_percent_change_from_baseline": -22.0,
    "max_symptom_severity": 3,
    "symptom_count": 2,
}


def test_realtime_ood_allows_normal_synthetic_row():
    result = assess_realtime_ood(NORMAL_ROW)
    assert result.severity == "none"
    assert result.action == "allow"


def test_realtime_ood_severe_for_extreme_lab():
    result = assess_realtime_ood({**NORMAL_ROW, "pre_wbc": 9999})
    assert result.severity == "severe"
    assert result.action == "abstain_or_clinician_review"


def test_realtime_ood_detects_unknown_unit():
    result = assess_realtime_ood({**NORMAL_ROW, "pre_wbc_unit": "bananas"})
    assert result.severity in {"moderate", "severe"}
    assert any("unit" in reason for reason in result.reasons)


def test_realtime_ood_eval_writes_expected_metrics(tmp_path):
    payload = run_realtime_ood_eval(output_path=tmp_path / "ood.json", baseline_csv="missing.csv")
    assert payload["summary"]["ood_detection_rate"] >= 0.8
    assert payload["summary"]["severe_ood_abstention_rate"] == 1.0
