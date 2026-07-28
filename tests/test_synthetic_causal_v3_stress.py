from pathlib import Path

from backend.services.synthetic_causal_v3_stress import SCENARIOS, build_synthetic_causal_v3_stress


def test_causal_v3_runs_multiple_seeds_and_never_promotes(tmp_path: Path):
    report = build_synthetic_causal_v3_stress(tmp_path / "v3.json", seeds=3, n_train=180, n_test=100)
    assert report["seed_count"] == 3
    assert len(report["seed_level_rows"]) == 3 * len(SCENARIOS)
    assert report["model_promotion_decision"] == "HOLD"
    assert report["clinical_validation"] is False
    assert report["realism_claim"] is False
    assert set(report["paired_seed_summaries"]) == set(SCENARIOS)


def test_treatment_context_is_declared_non_causal(tmp_path: Path):
    report = build_synthetic_causal_v3_stress(tmp_path / "v3.json", seeds=2, n_train=120, n_test=80)
    assert report["explicit_non_causal_context"] == ["treatment_context"]
    assert "treatment effect estimation" in report["blocked_uses"]
    assert "no clinical validation" in report["claim_boundary"].lower()
