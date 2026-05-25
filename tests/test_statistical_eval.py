from __future__ import annotations

import json
from pathlib import Path

from backend.services.statistical_eval import mean_interval, two_proportion_delta, wilson_interval
from scripts.run_statistical_eval_summary import build_statistical_summary


def test_wilson_interval_contains_estimate():
    interval = wilson_interval(188, 200)
    assert interval["estimate"] == 0.94
    assert interval["ci_low"] < interval["estimate"] < interval["ci_high"]
    assert interval["total_n"] == 200


def test_mean_interval_reports_uncertainty_for_multiple_values():
    interval = mean_interval([0.8, 0.9, 1.0])
    assert interval["total_n"] == 3
    assert interval["ci_low"] < interval["estimate"] < interval["ci_high"]


def test_two_proportion_delta_schema():
    delta = two_proportion_delta(
        baseline_successes=188,
        baseline_total=200,
        candidate_successes=31,
        candidate_total=32,
    )
    assert delta["delta"] > 0
    assert "baseline" in delta
    assert "candidate" in delta


def test_build_statistical_summary_writes_artifact(tmp_path: Path):
    output = tmp_path / "stats.json"
    report = build_statistical_summary(output)
    assert report["metric_count"] >= 4
    assert report["status"] in {"acceptable", "needs_attention"}
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"].startswith("statistical_eval_v1")
    assert all("claim_boundary" in metric for metric in payload["metrics"])
