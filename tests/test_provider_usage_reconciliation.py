import json

from backend.services.provider_usage_reconciliation import (
    build_provider_usage_reconciliation,
)


def _write(path, requests):
    path.write_text(json.dumps({"requests": requests}), encoding="utf-8")


def test_missing_provider_usage_is_not_fabricated(tmp_path):
    source = tmp_path / "cost.json"
    _write(source, [{"estimated_total_tokens": 120} for _ in range(40)])
    result = build_provider_usage_reconciliation(source, env={})
    assert result["status"] == "blocked_configuration"
    assert result["completed"] is False
    assert result["requests_with_provider_reported_usage"] == 0
    assert result["reconciliation_metrics"]["mean_absolute_percentage_error"] is None
    assert result["clinical_validation"] is False


def test_reconciles_only_when_sample_and_coverage_are_credible(tmp_path):
    source = tmp_path / "cost.json"
    rows = [
        {
            "request_id": str(index),
            "route": "full_api_path",
            "provider_reported_total_tokens": 100,
            "estimated_total_tokens": 110,
        }
        for index in range(30)
    ]
    _write(source, rows)
    result = build_provider_usage_reconciliation(source, env={})
    assert result["status"] == "reconciled"
    assert result["completed"] is True
    assert result["actual_usage_coverage_rate"] == 1.0
    assert result["reconciliation_metrics"]["mean_absolute_percentage_error"] == 0.1


def test_configured_without_observations_stays_incomplete(tmp_path):
    source = tmp_path / "cost.json"
    _write(source, [{"estimated_total_tokens": 120}])
    result = build_provider_usage_reconciliation(
        source, env={"GROQ_API_KEY": "configured-test-key"}
    )
    assert result["status"] == "configured_no_observations"
    assert result["completed"] is False


def test_normal_api_probe_rows_can_supply_paired_usage(tmp_path):
    source = tmp_path / "cost.json"
    _write(source, [])
    probe = tmp_path / "probe.json"
    rows = [
        {
            "request_id": f"probe-{index}",
            "route": "/me/chat",
            "provider_reported_total_tokens": 100,
            "estimated_total_tokens": 90,
        }
        for index in range(30)
    ]
    probe.write_text(
        json.dumps({"normal_api_path": True, "requests": rows}),
        encoding="utf-8",
    )
    result = build_provider_usage_reconciliation(
        source, probe_path=probe, env={}
    )
    assert result["completed"] is True
    assert result["status"] == "reconciled"
    assert result["paired_request_count"] == 30
    assert result["actual_usage_coverage_rate"] == 1.0
    assert result["normal_api_probe_request_count"] == 30
