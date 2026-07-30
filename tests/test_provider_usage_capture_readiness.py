from __future__ import annotations

import json
from pathlib import Path

from backend.services.provider_usage_capture_readiness import (
    build_provider_usage_capture_readiness,
)


def test_capture_readiness_proves_plumbing_without_provider_call(
    tmp_path: Path,
):
    requests = [
        {
            "provider_reported_input_tokens": None,
            "provider_reported_output_tokens": None,
            "provider_reported_total_tokens": None,
            "estimated_total_tokens": 10,
            "actual_usage_coverage_rate": 0.0,
            "llm_call_latency_ms": 0.0,
        }
    ]
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps({"requests": requests}), encoding="utf-8")
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_text(
        json.dumps(
            {
                "completed": False,
                "requests_with_provider_reported_usage": 0,
                "paired_request_count": 0,
                "reconciliation_metrics": {
                    "estimated_tokens_are_billing_truth": False
                },
            }
        ),
        encoding="utf-8",
    )
    payload = build_provider_usage_capture_readiness(
        cost_report_path=cost,
        reconciliation_path=reconciliation,
        env={},
    )
    assert payload["status"] == "ready_for_nonpatient_provider_capture"
    assert payload["passed_count"] == payload["check_count"]
    assert payload["provider_call_performed_by_this_artifact"] is False
    assert payload["synthetic_fixture_call_count"] == 1
    assert payload["reconciliation_completed"] is False
    assert payload["clinical_validation"] is False


def test_placeholder_credentials_are_not_reported_as_configured():
    payload = build_provider_usage_capture_readiness(
        env={"GROQ_API_KEY": "replace_with_real_key"}
    )
    assert payload["provider_credentials_configured"] is False
    assert payload["configured_provider_names"] == []
