"""Verify provider-usage capture plumbing without spending provider quota."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.llm_telemetry import (
    provider_usage,
    record_llm_call,
    reset_llm_telemetry,
    snapshot_llm_telemetry,
    start_llm_telemetry,
)
from backend.services.provider_usage_reconciliation import (
    MIN_ACTUAL_COVERAGE,
    MIN_PAIRED_REQUESTS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_REPORT = Path("Data/evals/ops/latest_cost_latency_report.json")
DEFAULT_RECONCILIATION = Path(
    "Data/evals/ops/latest_provider_usage_reconciliation.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/ops/latest_provider_usage_capture_readiness.json"
)

CLAIM_BOUNDARY = (
    "This artifact proves local telemetry plumbing with a synthetic usage "
    "fixture. It does not perform a provider call, spend quota, establish "
    "billing truth, complete the 30-call reconciliation, validate clinical "
    "behavior, or establish production healthcare readiness."
)


def build_provider_usage_capture_readiness(
    *,
    root: str | Path = ROOT,
    cost_report_path: str | Path = DEFAULT_COST_REPORT,
    reconciliation_path: str | Path = DEFAULT_RECONCILIATION,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    cost_report = _read_json(_resolve(repo, cost_report_path))
    reconciliation = _read_json(_resolve(repo, reconciliation_path))
    requests = [
        row
        for row in cost_report.get("requests") or []
        if isinstance(row, dict)
    ]
    required_fields = {
        "provider_reported_input_tokens",
        "provider_reported_output_tokens",
        "provider_reported_total_tokens",
        "estimated_total_tokens",
        "actual_usage_coverage_rate",
        "llm_call_latency_ms",
    }
    fixture = {
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }
    }
    parsed = provider_usage(fixture)
    token = start_llm_telemetry()
    try:
        record_llm_call(
            provider="synthetic_fixture",
            model="fixture",
            operation="capture_readiness",
            latency_ms=1.25,
            usage=parsed,
        )
        telemetry = snapshot_llm_telemetry()
    finally:
        reset_llm_telemetry(token)

    local_llm_text = (repo / "backend/services/local_llm.py").read_text(
        encoding="utf-8"
    )
    support_response_text = (
        repo / "backend/services/support_chat_response.py"
    ).read_text(encoding="utf-8")
    values = env if env is not None else os.environ
    configured_names = [
        name
        for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
        if _configured(values.get(name))
    ]
    checks = [
        _check("provider_usage_fixture_parsed", parsed == {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        }),
        _check(
            "provider_usage_fixture_recorded",
            telemetry["provider_reported_call_count"] == 1
            and telemetry["actual_usage_coverage_rate"] == 1.0,
        ),
        _check("telemetry_retains_no_content", telemetry["content_retained"] is False),
        _check(
            "cost_report_has_reconciliation_fields",
            bool(requests)
            and all(required_fields <= set(row) for row in requests),
        ),
        _check(
            "live_provider_wrappers_call_usage_extractor",
            "provider_usage(" in local_llm_text
            and "provider_usage(" in support_response_text,
        ),
        _check(
            "reconciliation_keeps_estimates_separate",
            reconciliation.get("reconciliation_metrics", {}).get(
                "estimated_tokens_are_billing_truth"
            )
            is False,
        ),
    ]
    ready = all(check["passed"] for check in checks)
    completed = reconciliation.get("completed") is True
    return {
        "schema_version": "provider_usage_capture_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "reconciliation_completed"
            if completed
            else "ready_for_nonpatient_provider_capture"
            if ready
            else "needs_attention"
        ),
        "checks": checks,
        "passed_count": sum(check["passed"] for check in checks),
        "check_count": len(checks),
        "provider_credentials_configured": bool(configured_names),
        "configured_provider_names": configured_names,
        "provider_call_performed_by_this_artifact": False,
        "synthetic_fixture_call_count": 1,
        "real_provider_observations_collected": reconciliation.get(
            "requests_with_provider_reported_usage", 0
        ),
        "paired_request_count": reconciliation.get("paired_request_count", 0),
        "required_paired_request_count": MIN_PAIRED_REQUESTS,
        "required_actual_usage_coverage_rate": MIN_ACTUAL_COVERAGE,
        "reconciliation_completed": completed,
        "capture_protocol": [
            "Configure one supported provider credential outside source control.",
            "Run 30 or more non-patient test prompts through the normal API path.",
            "Do not log prompt or response content in telemetry artifacts.",
            "Refresh the cost/latency report, then rerun provider reconciliation.",
            "Report provider counts separately from local token estimates.",
        ],
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_provider_usage_capture_readiness(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_provider_usage_capture_readiness(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _configured(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(
        text
        and "replace" not in text.lower()
        and "placeholder" not in text.lower()
    )


def _check(check_id: str, passed: bool) -> dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed)}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


__all__ = [
    "build_provider_usage_capture_readiness",
    "write_provider_usage_capture_readiness",
]
