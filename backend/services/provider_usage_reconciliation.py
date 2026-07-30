"""Reconcile provider-reported token usage with local estimates when available."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_SOURCE_PATH = Path("Data/evals/ops/latest_cost_latency_report.json")
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/ops/latest_provider_usage_reconciliation.json"
)
MIN_PAIRED_REQUESTS = 30
MIN_ACTUAL_COVERAGE = 0.8

CLAIM_BOUNDARY = (
    "Provider-reported token counts are operational telemetry, not audited "
    "billing. Local character-based estimates remain visibly separate. This "
    "artifact is not clinical validation or production healthcare evidence."
)


def build_provider_usage_reconciliation(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = _read_json(source_path)
    requests = [
        row for row in payload.get("requests") or [] if isinstance(row, dict)
    ]
    paired = []
    for row in requests:
        actual = _positive(row.get("provider_reported_total_tokens"))
        estimate = _positive(row.get("estimated_total_tokens"))
        if actual is None or estimate is None:
            continue
        paired.append(
            {
                "request_id": row.get("request_id"),
                "route": row.get("route"),
                "actual_total_tokens": actual,
                "estimated_total_tokens": estimate,
                "absolute_percentage_error": round(abs(estimate - actual) / actual, 6),
                "estimate_to_actual_ratio": round(estimate / actual, 6),
            }
        )

    actual_count = sum(
        1
        for row in requests
        if _positive(row.get("provider_reported_total_tokens")) is not None
    )
    coverage = round(actual_count / len(requests), 4) if requests else 0.0
    configured = _provider_configured(env if env is not None else os.environ)
    completed = len(paired) >= MIN_PAIRED_REQUESTS and coverage >= MIN_ACTUAL_COVERAGE
    if completed:
        status = "reconciled"
        reason = "Minimum paired sample and provider-usage coverage contracts met."
    elif actual_count:
        status = "insufficient_provider_sample"
        reason = "Some provider usage exists, but paired sample or coverage is below target."
    elif configured:
        status = "configured_no_observations"
        reason = "A provider credential is configured, but no provider-reported usage was captured."
    else:
        status = "blocked_configuration"
        reason = "No provider credential or provider-reported usage is available."

    errors = [row["absolute_percentage_error"] for row in paired]
    ratios = [row["estimate_to_actual_ratio"] for row in paired]
    return {
        "schema_version": "provider_usage_reconciliation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed": completed,
        "reason": reason,
        "source_artifact": str(source_path).replace("\\", "/"),
        "provider_configured": configured,
        "request_count": len(requests),
        "requests_with_provider_reported_usage": actual_count,
        "actual_usage_coverage_rate": coverage,
        "paired_request_count": len(paired),
        "minimum_paired_requests": MIN_PAIRED_REQUESTS,
        "minimum_actual_usage_coverage_rate": MIN_ACTUAL_COVERAGE,
        "reconciliation_metrics": {
            "mean_absolute_percentage_error": (
                round(sum(errors) / len(errors), 6) if errors else None
            ),
            "median_estimate_to_actual_ratio": (
                round(median(ratios), 6) if ratios else None
            ),
            "estimated_tokens_are_billing_truth": False,
        },
        "paired_requests": paired[:100],
        "execution_policy": {
            "automatic_paid_provider_probe": False,
            "reason": (
                "The evaluation does not spend provider quota automatically. "
                "Capture normal non-patient test traffic with provider metadata, "
                "then rerun this artifact."
            ),
        },
        "production_ready": False,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_provider_usage_reconciliation(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = build_provider_usage_reconciliation(source_path, env=env)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _provider_configured(env: dict[str, str]) -> bool:
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"):
        value = str(env.get(name) or "").strip()
        if value and "replace" not in value.lower() and "placeholder" not in value.lower():
            return True
    return False


def _positive(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


__all__ = [
    "build_provider_usage_reconciliation",
    "write_provider_usage_reconciliation",
]
