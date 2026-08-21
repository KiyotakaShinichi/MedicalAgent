"""Normalized admin benchmark artifact response.

The older admin endpoints return raw artifacts with slightly different shapes.
This adapter gives the frontend and generated OpenAPI schema one stable
contract while keeping the existing endpoints backward compatible.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.services.benchmark_registry import BENCHMARK_SPECS, ROOT_DIR


CLAIM_BOUNDARY = (
    "Admin benchmark artifacts are engineering release evidence. They do not "
    "establish clinical safety, clinician validation, or real-world model utility."
)


def get_normalized_benchmark_artifact(artifact_id: str) -> dict[str, Any]:
    spec = next((item for item in BENCHMARK_SPECS if item.get("id") == artifact_id), None)
    if spec is None:
        return _envelope(
            status="missing",
            headline_metric=None,
            metrics={},
            rows=[],
            artifact_path=None,
            last_run_at=None,
            can_rerun=False,
            errors=[f"Unknown benchmark artifact id: {artifact_id}"],
        )

    payload, artifact_path, errors = _load_payload(spec)
    metrics = _extract_metrics(payload, spec.get("metrics") or {}) if payload else {}
    rows = _extract_rows(payload)
    status = str((payload or {}).get("status") or _dig(payload or {}, ["summary", "status"]) or "missing")
    return _envelope(
        status=status,
        headline_metric=_headline(metrics),
        metrics=metrics,
        rows=rows,
        artifact_path=artifact_path,
        last_run_at=_last_run_at(payload or {}),
        claim_boundary=str((payload or {}).get("claim_boundary") or CLAIM_BOUNDARY),
        can_rerun=True,
        errors=errors,
    )


def _load_payload(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    rel_path = str(spec.get("path") or "")
    path = ROOT_DIR / rel_path
    if not path.exists():
        fallback = spec.get("fallback")
        if fallback:
            fallback_path = ROOT_DIR / str(fallback)
            if fallback_path.exists():
                path = fallback_path
                rel_path = str(fallback)
        if not path.exists():
            return None, rel_path or None, ["artifact_missing"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), rel_path, []
    except Exception as exc:  # noqa: BLE001 - admin artifact adapter should not crash the dashboard
        return None, rel_path, [f"artifact_unparseable: {exc!s}"]


def _extract_metrics(payload: dict[str, Any], metric_paths: dict[str, list[Any]]) -> dict[str, Any]:
    if not metric_paths:
        summary = payload.get("summary")
        return summary if isinstance(summary, dict) else {"status": payload.get("status")}
    return {name: _dig(payload, path) for name, path in metric_paths.items()}


def _extract_rows(payload: dict[str, Any] | None) -> list[Any]:
    if not payload:
        return []
    for key in ("rows", "cases", "benchmarks", "artifacts", "scenarios", "entries", "traces"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    summary = payload.get("summary")
    return [summary] if isinstance(summary, dict) else []


def _headline(metrics: dict[str, Any]) -> str | None:
    for key in ("pass_rate", "status", "hard_failures", "unsafe_answer_rate", "case_count"):
        if key in metrics and metrics[key] is not None:
            return f"{key}={metrics[key]}"
    for key, value in metrics.items():
        if value is not None:
            return f"{key}={value}"
    return None


def _last_run_at(payload: dict[str, Any]) -> str | None:
    value = payload.get("generated_at") or payload.get("generated_at_iso") or payload.get("timestamp")
    if isinstance(value, str):
        return value
    return None


def _dig(payload: Any, path: list[Any]) -> Any:
    value = payload
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def _envelope(
    *,
    status: str,
    headline_metric: str | None,
    metrics: dict[str, Any],
    rows: list[Any],
    artifact_path: str | None,
    last_run_at: str | None,
    claim_boundary: str = CLAIM_BOUNDARY,
    can_rerun: bool,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "admin_benchmark_response_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "headline_metric": headline_metric,
        "metrics": metrics,
        "rows": rows,
        "artifact_path": artifact_path,
        "last_run_at": last_run_at,
        "claim_boundary": claim_boundary,
        "can_rerun": can_rerun,
        "errors": errors,
    }


__all__ = ["get_normalized_benchmark_artifact"]
