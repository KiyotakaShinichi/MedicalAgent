from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_service_health_snapshot.json"


def build_service_health_snapshot(
    *,
    release_gate_path: str = "Data/evals/benchmark/latest_benchmark_summary.json",
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    registry = _read_json(_resolve(release_gate_path))
    benchmarks = registry.get("benchmarks") or []
    stale = [row for row in benchmarks if row.get("freshness") == "stale"]
    failed = [row for row in benchmarks if row.get("status") in {"failed", "missing", "error", "unavailable"}]
    payload = {
        "schema_version": "service_health_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if not failed else "needs_attention",
        "metrics": {
            "rag_latency_ms": None,
            "retrieval_failure_rate": None,
            "citation_validation_failure_rate": None,
            "post_generation_validator_trigger_rate": None,
            "abstention_rate": None,
            "unsafe_block_rate": None,
            "release_gate_status": registry.get("status"),
            "stale_artifact_count": len(stale),
            "failed_benchmark_count": len(failed),
        },
        "stale_artifacts": [row.get("id") for row in stale[:50]],
        "failed_artifacts": [row.get("id") for row in failed[:50]],
        "claim_boundary": "PoC operational health snapshot only; no production SLO/SLA or compliance claim.",
    }
    _write_json(_resolve(output_path), payload)
    _write_doc()
    return payload


def _write_doc() -> None:
    path = _resolve("docs/ops_health_metrics.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Ops Health Metrics\n\n"
        "This PoC snapshot tracks release-gate status, stale artifacts, failed benchmarks, "
        "and placeholders for RAG latency, retrieval failures, citation-validation failures, "
        "post-generation validator triggers, abstention rate, and unsafe-block rate. "
        "It is not production SRE monitoring.\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
