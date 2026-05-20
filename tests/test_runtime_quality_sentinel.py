from __future__ import annotations

from pathlib import Path

from backend.services.runtime_quality_sentinel import build_runtime_quality_sentinel


def test_runtime_quality_sentinel_writes_thresholded_snapshot(tmp_path: Path):
    output = tmp_path / "runtime_quality_sentinel.json"

    payload = build_runtime_quality_sentinel(
        output_path=output,
        thresholds={"latency_p95_ms": 1_000_000.0},
    )

    assert output.exists()
    assert payload["schema_version"] == "runtime_quality_sentinel_v1"
    assert payload["status"] in {"strong", "needs_attention"}
    assert "unsupported_claim_rate" in payload["summary"]
    assert "unsafe_answer_rate" in payload["summary"]
    assert "latency_p95_ms" in payload["summary"]
    assert payload["claim_boundary"].startswith("Runtime quality sentinel")
