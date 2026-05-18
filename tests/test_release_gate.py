import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_release_gate import ROOT, run_release_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_release_gate_accepts_yaml_config_with_thresholds(tmp_path):
    artifact = ROOT / "Data" / "test_tmp" / "release_gate_pass.json"
    config = ROOT / "Data" / "test_tmp" / "release_gate_pass.yaml"
    _write_json(
        artifact,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "strong",
            "summary": {"pass_rate": 1.0},
        },
    )
    config.write_text(
        """
artifacts:
  - path: Data/test_tmp/release_gate_pass.json
    required: true
    max_age_days: 1
    accepted_status: [strong]
    metric_thresholds:
      - {path: [summary, pass_rate], op: ">=", value: 1.0}
""".strip(),
        encoding="utf-8",
    )

    report = run_release_gate(config)

    assert report["status"] == "passed"
    assert report["failure_count"] == 0
    assert report["config_path"] == "Data\\test_tmp\\release_gate_pass.yaml" or report["config_path"] == "Data/test_tmp/release_gate_pass.yaml"


def test_release_gate_fails_missing_required_artifact(tmp_path):
    missing = ROOT / "Data" / "test_tmp" / "release_gate_missing.json"
    missing.unlink(missing_ok=True)
    config = ROOT / "Data" / "test_tmp" / "release_gate_missing.yaml"
    config.write_text(
        """
artifacts:
  - path: Data/test_tmp/release_gate_missing.json
    required: true
    accepted_status: [strong]
""".strip(),
        encoding="utf-8",
    )

    report = run_release_gate(config)

    assert report["status"] == "failed"
    assert report["failure_count"] == 1
    assert report["artifacts"][0]["issues"] == ["missing"]


def test_release_gate_fails_metric_regression(tmp_path):
    artifact = ROOT / "Data" / "test_tmp" / "release_gate_metric.json"
    config = ROOT / "Data" / "test_tmp" / "release_gate_metric.yaml"
    _write_json(
        artifact,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "strong",
            "summary": {"pass_rate": 0.8},
        },
    )
    config.write_text(
        """
artifacts:
  - path: Data/test_tmp/release_gate_metric.json
    required: true
    accepted_status: [strong]
    metric_thresholds:
      - {path: [summary, pass_rate], op: ">=", value: 0.95}
""".strip(),
        encoding="utf-8",
    )

    report = run_release_gate(config)

    assert report["status"] == "failed"
    assert report["failure_count"] == 1
    assert "failed >= 0.95" in report["artifacts"][0]["issues"][0]
