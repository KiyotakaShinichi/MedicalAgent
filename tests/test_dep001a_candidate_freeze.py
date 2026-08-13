from __future__ import annotations

import json
from pathlib import Path

from backend.services.dep001a_candidate_freeze import FROZEN_PATHS, build_freeze_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_freeze_is_hash_bound_and_keeps_dep001_blocked(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    result = build_freeze_manifest(output)
    assert result["status"] == "frozen_for_external_no_read_evaluation"
    assert result["dep001_status"] == "blocked_pending_new_external_no_read_holdout"
    assert result["new_external_holdout_completed"] is False
    assert result["old_frozen_holdout_rerun"] is False
    assert result["clinical_validation"] is False
    assert {row["path"] for row in result["files"]} == set(FROZEN_PATHS)
    assert all(len(row["sha256"]) == 64 for row in result["files"])
    assert json.loads(output.read_text(encoding="utf-8"))["model_version"].endswith("v10")


def test_external_protocol_has_no_authored_examples_or_completion_claim() -> None:
    text = (ROOT / "reports/dep001a_external_no_read_protocol.md").read_text(encoding="utf-8").lower()
    assert "prepared, not completed" in text
    assert "at least 300 cases" in text
    assert "unsafe pass rate exactly `0`" in text
    assert "production healthcare readiness" in text
