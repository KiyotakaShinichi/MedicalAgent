from __future__ import annotations

import json

from backend.services.adversarial_v6_retrospective import build_retrospective, write_retrospective


def test_retrospective_declares_contamination_without_rerunning_bank():
    result = build_retrospective()
    assert result["was_used_for_tuning"] is True
    assert result["frozen_bank_was_rerun"] is False
    assert result["internal_vs_external"].startswith("internal_")
    assert result["clinical_validation"] is False
    assert result["failed_n"] > 0
    assert "cross_patient_exfiltration" in result["failure_categories"]


def test_retrospective_writer_preserves_anti_overclaim_boundary(tmp_path):
    output = tmp_path / "retrospective.json"
    result = write_retrospective(output)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == result
    assert "not clinical validation" in result["claim_boundary"].lower()
    assert any("independent held-out" in item for item in result["blocked_readings"])
