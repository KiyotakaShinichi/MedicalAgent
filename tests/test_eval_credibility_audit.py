from __future__ import annotations

import json

from backend.services.eval_credibility_audit import run_eval_credibility_audit


def test_eval_credibility_audit_surfaces_metadata_and_overclaim_risks(tmp_path):
    artifact_dir = tmp_path / "Data/evals/safety"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "sample.json"
    artifact.write_text(json.dumps({
        "status": "strong",
        "total_n": 2,
        "pass_count": 2,
        "fail_count": 0,
        "pass_rate": 1.0,
        "clinical_validation": False,
        "claim_boundary": "engineering self-test only",
        "rows": [
            {
                "authored_by": "engineering_internal",
                "authored_date": "2026-05-25",
                "case_source": "unit_test",
                "was_used_for_tuning": False,
                "contamination_note": "internal test case",
            }
        ],
    }), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "artifacts:\n"
        "  - path: Data/evals/safety/sample.json\n"
        "    required: false\n"
        "    accepted_status: [strong]\n",
        encoding="utf-8",
    )

    payload = run_eval_credibility_audit(config_path=config, output_path=tmp_path / "audit.json")

    assert payload["status"] == "acceptable"
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["n_size_metadata_rate"] == 1.0
    assert payload["summary"]["contamination_disclosure_rate"] == 1.0
    assert payload["summary"]["perfect_internal_score_count"] == 1
