from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from backend.services.synthetic_automation_staging_readiness import (
    build_synthetic_automation_staging_readiness,
)


def test_repository_synthetic_staging_contract_is_bounded_and_ready(tmp_path) -> None:
    with patch(
        "backend.services.synthetic_automation_staging_readiness._validate_compose",
        return_value={"available": True, "executed": True, "valid": True},
    ):
        report = build_synthetic_automation_staging_readiness(output_path=tmp_path / "result.json")

    assert report["status"] == "ready_for_synthetic_runtime"
    assert report["passed_count"] == report["check_count"]
    assert report["runtime_completed"] is False
    assert report["external_delivery_completed"] is False
    assert report["human_acknowledgement_completed"] is False
    assert report["synthetic_recipient_only"] is True
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False


def test_workflow_contains_real_hmac_check_and_no_patient_payload(tmp_path) -> None:
    workflow = json.loads(
        Path("infra/n8n/synthetic_high_risk_review_alert.json").read_text(encoding="utf-8")
    )
    encoded = json.dumps(workflow)
    assert "createHmac('sha256'" in encoded
    assert "timingSafeEqual" in encoded
    assert "nlcare-synthetic-review@invalid.example" in encoded
    assert workflow["active"] is False
    assert workflow["meta"]["phi_allowed"] is False
    assert workflow["meta"]["requires_manual_mailhog_smtp_credential"] is True
