from backend.services.automation_channel_drill import (
    build_automation_channel_drill,
)


def test_loopback_channel_drill_exercises_signed_dispatch_and_receipt(tmp_path):
    payload = build_automation_channel_drill(
        tmp_path / "channel.json",
        attempts=10,
    )
    assert payload["status"] == "strong"
    assert payload["pass_rate"] == 1.0
    assert payload["local_network_delivery_performed"] is True
    assert payload["external_delivery_performed"] is False
    assert payload["live_n8n_delivery_completed"] is False
    assert payload["synthetic_test_recipient_only"] is True
    assert payload["delivery_receipt_is_human_acknowledgement"] is False
    assert payload["clinical_action_automated"] is False
    assert payload["clinical_validation"] is False
    assert all(case["receipt_valid"] for case in payload["cases"])
