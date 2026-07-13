from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.n8n_webhook_dispatcher import (  # noqa: E402
    build_signed_dispatch,
    find_blocked_fields,
    verify_signed_dispatch,
)
from backend.services.oncology_canonical_schema import ROOT_DIR  # noqa: E402


OUTPUT = ROOT_DIR / "Data/evals/ops/latest_n8n_signed_dispatch_eval.json"


if __name__ == "__main__":
    secret = "eval-only-signing-secret"
    signed = build_signed_dispatch(
        workflow_id="release_gate_alert",
        payload={"run_id": "eval-001", "status": "passed", "artifact_count": 145, "failure_count": 0},
        secret=secret,
        timestamp="2026-01-01T00:00:00+00:00",
        event_id="n8n-signature-eval",
    )
    signature = signed["headers"]["X-NLCare-Signature"]
    valid = verify_signed_dispatch(body=signed["body"], signature=signature, secret=secret)
    tamper_rejected = not verify_signed_dispatch(
        body=signed["body"] + " ",
        signature=signature,
        secret=secret,
    )
    blocked_catch = bool(find_blocked_fields({"nested": {"raw_patient_message": "blocked"}}))
    report = {
        "schema_version": "n8n_signed_dispatch_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if valid and tamper_rejected and blocked_catch else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "phi_allowed": False,
        "network_request_sent": False,
        "signature_valid": valid,
        "tampered_body_rejected": tamper_rejected,
        "nested_blocked_field_caught": blocked_catch,
        "claim_boundary": (
            "This eval verifies an engineering webhook-signing and redaction contract only. It is not a security "
            "certification, clinical validation, or permission to send patient data."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
