"""Local signed-channel and receipt drill for redacted automation events."""

from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any

from backend.services.n8n_webhook_dispatcher import (
    build_signed_receipt,
    dispatch_signed_webhook,
    validate_signed_dispatch_envelope,
    validate_signed_receipt,
)


DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_automation_channel_drill.json")
DEFAULT_ATTEMPTS = 30
CLAIM_BOUNDARY = (
    "This drill uses a localhost HTTP receiver and synthetic test-recipient "
    "metadata only. A valid channel receipt proves local protocol delivery, "
    "not clinician acknowledgement, emergency coverage, patient contact, "
    "clinical action, external-channel reliability, or healthcare production readiness."
)


def build_automation_channel_drill(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
) -> dict[str, Any]:
    if attempts < 10:
        raise ValueError("At least 10 loopback attempts are required")
    secret = secrets.token_urlsafe(32)
    receiver = _LoopbackReceiver(secret=secret)
    receiver.start()
    latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    try:
        for index in range(attempts):
            event_id = f"loopback-alert-{index + 1:03d}"
            started = perf_counter()
            result = dispatch_signed_webhook(
                workflow_id="high_risk_review_alert",
                event_id=event_id,
                payload={
                    "alert_id": index + 1,
                    "event_type": "high_priority_review_item",
                    "priority": "urgent_review",
                    "review_path": (
                        f"/clinician/high-risk-conversation-alerts/{index + 1}"
                    ),
                    "delivery_scope": "redacted_internal_review_notification",
                    "recipient_scope": "synthetic_test_recipient_only",
                },
                env={
                    "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
                    "N8N_WEBHOOK_BASE_URL": receiver.base_url,
                    "N8N_WEBHOOK_SIGNING_SECRET": secret,
                    "NLCARE_ALERT_TEST_RECIPIENT_ONLY": "true",
                },
                timeout_seconds=3.0,
            )
            elapsed_ms = (perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)
            received = receiver.receipts.get(event_id) or {}
            receipt_validation = validate_signed_receipt(
                body=str(received.get("body") or ""),
                signature=str(received.get("signature") or ""),
                secret=secret,
            )
            rows.append(
                {
                    "event_id": event_id,
                    "dispatch_sent": result.get("sent") is True,
                    "http_status": (result.get("response") or {}).get("status_code"),
                    "dispatch_envelope_valid": received.get("dispatch_valid") is True,
                    "receipt_valid": receipt_validation.get("valid") is True,
                    "latency_ms": round(elapsed_ms, 3),
                }
            )
    finally:
        receiver.stop()

    passed = sum(
        bool(
            row["dispatch_sent"]
            and row["http_status"] == 202
            and row["dispatch_envelope_valid"]
            and row["receipt_valid"]
        )
        for row in rows
    )
    payload = {
        "schema_version": "automation_channel_drill_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if passed == attempts else "needs_attention",
        "attempt_count": attempts,
        "passed_count": passed,
        "failed_count": attempts - passed,
        "pass_rate": round(passed / attempts, 6),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "transport": "localhost_http_loopback",
        "local_network_delivery_performed": True,
        "external_delivery_performed": False,
        "live_n8n_delivery_completed": False,
        "synthetic_test_recipient_only": True,
        "payload_redacted": True,
        "phi_allowed": False,
        "delivery_receipt_is_human_acknowledgement": False,
        "clinician_acknowledgement_proven": False,
        "emergency_coverage_proven": False,
        "clinical_action_automated": False,
        "healthcare_production_ready": False,
        "clinical_validation": False,
        "cases": rows,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class _LoopbackReceiver:
    def __init__(self, *, secret: str) -> None:
        self.secret = secret
        self.receipts: dict[str, dict[str, Any]] = {}
        self._seen: set[str] = set()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Loopback receiver is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/webhook/nlcare"

    def start(self) -> None:
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                signature = str(self.headers.get("X-NLCare-Signature") or "")
                validation = validate_signed_dispatch_envelope(
                    body=body,
                    signature=signature,
                    secret=receiver.secret,
                    seen_event_ids=receiver._seen,
                )
                if not validation.get("valid"):
                    self.send_response(401)
                    self.end_headers()
                    return
                event_id = str(validation["event_id"])
                receipt = build_signed_receipt(
                    event_id=event_id,
                    receipt_id=f"loopback-receipt-{event_id}",
                    delivery_status="delivered",
                    secret=receiver.secret,
                )
                receiver.receipts[event_id] = {
                    "dispatch_valid": True,
                    "body": receipt["body"],
                    "signature": receipt["headers"][
                        "X-NLCare-Receipt-Signature"
                    ],
                }
                self.send_response(202)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _percentile(values: list[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * fraction,
        3,
    )


__all__ = ["build_automation_channel_drill"]
