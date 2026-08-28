"""Behavioral contracts for the R1 cohesion-only decompositions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from backend.api.main import app
from backend.services.unsafe_intent_semantic_classifier import FAMILIES


PATIENT_INTERACTION_OPERATIONS = {
    ("post", "/me/symptoms"),
    ("post", "/me/labs"),
    ("post", "/me/imaging-reports"),
    ("post", "/me/medications"),
    ("post", "/me/treatments"),
    ("post", "/me/family-history"),
    ("post", "/me/genetic-test-records"),
    ("post", "/me/biomarker-records"),
    ("post", "/me/tumor-marker-records"),
    ("post", "/patients/{patient_id}/genetic-counseling-review"),
    ("get", "/patients/{patient_id}/chat"),
    ("post", "/patients/{patient_id}/chat"),
    ("get", "/me/chat"),
    ("post", "/me/chat"),
    ("delete", "/me/record-write-actions/{audit_id}"),
    ("post", "/me/chat/stream"),
    ("post", "/patients/{patient_id}/chat/stream"),
    ("post", "/me/agent-feedback"),
    ("get", "/agent-feedback"),
    ("get", "/me/uploads"),
    ("get", "/me/uploads/{upload_id}/content"),
    ("post", "/me/uploads"),
}


def test_patient_interaction_route_contract_is_complete_and_singly_tagged() -> None:
    spec = app.openapi()
    observed = {
        (method, path)
        for path, operations in spec["paths"].items()
        for method, operation in operations.items()
        if isinstance(operation, dict)
        and operation.get("operationId")
        and (method, path) in PATIENT_INTERACTION_OPERATIONS
    }
    assert observed == PATIENT_INTERACTION_OPERATIONS
    for method, path in observed:
        assert spec["paths"][path][method]["tags"] == ["patient"]


def test_unsafe_family_policy_data_fingerprint_is_stable() -> None:
    payload = json.dumps(
        [asdict(family) for family in FAMILIES],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert len(FAMILIES) == 11
    assert hashlib.sha256(payload).hexdigest() == (
        "b6ec731187f34f049ab0eadcb706adbde73163cc749c541020044d80271c60d9"
    )
