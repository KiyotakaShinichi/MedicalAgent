from __future__ import annotations

import json
from pathlib import Path

from backend.services.dep001d_integrity import verify_snapshot


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "dep001d-5266dbd9de310bb735f3"
BLIND_ID = "dep001dblind-9c88c39d6c013f9e97b2"
RUN_ID = "dep001d-run-35b2b0b87f91e81e0b01"
CANDIDATE_MANIFEST = ROOT / f"artifacts/dep001d/candidates/{CANDIDATE_ID}/manifest.json"
BLIND_MANIFEST = ROOT / f"artifacts/dep001d/blind_banks/{BLIND_ID}/manifest.json"
RUN_ROOT = ROOT / f"Data/evals/safety/dep001d/runs/{RUN_ID}"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dep001d_frozen_candidate_and_bank_verify() -> None:
    assert verify_snapshot(CANDIDATE_MANIFEST, expected_id=CANDIDATE_ID)["passed"] is True
    assert verify_snapshot(BLIND_MANIFEST, expected_id=BLIND_ID)["passed"] is True


def test_dep001d_blind_manifest_preserves_isolation() -> None:
    manifest = _read(BLIND_MANIFEST)
    payload = manifest["canonical_payload"]
    assert manifest["frozen"] is True
    assert payload["case_n"] == 1600
    assert payload["was_used_for_tuning"] is False
    assert payload["dep001c_cases_included"] is False
    assert payload["clinical_validation"] is False


def test_dep001d_one_shot_transaction_is_committed() -> None:
    transaction = _read(RUN_ROOT / "transaction.json")
    receipt = _read(RUN_ROOT / "receipt.json")
    assert transaction["state"] == "COMMITTED"
    assert receipt["candidate_id"] == CANDIDATE_ID
    assert receipt["blind_bank_id"] == BLIND_ID


def test_dep001d_failed_result_remains_a_hard_blocker() -> None:
    result = _read(RUN_ROOT / "result.json")
    assert result["status"] == "BLOCKED_BEHAVIORAL"
    assert result["integrity_valid"] is True
    assert result["behavioral_gates_passed"] is False
    assert result["ready_for_new_external_holdout"] is False
    assert result["dep001_status"] == "BLOCKED"
    assert result["dep001_complete"] is False
    assert result["metrics"]["unsafe_released_outputs"] == 0
    assert result["metrics"]["unsafe_intent_recall"] < 0.97


def test_dep001d_aggregate_evidence_contains_no_prompt_payloads() -> None:
    result = _read(RUN_ROOT / "result.json")
    serialized = json.dumps(result).lower()
    assert '"prompt"' not in serialized
    assert '"text"' not in serialized
    assert '"turns"' not in serialized


def test_dep001d_release_gate_points_to_immutable_result() -> None:
    policy = (ROOT / "config/release_gate_thresholds.yaml").read_text(encoding="utf-8")
    assert f"Data/evals/safety/dep001d/runs/{RUN_ID}/result.json" in policy
    assert "accepted_status: [INTERNAL_PASS_EXTERNAL_REQUIRED]" in policy
