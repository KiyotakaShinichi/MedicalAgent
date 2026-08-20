from __future__ import annotations

import json
import hashlib
from pathlib import Path

from backend.services import dep001b_overlap_audit as overlap_module


def test_overlap_audit_emits_aggregate_only_and_never_external_text(
    tmp_path: Path, monkeypatch,
) -> None:
    external = tmp_path / "external.json"
    external.write_text(json.dumps({"cases": [{"case_id": "SECRET-ID", "text": "SECRET HOLDOUT TEXT"}]}), encoding="utf-8")
    new_bank = tmp_path / "new_bank.jsonl"
    new_bank.write_text(json.dumps({"case_id": "new-1", "text": "general portal education"}) + "\n", encoding="utf-8")
    prior_bank = tmp_path / "prior_bank.jsonl"
    prior_bank.write_text(json.dumps({"case_id": "prior-1", "text": "unrelated prior control"}) + "\n", encoding="utf-8")
    output = tmp_path / "latest_overlap_audit.json"
    repository_output = overlap_module.OUTPUT_PATH
    repository_hash_before = (
        hashlib.sha256(repository_output.read_bytes()).hexdigest()
        if repository_output.is_file()
        else None
    )
    monkeypatch.setattr(overlap_module, "OUTPUT_PATH", output)
    monkeypatch.setattr(overlap_module, "NEW_PATHS", {"train": new_bank})
    monkeypatch.setattr(overlap_module, "PREVIOUS_PATHS", {"prior": prior_bank})
    payload = overlap_module.run_overlap_audit(external)
    encoded = json.dumps(payload)
    assert output.is_file()
    assert "SECRET-ID" not in encoded
    assert "SECRET HOLDOUT TEXT" not in encoded
    assert payload["external_text_or_case_ids_emitted"] is False
    assert payload["used_for_tuning"] is False
    assert payload["clinical_validation"] is False
    repository_hash_after = (
        hashlib.sha256(repository_output.read_bytes()).hexdigest()
        if repository_output.is_file()
        else None
    )
    assert repository_hash_after == repository_hash_before


def test_current_overlap_artifact_has_no_exact_burned_external_overlap() -> None:
    path = Path(__file__).resolve().parents[1] / "Data/evals/safety/dep001b/latest_overlap_audit.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exact_burned_external_overlap_count"] == 0
    assert payload["external_text_or_case_ids_emitted"] is False
