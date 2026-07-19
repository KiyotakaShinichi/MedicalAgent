import json

import pytest

from backend.services.adversarial_holdout_v5 import build_holdout_v5_cases, evaluate_holdout_v5, freeze_holdout_v5


def test_v5_has_broad_coverage_and_safe_controls():
    rows = build_holdout_v5_cases()
    assert len(rows) >= 120
    assert len({row["category"] for row in rows if not row["safe_negative"]}) == 11
    assert sum(row["safe_negative"] for row in rows) >= 20
    assert all(row["was_used_for_tuning"] is False for row in rows)


def test_freeze_is_immutable_and_hash_checked(tmp_path):
    bank = tmp_path / "v5.jsonl"
    manifest = tmp_path / "manifest.json"
    frozen = freeze_holdout_v5(bank, manifest)
    assert frozen["clinical_validation"] is False
    with pytest.raises(FileExistsError):
        freeze_holdout_v5(bank, manifest)
    bank.write_text(bank.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_holdout_v5(bank, manifest, tmp_path / "result.json")


def test_manifest_is_explicitly_internal(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    freeze_holdout_v5(tmp_path / "v5.jsonl", manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["internal_vs_external"] == "internal_frozen_holdout"
    assert "not external validation" in payload["claim_boundary"].lower()
