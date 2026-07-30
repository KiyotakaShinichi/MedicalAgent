from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.services.adversarial_holdout_v7 import (
    build_holdout_v7_cases,
    evaluate_holdout_v7,
    freeze_holdout_v7,
)
from backend.services.unsafe_intent_mutation_dev_eval import (
    COMPOSITIONAL_MUTATIONS,
    DEV_CASES,
    SAFE_BOUNDARY_CONTROLS,
    SAFE_CASES,
)


def test_v7_builder_has_distinct_coverage_and_honest_metadata():
    rows = build_holdout_v7_cases()
    assert len(rows) == 142
    assert sum(row["safe_negative"] for row in rows) == 32
    assert len({row["category"] for row in rows if not row["safe_negative"]}) == 11
    assert all(row["was_used_for_tuning"] is False for row in rows)
    assert all("author_contaminated" in row["case_source"] for row in rows)
    development_queries = {
        query
        for _, query, _ in DEV_CASES + COMPOSITIONAL_MUTATIONS
    } | set(SAFE_CASES + SAFE_BOUNDARY_CONTROLS)
    assert not ({row["query"] for row in rows} & development_queries)


def test_v7_freeze_is_hash_locked_and_non_overwriting(tmp_path: Path):
    bank = tmp_path / "bank.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_holdout_v7(bank, manifest_path)
    content = bank.read_text(encoding="utf-8")
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == manifest["sha256"]
    assert manifest["clinical_validation"] is False
    assert manifest["evaluated_once"] is False
    with pytest.raises(FileExistsError):
        freeze_holdout_v7(bank, manifest_path)


def test_v7_hash_mismatch_fails_before_evaluation(tmp_path: Path):
    bank = tmp_path / "bank.jsonl"
    manifest_path = tmp_path / "manifest.json"
    freeze_holdout_v7(bank, manifest_path)
    bank.write_text(bank.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_holdout_v7(bank, manifest_path, tmp_path / "result.json")


def test_v7_default_contract_is_one_pass(tmp_path: Path):
    bank = tmp_path / "bank.jsonl"
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "result.json"
    freeze_holdout_v7(bank, manifest_path)
    result = evaluate_holdout_v7(bank, manifest_path, result_path)
    assert result["clinical_validation"] is False
    assert result["was_used_for_tuning"] is False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evaluated_once"] is True
    with pytest.raises(RuntimeError, match="one-pass"):
        evaluate_holdout_v7(bank, manifest_path, result_path)
