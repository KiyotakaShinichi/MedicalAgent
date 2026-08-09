from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from backend.services.prototype_independent_prompt_bank_v2 import (
    SAFE_NEGATIVE_CASES,
    TOTAL_CASES,
    build_prompt_bank_v2_cases,
    freeze_prompt_bank_v2,
)
from backend.services.prototype_independent_prompt_eval_v2 import (
    evaluate_frozen_prompt_bank_v2,
)


def test_bank_is_exactly_3000_unique_and_balanced() -> None:
    rows = build_prompt_bank_v2_cases()
    assert len(rows) == TOTAL_CASES == 3000
    assert len({row["query"] for row in rows}) == TOTAL_CASES
    assert sum(row["safe_negative"] for row in rows) == SAFE_NEGATIVE_CASES
    unsafe_counts = {}
    for row in rows:
        if not row["safe_negative"]:
            unsafe_counts[row["category"]] = unsafe_counts.get(row["category"], 0) + 1
    assert set(unsafe_counts.values()) == {240}


def test_generator_has_no_project_service_imports_or_file_reads() -> None:
    source_path = Path("backend/services/prototype_independent_prompt_bank_v2.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith("backend.") for name in imported)
    assert not any(name.startswith("tests.") for name in imported)


def test_freeze_is_hash_locked_and_one_way(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    manifest = tmp_path / "manifest.json"
    result = freeze_prompt_bank_v2(bank, manifest)
    assert result["evaluated_once"] is False
    assert result["generator_independent_of_classifier_prototypes"] is True
    assert result["clinical_validation"] is False
    with pytest.raises(FileExistsError):
        freeze_prompt_bank_v2(bank, manifest)


def test_evaluator_rejects_hash_change(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    manifest = tmp_path / "manifest.json"
    freeze_prompt_bank_v2(bank, manifest)
    bank.write_text(bank.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_frozen_prompt_bank_v2(bank, manifest, tmp_path / "out.json", tmp_path / "fail.json")


def test_evaluator_marks_manifest_and_refuses_second_run(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    manifest = tmp_path / "manifest.json"
    freeze_prompt_bank_v2(bank, manifest)
    case_by_query = {
        row["query"]: row
        for row in (
            json.loads(line)
            for line in bank.read_text(encoding="utf-8").splitlines()
        )
    }

    def planner(query: str):
        case = case_by_query[query]
        return {"route": case["expected_route"], "allowed_tools": []}

    def classifier(query: str):
        case = case_by_query[query]
        return {"family": case["expected_family"]}

    result = evaluate_frozen_prompt_bank_v2(
        bank,
        manifest,
        tmp_path / "out.json",
        tmp_path / "fail.json",
        planner=planner,
        classifier=classifier,
    )
    assert result["total_n"] == 3000
    assert result["pass_rate"] == 1.0
    assert json.loads(manifest.read_text(encoding="utf-8"))["evaluated_once"] is True
    with pytest.raises(RuntimeError, match="one-pass"):
        evaluate_frozen_prompt_bank_v2(
            bank, manifest, tmp_path / "out2.json", tmp_path / "fail2.json"
        )
