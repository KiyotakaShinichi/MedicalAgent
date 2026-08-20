from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services import dep001b_internal_blind_evaluation as blind_module
from backend.services.dep001b_candidate_freeze import freeze_dep001b_candidate, verify_frozen_candidate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "repo"
    directory = root / "Data/evals/safety/dep001b"
    directory.mkdir(parents=True)
    blind = directory / "internal_blind_safety_bank.jsonl"
    blind.write_text(json.dumps({"case_id": "blind-1"}) + "\n", encoding="utf-8")
    _write_json(directory / "latest_runtime_assurance.json", {
        "status": "eligible_to_freeze",
        "fault_injection": {"passed": True, "cases": []},
    })
    _write_json(directory / "latest_overlap_audit.json", {
        "status": "passed",
        "exact_overlap_counts": {
            "train_vs_burned_external": 0,
            "validation_vs_burned_external": 0,
            "internal_test_vs_burned_external": 0,
            "internal_blind_vs_burned_external": 0,
        },
        "exact_burned_external_overlap_count": 0,
        "used_for_tuning": False,
        "external_text_or_case_ids_emitted": False,
    })
    return root, blind, directory / "candidate_freeze_manifest.json", directory / "internal_blind_run_receipt.json"


def test_freeze_hashes_candidate_and_detects_mutation(tmp_path: Path) -> None:
    root, blind, manifest, receipt = _workspace(tmp_path)
    relative = "Data/evals/safety/dep001b/internal_blind_safety_bank.jsonl"
    frozen = freeze_dep001b_candidate(
        root=root, manifest_path=manifest, frozen_paths=(relative,), receipt_path=receipt
    )
    assert frozen["candidate_frozen"] is True
    assert verify_frozen_candidate(root=root, manifest_path=manifest)["passed"] is True
    blind.write_text("mutated", encoding="utf-8")
    assert verify_frozen_candidate(root=root, manifest_path=manifest)["passed"] is False


def test_one_shot_blind_run_cannot_be_repeated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, blind, manifest, receipt = _workspace(tmp_path)
    relative = "Data/evals/safety/dep001b/internal_blind_safety_bank.jsonl"
    freeze_dep001b_candidate(root=root, manifest_path=manifest, frozen_paths=(relative,), receipt_path=receipt)
    output = blind.parent / "result.json"
    raw = blind.parent / "raw.json"
    monkeypatch.setattr(blind_module, "evaluate_dep001b_rows", lambda *_a, **_k: {
        "metrics": {"unsafe_pass_rate": 0.0},
        "confidence_intervals": {},
        "targets_passed": True,
        "failed_case_ids": [],
        "cases": [],
    })
    result = blind_module.run_dep001b_internal_blind_once(
        root=root,
        blind_path=blind,
        manifest_path=manifest,
        receipt_path=receipt,
        output_path=output,
        raw_results_path=raw,
    )
    assert result["ready_for_new_external_holdout"] is True
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "completed"
    with pytest.raises(RuntimeError, match="one-shot"):
        blind_module.run_dep001b_internal_blind_once(
            root=root,
            blind_path=blind,
            manifest_path=manifest,
            receipt_path=receipt,
            output_path=output,
            raw_results_path=raw,
        )


def test_freeze_rejects_nonzero_overlap_in_current_schema(tmp_path: Path) -> None:
    root, _blind, manifest, receipt = _workspace(tmp_path)
    overlap = root / "Data/evals/safety/dep001b/latest_overlap_audit.json"
    payload = json.loads(overlap.read_text(encoding="utf-8"))
    payload["exact_overlap_counts"]["train_vs_burned_external"] = 1
    payload["exact_burned_external_overlap_count"] = 1
    _write_json(overlap, payload)
    with pytest.raises(RuntimeError, match="exact overlap"):
        freeze_dep001b_candidate(
            root=root,
            manifest_path=manifest,
            frozen_paths=("Data/evals/safety/dep001b/internal_blind_safety_bank.jsonl",),
            receipt_path=receipt,
        )
