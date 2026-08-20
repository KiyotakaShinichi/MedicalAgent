from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.dep001c_integrity import IntegrityViolation, atomic_write_json, canonical_hash, sha256_file
from backend.services.dep001c_official_evaluation import run_dep001c_official_internal_once


PASSING_METRICS = {
    "unsafe_released_outputs": 0,
    "unsafe_intent_recall": 1.0,
    "urgent_escalation_recall": 1.0,
    "safe_educational_acceptance_rate": 1.0,
    "over_refusal_rate": 0.0,
    "english_unsafe_recall": 1.0,
    "taglish_unsafe_recall": 1.0,
    "filipino_heavy_unsafe_recall": 1.0,
    "en_taglish_gap": 0.0,
    "multi_turn_unsafe_recall": 1.0,
    "rag_conditioned_unsafe_recall": 1.0,
}


def _write_manifest(directory: Path, prefix: str, extra: dict, files: dict[str, bytes]) -> tuple[str, Path]:
    for relative, content in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    artifacts = {
        relative: {"sha256": sha256_file(directory / relative), "bytes": (directory / relative).stat().st_size}
        for relative in sorted(files)
    }
    canonical = {"artifacts": artifacts, **extra}
    manifest_hash = canonical_hash(canonical)
    snapshot_id = f"{prefix}{manifest_hash[:20]}"
    manifest = directory / "manifest.json"
    payload = {
        "snapshot_id": snapshot_id,
        "canonical_manifest_sha256": manifest_hash,
        "canonical_payload": canonical,
    }
    if prefix == "dep001c-":
        payload.update({"candidate_id": snapshot_id, "frozen_artifact_count": len(artifacts)})
    else:
        payload.update({"blind_bank_id": snapshot_id, "blind_bank_sha256": artifacts["bank/internal_blind_safety_bank.jsonl"]["sha256"]})
    atomic_write_json(manifest, payload)
    return snapshot_id, manifest


def _workspace(tmp_path: Path, mode: str = "success") -> tuple[str, str, Path, Path, Path]:
    candidates = tmp_path / "candidates"
    blinds = tmp_path / "blinds"
    runs = tmp_path / "runs"
    candidate_staging = tmp_path / "candidate-staging"
    blind_staging = tmp_path / "blind-staging"
    mutation = ""
    exit_code = "0"
    checkpoint_pass = "True"
    if mode == "post_mutation":
        mutation = "\nimport os, stat\np=Path(args.candidate_manifest).parent/'runtime/artifact.bin'\np.chmod(stat.S_IREAD|stat.S_IWRITE)\np.write_bytes(b'mutated')\n"
    if mode == "mid_failure":
        checkpoint_pass = "False"
        exit_code = "2"
    worker = f'''import argparse, json\nfrom pathlib import Path\np=argparse.ArgumentParser()\np.add_argument("--candidate-manifest")\np.add_argument("--blind-manifest")\np.add_argument("--output")\np.add_argument("--progress")\np.add_argument("--checkpoint-interval")\nargs=p.parse_args()\nPath(args.progress).write_text(json.dumps({{"checkpoints":[{{"candidate_passed":{checkpoint_pass},"blind_passed":True}}]}}))\nPath(args.output).write_text(json.dumps({{"case_n":10,"metrics":{json.dumps(PASSING_METRICS)},"confidence_intervals":{{}},"failed_case_ids":[]}})){mutation}\nraise SystemExit({exit_code})\n'''
    assurance = json.dumps({"fault_injection": {"passed": True, "passed_n": 1, "total_n": 1}}).encode()
    integrity = json.dumps({"status": "passed"}).encode()
    candidate_files = {
        "source/dep001c_snapshot_worker.py": worker.encode(),
        "runtime/artifact.bin": b"frozen",
        "assurance/runtime_assurance.json": assurance,
        "assurance/integrity_fault_injection.json": integrity,
    }
    temp_id, _ = _write_manifest(candidate_staging, "dep001c-", {}, candidate_files)
    candidate_dir = candidates / temp_id
    candidate_dir.parent.mkdir(parents=True)
    candidate_staging.rename(candidate_dir)
    bank_files = {"bank/internal_blind_safety_bank.jsonl": b"{}\n"}
    blind_id, _ = _write_manifest(blind_staging, "dep001cblind-", {"bank_path": "bank/internal_blind_safety_bank.jsonl"}, bank_files)
    blind_dir = blinds / blind_id
    blind_dir.parent.mkdir(parents=True)
    blind_staging.rename(blind_dir)
    return temp_id, blind_id, candidates, blinds, runs


def _run(tmp_path: Path, mode: str = "success") -> dict:
    candidate_id, blind_id, candidates, blinds, runs = _workspace(tmp_path, mode)
    return run_dep001c_official_internal_once(
        candidate_id=candidate_id,
        blind_bank_id=blind_id,
        candidate_root=candidates,
        blind_root=blinds,
        run_root=runs,
        lock_root=tmp_path / "locks",
        process_rows=[],
        timeout_seconds=30,
    )


def test_successful_immutable_evaluation_lifecycle(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result["status"] == "INTERNAL_PASS_EXTERNAL_REQUIRED"
    assert result["ready_for_new_external_holdout"] is True
    transaction = json.loads(next((tmp_path / "runs").rglob("transaction.json")).read_text())
    assert [row["state"] for row in transaction["history"]] == [
        "PREPARED", "LOCKED", "VERIFIED_PRE", "RUNNING", "VERIFIED_POST", "COMMITTED"
    ]


def test_pre_run_integrity_failure_invalidates(tmp_path: Path) -> None:
    candidate_id, blind_id, candidates, blinds, runs = _workspace(tmp_path)
    (candidates / candidate_id / "runtime/artifact.bin").write_bytes(b"changed-before-run")
    with pytest.raises(IntegrityViolation, match="pre_run"):
        run_dep001c_official_internal_once(
            candidate_id=candidate_id,
            blind_bank_id=blind_id,
            candidate_root=candidates,
            blind_root=blinds,
            run_root=runs,
            lock_root=tmp_path / "locks",
            process_rows=[],
            timeout_seconds=30,
        )
    invalid = json.loads(next(runs.rglob("invalidated.json")).read_text())
    assert invalid["behavioral_metrics_admissible"] is False


def test_mid_run_integrity_failure_invalidates(tmp_path: Path) -> None:
    with pytest.raises(IntegrityViolation, match="frozen_worker_failed"):
        _run(tmp_path, "mid_failure")
    transaction = json.loads(next((tmp_path / "runs").rglob("transaction.json")).read_text())
    assert transaction["state"] == "INVALIDATED"


def test_post_run_mutation_invalidates(tmp_path: Path) -> None:
    with pytest.raises(IntegrityViolation, match="post_run"):
        _run(tmp_path, "post_mutation")
    transaction = json.loads(next((tmp_path / "runs").rglob("transaction.json")).read_text())
    assert transaction["state"] == "INVALIDATED"


def test_official_bank_cannot_be_evaluated_twice(tmp_path: Path) -> None:
    candidate_id, blind_id, candidates, blinds, runs = _workspace(tmp_path)
    kwargs = dict(
        candidate_id=candidate_id,
        blind_bank_id=blind_id,
        candidate_root=candidates,
        blind_root=blinds,
        run_root=runs,
        lock_root=tmp_path / "locks",
        process_rows=[],
        timeout_seconds=30,
    )
    assert run_dep001c_official_internal_once(**kwargs)["status"] == "INTERNAL_PASS_EXTERNAL_REQUIRED"
    with pytest.raises(IntegrityViolation, match="one_shot"):
        run_dep001c_official_internal_once(**kwargs)

