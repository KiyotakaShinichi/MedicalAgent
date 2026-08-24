from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend.services.dep001c_integrity import (
    EvaluationLock,
    IntegrityViolation,
    assert_immutable_identifier_path,
    atomic_write_json,
    canonical_hash,
    detect_conflicting_writers,
    make_tree_read_only,
    pid_is_alive,
    sha256_file,
    transition_transaction,
    verify_snapshot,
)
from backend.services.dep001c_integrity_assurance import run_integrity_fault_injection


def _snapshot(tmp_path: Path) -> tuple[Path, str, Path]:
    root = tmp_path / "dep001c-snapshot"
    artifact = root / "runtime/model.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model-v1")
    canonical = {
        "artifacts": {
            "runtime/model.bin": {
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
        }
    }
    manifest_hash = canonical_hash(canonical)
    snapshot_id = f"dep001c-{manifest_hash[:20]}"
    manifest = root / "manifest.json"
    atomic_write_json(manifest, {
        "snapshot_id": snapshot_id,
        "canonical_manifest_sha256": manifest_hash,
        "canonical_payload": canonical,
    })
    return manifest, snapshot_id, artifact


def test_manifest_hashing_and_post_freeze_mutation_detection(tmp_path: Path) -> None:
    manifest, snapshot_id, artifact = _snapshot(tmp_path)
    assert verify_snapshot(manifest, expected_id=snapshot_id)["passed"] is True
    artifact.write_bytes(b"model-v2")
    verification = verify_snapshot(manifest, expected_id=snapshot_id)
    assert verification["passed"] is False
    assert verification["mismatches"] == ["hash:runtime/model.bin"]


def test_snapshot_is_marked_read_only(tmp_path: Path) -> None:
    manifest, _snapshot_id, artifact = _snapshot(tmp_path)
    make_tree_read_only(manifest.parent)
    assert artifact.stat().st_mode & stat.S_IWRITE == 0


def test_mutable_latest_alias_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(IntegrityViolation, match="mutable_alias"):
        assert_immutable_identifier_path(tmp_path / "latest_candidate" / "manifest.json")


def test_writer_process_detection_is_command_aggregate_only() -> None:
    records = [
        {"pid": 10, "parent_pid": 0, "name": "python.exe", "command_line": "pytest tests/test_dep001b_overlap_audit.py"},
        {"pid": 11, "parent_pid": 0, "name": "python.exe", "command_line": "python unrelated.py"},
    ]
    conflicts = detect_conflicting_writers(records, current_pid=999)
    assert [row["pid"] for row in conflicts] == [10]
    assert "command_line" not in conflicts[0]
    assert len(conflicts[0]["command_sha256"]) == 64


def test_concurrent_evaluation_is_rejected(tmp_path: Path) -> None:
    with EvaluationLock("dep001c-" + "a" * 20, "run-a", lock_root=tmp_path):
        with pytest.raises(IntegrityViolation, match="concurrent"):
            with EvaluationLock("dep001c-" + "a" * 20, "run-b", lock_root=tmp_path):
                pass


def test_stale_lock_is_recovered_without_killing_process(tmp_path: Path) -> None:
    candidate_id = "dep001c-" + "b" * 20
    lock = tmp_path / f"{candidate_id}.lock.json"
    atomic_write_json(lock, {"pid": 99999999, "run_id": "old"})
    with EvaluationLock(candidate_id, "new", lock_root=tmp_path) as acquired:
        assert acquired.recovered_stale_lock is True
        assert lock.exists()
    assert not lock.exists()
    assert list(tmp_path.glob("*.stale-*"))


# --- the liveness probe behind lock recovery ------------------------------
#
# Whether a lock is "held" or "abandoned" is decided entirely by this probe, so
# both directions of a wrong answer are integrity failures: a false "dead" lets
# a second run steal the lock from a live evaluation, and a probe that raises
# instead of answering takes lock acquisition down with it.


def test_probe_reports_a_running_process_as_alive() -> None:
    assert pid_is_alive(os.getpid()) is True


def test_probe_reports_an_exited_process_as_dead() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert pid_is_alive(child.pid) is False


@pytest.mark.parametrize("pid", [0, -1, -99999])
def test_probe_rejects_non_positive_pids(pid: int) -> None:
    assert pid_is_alive(pid) is False


def test_probe_answers_without_blocking_lock_acquisition() -> None:
    """A regression guard on cost, because cost was the actual failure.

    The Windows implementation used to shell out to PowerShell: ~1.8s per probe
    idle, and past its own 10s timeout under the load of a full test suite, so
    it raised TimeoutExpired out of `EvaluationLock.__enter__` instead of
    returning True or False. A probe on the lock path has to be cheap enough
    that load cannot turn it into an exception.
    """
    pid_is_alive(os.getpid())  # discount one-off library loading

    started = time.perf_counter()
    for _ in range(20):
        pid_is_alive(os.getpid())
        pid_is_alive(99999999)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, f"40 probes took {elapsed:.2f}s; this belongs on a lock path"


def test_probe_treats_an_unopenable_process_as_alive(monkeypatch) -> None:
    """Fail closed: a process we may not query is still a process.

    A PID owned by another user answers "permission denied", which is evidence
    that it exists. Reading that as "dead" would hand its lock to a second run.
    """
    monkeypatch.setattr(os, "name", "posix")

    def denied(_pid: int, _signal: int) -> None:
        raise PermissionError("not yours")

    monkeypatch.setattr(os, "kill", denied)
    assert pid_is_alive(4242) is True


def test_probe_reports_no_such_process_as_dead(monkeypatch) -> None:
    monkeypatch.setattr(os, "name", "posix")

    def missing(_pid: int, _signal: int) -> None:
        raise ProcessLookupError("gone")

    monkeypatch.setattr(os, "kill", missing)
    assert pid_is_alive(4242) is False


def test_a_live_owner_keeps_its_lock_even_under_probe_pressure(tmp_path: Path) -> None:
    """The end-to-end property: a lock held by this (live) process is refused.

    Distinct from `test_concurrent_evaluation_is_rejected` in that it asserts
    the refusal survives repeated probing rather than a single acquisition.
    """
    candidate_id = "dep001c-" + "c" * 20
    with EvaluationLock(candidate_id, "run-a", lock_root=tmp_path):
        for _ in range(5):
            with pytest.raises(IntegrityViolation, match="concurrent"):
                with EvaluationLock(candidate_id, "run-b", lock_root=tmp_path):
                    pass


def test_invalidated_transaction_cannot_be_promoted(tmp_path: Path) -> None:
    path = tmp_path / "transaction.json"
    transaction = transition_transaction(path, state="PREPARED")
    transaction = transition_transaction(path, state="INVALIDATED", transaction=transaction)
    with pytest.raises(IntegrityViolation, match="cannot_be_promoted"):
        transition_transaction(path, state="COMMITTED", transaction=transaction)


def test_integrity_fault_injection_detects_write_and_blocks_promotion(tmp_path: Path) -> None:
    payload = run_integrity_fault_injection(tmp_path / "fault.json")
    assert payload["status"] == "passed"
    assert payload["mid_run_mutation_detected"] is True
    assert payload["invalid_run_promotion_rejected"] is True
    assert payload["production_candidate_modified"] is False

