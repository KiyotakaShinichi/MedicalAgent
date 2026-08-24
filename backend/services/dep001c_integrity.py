"""Immutable snapshots and transactional integrity controls for DEP-001C.

This module is deliberately behavior-agnostic. It does not train, calibrate,
score, or tune the safety model. Its only authority is evidence integrity.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/dep001c"
CANDIDATE_ROOT = ARTIFACT_ROOT / "candidates"
BLIND_ROOT = ARTIFACT_ROOT / "blind_banks"
LOCK_ROOT = ARTIFACT_ROOT / "locks"
RUN_ROOT = ROOT / "Data/evals/safety/dep001c/runs"

TRANSACTION_STATES = (
    "PREPARED",
    "LOCKED",
    "VERIFIED_PRE",
    "RUNNING",
    "VERIFIED_POST",
    "COMMITTED",
    "INVALIDATED",
)

MUTABLE_ALIAS_PATTERN = re.compile(r"^(latest|current)(?:[._-]|$)", re.IGNORECASE)
WRITER_MARKERS = (
    "pytest",
    "train_dep001",
    "build_dep001",
    "run_dep001b_overlap",
    "run_dep001b_runtime",
    "freeze_dep001",
    "run_dep001b_internal_blind",
    "run_dep001c_official",
    "prepare_dep001c",
    "build_dep001c_blind",
)
SAFETY_MARKERS = ("dep001b", "dep001c", "safety_policy_action")


class IntegrityViolation(RuntimeError):
    """Raised whenever official evidence can no longer be trusted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assert_immutable_identifier_path(path: Path) -> None:
    for part in path.parts:
        if MUTABLE_ALIAS_PATTERN.match(part):
            raise IntegrityViolation(f"mutable_alias_rejected:{part}")


def make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    if root.exists():
        try:
            root.chmod(stat.S_IREAD | stat.S_IEXEC)
        except OSError:
            pass


def make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    try:
        root.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    except OSError:
        pass
    for path in root.rglob("*"):
        try:
            path.chmod(stat.S_IREAD | stat.S_IWRITE | (stat.S_IEXEC if path.is_dir() else 0))
        except OSError:
            pass


def verify_snapshot(manifest_path: Path, *, expected_id: str | None = None) -> dict[str, Any]:
    assert_immutable_identifier_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = manifest.get("canonical_payload")
    if not isinstance(canonical, dict):
        raise IntegrityViolation("manifest_missing_canonical_payload")
    observed_manifest_hash = canonical_hash(canonical)
    expected_manifest_hash = str(manifest.get("canonical_manifest_sha256") or "")
    snapshot_id = str(manifest.get("snapshot_id") or "")
    mismatches: list[str] = []
    if observed_manifest_hash != expected_manifest_hash:
        mismatches.append("manifest:canonical_hash")
    if expected_id is not None and snapshot_id != expected_id:
        mismatches.append("manifest:snapshot_id")
    if not snapshot_id.endswith(expected_manifest_hash[:20]):
        mismatches.append("manifest:content_address")
    root = manifest_path.parent
    for relative, record in dict(canonical.get("artifacts") or {}).items():
        path = root / relative
        if not path.is_file():
            mismatches.append(f"missing:{relative}")
            continue
        if sha256_file(path) != str(record.get("sha256") or ""):
            mismatches.append(f"hash:{relative}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            mismatches.append(f"size:{relative}")
    return {
        "passed": not mismatches,
        "snapshot_id": snapshot_id,
        "canonical_manifest_sha256": observed_manifest_hash,
        "artifact_count": len(dict(canonical.get("artifacts") or {})),
        "mismatches": mismatches,
        "verified_at": utc_now(),
    }


def process_records() -> list[dict[str, Any]]:
    if os.name != "nt":
        return _process_records_procfs()
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise IntegrityViolation("process_inventory_unavailable")
    raw = json.loads(completed.stdout)
    rows = raw if isinstance(raw, list) else [raw]
    return [
        {
            "pid": int(row.get("ProcessId") or 0),
            "parent_pid": int(row.get("ParentProcessId") or 0),
            "name": str(row.get("Name") or ""),
            "command_line": str(row.get("CommandLine") or ""),
        }
        for row in rows
    ]


def detect_conflicting_writers(
    records: Iterable[Mapping[str, Any]] | None = None,
    *,
    current_pid: int | None = None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in (records if records is not None else process_records())]
    own_pid = int(current_pid or os.getpid())
    by_pid = {int(row.get("pid") or 0): row for row in rows}
    excluded = {own_pid}
    cursor = own_pid
    while cursor in by_pid:
        parent = int(by_pid[cursor].get("parent_pid") or 0)
        if not parent or parent in excluded:
            break
        excluded.add(parent)
        cursor = parent
    conflicts = []
    for row in rows:
        pid = int(row.get("pid") or 0)
        if not pid or pid in excluded:
            continue
        haystack = f"{row.get('name', '')} {row.get('command_line', '')}".lower()
        writer = any(marker in haystack for marker in WRITER_MARKERS)
        safety_related = any(marker in haystack for marker in SAFETY_MARKERS)
        if writer and safety_related:
            conflicts.append({
                "pid": pid,
                "parent_pid": int(row.get("parent_pid") or 0),
                "name": str(row.get("name") or ""),
                "command_sha256": hashlib.sha256(haystack.encode("utf-8")).hexdigest(),
            })
    return conflicts


def _process_records_procfs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise IntegrityViolation("process_inventory_unavailable")
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            status = (directory / "status").read_text(encoding="utf-8", errors="ignore")
            parent_match = re.search(r"^PPid:\s+(\d+)", status, flags=re.MULTILINE)
            command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="ignore")
            rows.append({
                "pid": int(directory.name),
                "parent_pid": int(parent_match.group(1)) if parent_match else 0,
                "name": directory.name,
                "command_line": command,
            })
        except (OSError, ValueError):
            continue
    return rows


def _windows_pid_is_alive(pid: int) -> bool:
    """Ask the kernel directly whether a PID is running.

    `ctypes.wintypes` does not import off Windows, so it is loaded here rather
    than at module scope.
    """
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    ERROR_INVALID_PARAMETER = 87
    STILL_ACTIVE = 259

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Without explicit restypes ctypes truncates a 64-bit HANDLE to int.
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # "No such PID" is the only definite no. Access denied means the
        # process is running under an account we may not open - still running.
        return ctypes.get_last_error() != ERROR_INVALID_PARAMETER
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def pid_is_alive(pid: int) -> bool:
    """Is a process with this PID running right now?

    This is asked of the PID written into an existing evaluation lock, to tell
    a lock held by a live run (refuse to proceed) from one left behind by a
    crashed run (recover it as stale). The permissive answer is the dangerous
    one: reporting a live evaluation as dead lets a second run steal its lock
    and overwrite evidence mid-flight. So every uncertain case here resolves to
    "alive", and only a positive "no such process" counts as dead.

    Windows asks the kernel through OpenProcess instead of shelling out to
    PowerShell. The subprocess it replaces cost ~1.8s per probe on an idle
    machine and, under the load of a full test suite, consistently blew its own
    10s timeout - raising TimeoutExpired out of lock acquisition rather than
    returning an answer at all.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists; it is simply owned by another user.
        return True
    except OSError:
        return True
    return True


@dataclass
class EvaluationLock(AbstractContextManager["EvaluationLock"]):
    candidate_id: str
    run_id: str
    lock_root: Path = LOCK_ROOT
    path: Path | None = None
    recovered_stale_lock: bool = False

    def __enter__(self) -> "EvaluationLock":
        self.lock_root.mkdir(parents=True, exist_ok=True)
        self.path = self.lock_root / f"{self.candidate_id}.lock.json"
        assert_immutable_identifier_path(self.path)
        if self.path.exists():
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            owner_pid = int(existing.get("pid") or 0)
            if pid_is_alive(owner_pid):
                raise IntegrityViolation(f"concurrent_evaluation_rejected:{owner_pid}")
            stale = self.path.with_name(f"{self.path.name}.stale-{canonical_hash(existing)[:12]}")
            os.replace(self.path, stale)
            self.recovered_stale_lock = True
        payload = {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "acquired_at": utc_now(),
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        descriptor = os.open(self.path, flags)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.path and self.path.exists():
            self.path.unlink()
        return None


def transition_transaction(
    path: Path,
    *,
    state: str,
    transaction: Mapping[str, Any] | None = None,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in TRANSACTION_STATES:
        raise ValueError(f"invalid_transaction_state:{state}")
    payload = dict(transaction or {})
    history = list(payload.get("history") or [])
    if history and history[-1].get("state") == "INVALIDATED" and state != "INVALIDATED":
        raise IntegrityViolation("invalidated_transaction_cannot_be_promoted")
    history.append({"state": state, "at": utc_now(), "detail": dict(detail or {})})
    payload.update({"state": state, "history": history, "updated_at": utc_now()})
    atomic_write_json(path, payload)
    return payload


__all__ = [
    "ARTIFACT_ROOT",
    "BLIND_ROOT",
    "CANDIDATE_ROOT",
    "EvaluationLock",
    "IntegrityViolation",
    "LOCK_ROOT",
    "RUN_ROOT",
    "TRANSACTION_STATES",
    "assert_immutable_identifier_path",
    "atomic_write_json",
    "canonical_hash",
    "detect_conflicting_writers",
    "make_tree_read_only",
    "make_tree_writable",
    "process_records",
    "sha256_file",
    "transition_transaction",
    "utc_now",
    "verify_snapshot",
]
