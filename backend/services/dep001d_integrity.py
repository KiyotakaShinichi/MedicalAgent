"""DEP-001D immutable evidence paths built on the DEP-001C integrity primitives."""
from __future__ import annotations

from pathlib import Path

# Explicit re-export rather than `import *`. The starred form hid which
# primitives this facade actually forwards and left every name in `__all__`
# unverifiable by static analysis; these are exactly the names `__all__`
# already declared.
from backend.services.dep001c_integrity import (
    EvaluationLock,
    IntegrityViolation,
    atomic_write_json,
    canonical_hash,
    detect_conflicting_writers,
    make_tree_read_only,
    sha256_file,
    transition_transaction,
    utc_now,
    verify_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/dep001d"
CANDIDATE_ROOT = ARTIFACT_ROOT / "candidates"
BLIND_ROOT = ARTIFACT_ROOT / "blind_banks"
LOCK_ROOT = ARTIFACT_ROOT / "locks"
RUN_ROOT = ROOT / "Data/evals/safety/dep001d/runs"


__all__ = [
    "ARTIFACT_ROOT", "BLIND_ROOT", "CANDIDATE_ROOT", "LOCK_ROOT", "ROOT",
    "RUN_ROOT", "EvaluationLock", "IntegrityViolation", "atomic_write_json",
    "canonical_hash", "detect_conflicting_writers", "make_tree_read_only",
    "sha256_file", "transition_transaction", "utc_now", "verify_snapshot",
]
