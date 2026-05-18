"""Install the OncoTrack pre-commit integration gate.

Drops a ``pre-commit`` hook into ``.git/hooks`` that runs the
breast-monitoring integration suite. The hook refuses the commit on any
failure.

Usage:
    python scripts/install_pre_commit.py
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK_DIR = ROOT / ".git" / "hooks"
HOOK_PATH = HOOK_DIR / "pre-commit"
TRACKED_HOOK = ROOT / ".githooks" / "pre-commit"

HOOK_BODY = """#!/bin/sh
# OncoTrack pre-commit integration gate.
# Auto-installed by scripts/install_pre_commit.py.
# Skip with: SKIP_ONCOTRACK_GATE=1 git commit ...
set -e
if [ "${SKIP_ONCOTRACK_GATE:-0}" = "1" ]; then
  echo "[oncotrack] gate skipped via SKIP_ONCOTRACK_GATE=1"
  exit 0
fi
echo "[oncotrack] running tests/test_breast_monitoring.py ..."
RAG_FORCE_SPARSE=true python -m pytest tests/test_breast_monitoring.py -q --tb=line
"""


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    if not HOOK_DIR.exists():
        print(f"[install_pre_commit] expected .git/hooks at {HOOK_DIR}", file=sys.stderr)
        return 1
    _write_executable(HOOK_PATH, HOOK_BODY)
    _write_executable(TRACKED_HOOK, HOOK_BODY)
    print(f"[install_pre_commit] installed local hook -> {HOOK_PATH}")
    print(f"[install_pre_commit] refreshed tracked hook -> {TRACKED_HOOK}")
    print("[install_pre_commit] override with: SKIP_ONCOTRACK_GATE=1 git commit ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
