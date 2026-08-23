"""Shared step primitives for the ship orchestrator.

Holds the pieces every step module needs — the ``Step`` record, the repository
and frontend roots, and the npm invocation helper. They live here rather than
in ``scripts/ship.py`` so the step modules can import them without importing
the orchestrator that imports *them*, which would be circular.

``scripts.ship`` re-exports these names, so existing imports of
``scripts.ship.Step`` keep working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend-react"

__all__ = ["FRONTEND", "ROOT", "Step", "npm_cmd"]


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    cwd: Path = ROOT
    env: dict[str, str] | None = None
    timeout_seconds: int | None = None


def npm_cmd(*args: str) -> list[str]:
    """npm invocation that also works where only ``npm.cmd`` is on PATH."""
    executable = "npm.cmd" if os.name == "nt" else "npm"
    return [executable, *args]
