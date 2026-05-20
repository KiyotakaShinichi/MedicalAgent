"""Append-only admin action audit artifact.

This lightweight file-backed audit log is for the local engineering prototype.
It is not a production audit-log subsystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.services.structured_logging import build_event


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT_DIR / "Data/evals/ops/admin_action_audit.jsonl"


def append_admin_action(
    action: str,
    *,
    admin_id: str | None = None,
    artifact_id: str | None = None,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
    output_path: str | Path = DEFAULT_PATH,
) -> dict[str, Any]:
    event = build_event(
        "admin_action",
        severity="info",
        request_id=request_id,
        user_role="admin",
        artifact_id=artifact_id,
        details={"action": action, "admin_id": admin_id, **(details or {})},
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


__all__ = ["append_admin_action"]
