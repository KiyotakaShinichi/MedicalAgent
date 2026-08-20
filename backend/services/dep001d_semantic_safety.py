"""DEP-001D runtime adapter over the audited DEP-001B inference contract."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from backend.services.dep001b_semantic_safety import (
    DEP001BSafetyPrediction,
    classify_dep001b_safety,
    clear_dep001b_runtime_cache,
)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "Data/evals/safety/dep001d/runtime"


def classify_dep001d_safety(
    query: str,
    *,
    previous_user_messages: Sequence[str] | None = None,
) -> DEP001BSafetyPrediction:
    return classify_dep001b_safety(
        query,
        previous_user_messages=previous_user_messages,
        artifact_dir=RUNTIME_DIR,
    )


def clear_dep001d_runtime_cache() -> None:
    clear_dep001b_runtime_cache()


__all__ = [
    "RUNTIME_DIR",
    "classify_dep001d_safety",
    "clear_dep001d_runtime_cache",
]
