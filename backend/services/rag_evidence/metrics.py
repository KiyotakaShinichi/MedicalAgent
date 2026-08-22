"""PHI-free counters for evidence release decisions."""

from __future__ import annotations

import threading
from collections import Counter


_METRICS: Counter[str] = Counter()
_METRICS_LOCK = threading.Lock()


def snapshot_evidence_release_metrics() -> dict[str, int]:
    with _METRICS_LOCK:
        return dict(_METRICS)


def record_rag_cache_rejection() -> None:
    """Increment the PHI-free cache rejection counter."""

    increment("rag_cache_rejected_total")


def increment(metric: str) -> None:
    with _METRICS_LOCK:
        _METRICS[metric] += 1
