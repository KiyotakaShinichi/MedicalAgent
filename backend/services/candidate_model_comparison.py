from __future__ import annotations

from backend.services.realism_candidate_ab_gate import (
    LEGACY_OUTPUT_PATH as DEFAULT_OUTPUT_PATH,
    run_realism_candidate_ab_gate,
)


def build_current_vs_candidate_report(
    current_metrics_path: str | None = None,
    candidate_metrics_path: str | None = None,
    current_realism_path: str | None = None,
    candidate_realism_path: str | None = None,
    output_path: str | None = DEFAULT_OUTPUT_PATH,
) -> dict:
    """Backward-compatible admin wrapper for the stricter A/B gate.

    The old implementation compared precomputed summary metrics and could emit
    optimistic promotion language. The current wrapper ignores the legacy metric
    path arguments and runs the explicit current-vs-public-realism-candidate
    A/B gate instead.
    """

    return run_realism_candidate_ab_gate(legacy_output_path=output_path)


__all__ = ["DEFAULT_OUTPUT_PATH", "build_current_vs_candidate_report"]
