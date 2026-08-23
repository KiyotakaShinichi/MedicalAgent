"""Evaluation artifacts the guide reports on, loaded once per document.

The guide is evidence-led: it reads the current local evaluation artifacts at
generation time rather than restating numbers written by hand. `Evidence.load`
performs every read in one place, so a section module cannot quietly introduce
a different source for a number, and a missing artifact degrades to "not
reported" instead of raising.

The loading logic is relocated unchanged from the head of the former
``build_story``; it is now named state rather than eighteen local variables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from scripts.project_guide.theme import ROOT


def _load(relative: str, default: Any | None = None) -> Any:
    path = ROOT / relative
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def _dig(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "not reported"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "not reported"
    return f"{100 * value:.{digits}f}%"


def _configuration(artifact: dict[str, Any], config_id: str) -> dict[str, Any]:
    configurations = artifact.get("configurations", {})
    if isinstance(configurations, dict):
        item = configurations.get(config_id, {})
        if isinstance(item, dict):
            summary = item.get("summary")
            return summary if isinstance(summary, dict) else item
        return {}
    for item in configurations:
        if isinstance(item, dict) and (
            item.get("configuration_id") == config_id or item.get("id") == config_id
        ):
            summary = item.get("summary")
            return summary if isinstance(summary, dict) else item
    return {}


@dataclass(frozen=True)
class Evidence:
    """Every artifact-derived value the document sections read."""

    rag: Any
    prompt_eval: Any
    safety: Any
    safety_v4: Any
    temporal: Any
    paired: Any
    per_head: Any
    conformal: Any
    latency: Any
    sentinel: Any
    bm25: Any
    full: Any
    rag_summary: Any
    best: Any
    temporal_cv: Any

    @classmethod
    def load(cls) -> "Evidence":
        """Read the current artifacts from disk. Missing files become {}."""
        rag = _load("Data/evals/rag/latest_rag_baseline_comparison.json")
        prompt_eval = _load("Data/evals/agentic_tool_use/latest_large_scale_agent_prompt_eval.json")
        safety = _load("Data/evals/safety/latest_adversarial_safety_regression.json")
        safety_v4 = _load("Data/evals/safety/latest_adversarial_holdout_v4_baseline.json")
        temporal = _load("Data/evals/models/latest_patient_temporal_cv.json")
        paired = _load("Data/evals/models/latest_paired_model_comparison.json")
        per_head = _load("Data/evals/models/latest_per_head_calibration.json")
        conformal = _load("Data/evals/models/latest_response_conformal_calibration.json")
        latency = _load("Data/evals/ops/latest_route_latency_budget.json")
        sentinel = _load("Data/evals/ops/latest_runtime_quality_sentinel.json")

        bm25 = _configuration(rag, "bm25_only")
        full = _configuration(rag, "hybrid_rrf_query_rewrite_parent_child_source_tier")
        rag_summary = rag.get("summary", {})
        best = _configuration(
            rag,
            str(rag_summary.get("best_configuration", rag.get("best_configuration_id", "hybrid_rrf_query_rewrite"))),
        )
        if not best:
            best = _configuration(rag, "hybrid_rrf_query_rewrite")

        # Derived in the former build_story between sections 10 and 11;
        # a field here so both section modules read one definition.
        temporal_cv = temporal.get("patient_level_temporal_cv", {})
        return cls(
            rag=rag,
            prompt_eval=prompt_eval,
            safety=safety,
            safety_v4=safety_v4,
            temporal=temporal,
            paired=paired,
            per_head=per_head,
            conformal=conformal,
            latency=latency,
            sentinel=sentinel,
            bm25=bm25,
            full=full,
            rag_summary=rag_summary,
            best=best,
            temporal_cv=temporal_cv,
        )


__all__ = ["Evidence", "_configuration", "_dig", "_fmt", "_load", "_pct"]
