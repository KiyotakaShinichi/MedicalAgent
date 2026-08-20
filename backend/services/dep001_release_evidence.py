"""Build release-gate-compatible evidence from the immutable DEP-001 result.

This module only adds compatibility aliases required by the legacy release-gate
schema. It must not recalculate metrics, alter the official decision, or mutate
the canonical external-evaluation artifact.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping


def build_release_compatible_evidence(
    canonical: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Return a release-compatible copy without changing official values."""
    if canonical.get("schema_version") != "dep001_external_safety_evaluation_v1":
        raise ValueError("unexpected DEP-001 external artifact schema")

    metrics = canonical.get("metrics")
    fault_injection = canonical.get("fault_injection")
    if not isinstance(metrics, Mapping) or not isinstance(fault_injection, Mapping):
        raise ValueError("canonical artifact is missing metrics or fault-injection evidence")

    required_metrics = (
        "unsafe_released_output_rate",
        "en_taglish_absolute_gap",
        "indirect_intent_unsafe_recall",
        "obfuscated_unsafe_recall",
        "multi_turn_unsafe_recall",
        "rag_conditioned_unsafe_recall",
    )
    missing = [name for name in required_metrics if name not in metrics]
    if missing:
        raise ValueError(f"canonical artifact is missing required metrics: {missing}")

    derived = deepcopy(dict(canonical))
    derived_metrics = derived["metrics"]
    alias_values = {
        "unsafe_pass_rate": float(metrics["unsafe_released_output_rate"]),
        "en_taglish_parity": 1.0 - float(metrics["en_taglish_absolute_gap"]),
        "adversarial_paraphrase_robustness": min(
            float(metrics["indirect_intent_unsafe_recall"]),
            float(metrics["obfuscated_unsafe_recall"]),
        ),
        "multi_turn_safety": float(metrics["multi_turn_unsafe_recall"]),
        "rag_conditioned_safety": float(metrics["rag_conditioned_unsafe_recall"]),
        "failure_path_safety": 1.0 if fault_injection.get("passed") is True else 0.0,
    }
    derived_metrics.update(alias_values)
    derived["release_compatibility"] = {
        "compatibility_aliases_only": True,
        "canonical_source_path": source_path,
        "canonical_source_sha256": source_sha256,
        "reconciled_at": datetime.now(timezone.utc).isoformat(),
        "aliases": {
            "metrics.unsafe_pass_rate": "metrics.unsafe_released_output_rate",
            "metrics.en_taglish_parity": "1 - metrics.en_taglish_absolute_gap",
            "metrics.adversarial_paraphrase_robustness": (
                "min(metrics.indirect_intent_unsafe_recall, "
                "metrics.obfuscated_unsafe_recall)"
            ),
            "metrics.multi_turn_safety": "metrics.multi_turn_unsafe_recall",
            "metrics.rag_conditioned_safety": "metrics.rag_conditioned_unsafe_recall",
            "metrics.failure_path_safety": "1.0 if fault_injection.passed else 0.0",
        },
        "official_decision_unchanged": (
            derived.get("status") == canonical.get("status")
            and derived.get("dep001_complete") == canonical.get("dep001_complete")
            and derived.get("dep001_decision") == canonical.get("dep001_decision")
        ),
    }
    return derived


__all__ = ["build_release_compatible_evidence"]
