"""Immutable data contract for unsafe-intent families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnsafeFamily:
    family: str
    expected_route: str
    expected_scope: str
    safe_template: str
    over_refusal_risk_notes: str
    positive_prototypes: tuple[str, ...]
    safe_negative_prototypes: tuple[str, ...]
    near_boundary_examples: tuple[str, ...]
    taglish_variants: tuple[str, ...] = ()
    deterministic_patterns: tuple[str, ...] = ()
