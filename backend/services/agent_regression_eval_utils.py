"""Pure aggregation helpers for agent regression evaluation."""

from __future__ import annotations

from statistics import mean


def source_ids(items):
    return [
        str(item.get("id"))
        for item in items
        if isinstance(item, dict) and item.get("id")
    ]


def numeric(values):
    return [float(value) for value in values if value is not None]


def round_mean(values):
    return round(mean(values), 3) if values else None


def rate(numerator, denominator):
    if not denominator:
        return None
    return round(numerator / denominator, 3)


def status_meaning(status):
    meanings = {
        "failed": "One or more hard safety or guardrail gates failed.",
        "unideal": "Safety passed, but retrieval or routing quality needs work.",
        "acceptable": "Safe enough for PoC regression, with quality gaps to improve.",
        "strong": "All current regression gates passed with good quality proxies.",
    }
    return meanings.get(status, "No status meaning available.")


__all__ = ["numeric", "rate", "round_mean", "source_ids", "status_meaning"]
