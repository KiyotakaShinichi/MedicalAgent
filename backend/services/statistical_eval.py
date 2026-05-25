"""Statistical reporting helpers for engineering eval artifacts.

The helpers in this module are intentionally lightweight.  They add
uncertainty intervals and before/after comparison scaffolding to internal
benchmarks, but they do not turn curated or synthetic metrics into clinical
evidence.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


STATISTICAL_EVAL_VERSION = "statistical_eval_v1_2026_05"
CLAIM_BOUNDARY = (
    "Confidence intervals and deltas here describe internal engineering "
    "benchmarks only. They do not establish clinical validation, real-world "
    "safety, patient benefit, or production healthcare readiness."
)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> dict[str, Any]:
    """Return a Wilson score interval for a binomial proportion."""

    if total <= 0:
        return {
            "method": "wilson_score",
            "confidence": confidence,
            "successes": successes,
            "total_n": total,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
        }
    z = 1.959963984540054 if confidence == 0.95 else _normal_z(confidence)
    phat = successes / total
    denom = 1 + (z * z / total)
    centre = phat + (z * z / (2 * total))
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return {
        "method": "wilson_score",
        "confidence": confidence,
        "successes": int(successes),
        "total_n": int(total),
        "estimate": round(phat, 6),
        "ci_low": round(max(0.0, (centre - radius) / denom), 6),
        "ci_high": round(min(1.0, (centre + radius) / denom), 6),
    }


def mean_interval(values: list[float], confidence: float = 0.95) -> dict[str, Any]:
    """Return a normal-approximation interval for a mean.

    This is used for fold-level or route-level summary metrics where raw
    examples are not always persisted.  The artifact names the method so a
    reviewer can tell this is a rough engineering interval, not a definitive
    statistical analysis.
    """

    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {
            "method": "normal_approximation_mean",
            "confidence": confidence,
            "total_n": 0,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
        }
    estimate = mean(clean)
    if len(clean) == 1:
        return {
            "method": "single_observation_no_interval",
            "confidence": confidence,
            "total_n": 1,
            "estimate": round(estimate, 6),
            "ci_low": None,
            "ci_high": None,
        }
    z = 1.959963984540054 if confidence == 0.95 else _normal_z(confidence)
    sd = pstdev(clean)
    margin = z * sd / math.sqrt(len(clean))
    return {
        "method": "normal_approximation_mean",
        "confidence": confidence,
        "total_n": len(clean),
        "estimate": round(estimate, 6),
        "ci_low": round(estimate - margin, 6),
        "ci_high": round(estimate + margin, 6),
        "std": round(sd, 6),
    }


def two_proportion_delta(
    *,
    baseline_successes: int,
    baseline_total: int,
    candidate_successes: int,
    candidate_total: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return an approximate CI for candidate - baseline proportion."""

    if baseline_total <= 0 or candidate_total <= 0:
        return {
            "method": "two_proportion_normal_approximation",
            "confidence": confidence,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
        }
    p0 = baseline_successes / baseline_total
    p1 = candidate_successes / candidate_total
    delta = p1 - p0
    se = math.sqrt((p0 * (1 - p0) / baseline_total) + (p1 * (1 - p1) / candidate_total))
    z = 1.959963984540054 if confidence == 0.95 else _normal_z(confidence)
    return {
        "method": "two_proportion_normal_approximation",
        "confidence": confidence,
        "baseline": {
            "successes": int(baseline_successes),
            "total_n": int(baseline_total),
            "estimate": round(p0, 6),
        },
        "candidate": {
            "successes": int(candidate_successes),
            "total_n": int(candidate_total),
            "estimate": round(p1, 6),
        },
        "delta": round(delta, 6),
        "ci_low": round(delta - z * se, 6),
        "ci_high": round(delta + z * se, 6),
    }


def binomial_metric(
    *,
    name: str,
    artifact_path: str,
    successes: int,
    total: int,
    estimate_name: str = "pass_rate",
    claim_boundary: str | None = None,
) -> dict[str, Any]:
    interval = wilson_interval(successes, total)
    return {
        "name": name,
        "artifact_path": artifact_path,
        "metric_type": "binomial_proportion",
        "estimate_name": estimate_name,
        **interval,
        "claim_boundary": claim_boundary or CLAIM_BOUNDARY,
    }


def fold_mean_metric(
    *,
    name: str,
    artifact_path: str,
    values: list[float],
    estimate_name: str,
    claim_boundary: str | None = None,
) -> dict[str, Any]:
    interval = mean_interval(values)
    return {
        "name": name,
        "artifact_path": artifact_path,
        "metric_type": "fold_mean",
        "estimate_name": estimate_name,
        **interval,
        "claim_boundary": claim_boundary or CLAIM_BOUNDARY,
    }


def artifact_exists(path: str | Path) -> bool:
    return Path(path).exists()


def _normal_z(confidence: float) -> float:
    # The current callers use 95%; this fallback keeps the API honest without
    # pulling scipy into CI.
    if abs(confidence - 0.90) < 1e-9:
        return 1.6448536269514722
    if abs(confidence - 0.99) < 1e-9:
        return 2.5758293035489004
    return 1.959963984540054


__all__ = [
    "CLAIM_BOUNDARY",
    "STATISTICAL_EVAL_VERSION",
    "artifact_exists",
    "binomial_metric",
    "fold_mean_metric",
    "mean_interval",
    "two_proportion_delta",
    "wilson_interval",
]
