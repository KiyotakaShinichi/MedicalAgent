"""Sensitivity sweep for the three RAG-agent magic numbers.

Three constants in the agent stack are unit-less and were chosen by
intuition during the original Phase-11 build:

  - ``SUPPORTED_THRESHOLD``         (rag_claim_validator.py, default 0.30)
  - ``WEAKLY_SUPPORTED_THRESHOLD``  (rag_claim_validator.py, default 0.12)
  - ``LLM_CONFIDENCE_FLOOR``        (agent_intent_router.py, default 0.72)

This module sweeps each constant against a fixed in-repo probe set and
reports how the downstream outcomes (claim-status distribution; LLM
override-accept-rate) move.  The goal is **not** to pick a "best"
value — the goal is to surface whether the current defaults are on a
plateau (insensitive to small changes) or on a cliff (one step changes
behaviour drastically).

Outputs
~~~~~~~
``Data/evals/rag/latest_rag_threshold_calibration.json``::

    {
      "schema_version": "1.0",
      "status": "informational",
      "label": "internal_engineering_eval_threshold_sensitivity",
      "constants": {
        "SUPPORTED_THRESHOLD":         {"default": 0.30, "sweep": [{...}]},
        "WEAKLY_SUPPORTED_THRESHOLD":  {"default": 0.12, "sweep": [{...}]},
        "LLM_CONFIDENCE_FLOOR":        {"default": 0.72, "sweep": [{...}]}
      },
      "verdict_per_constant": {...},
      "claim_boundary": "Threshold sensitivity is an engineering signal..."
    }
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.services.rag_claim_validator import (
    SUPPORTED_THRESHOLD,
    WEAKLY_SUPPORTED_THRESHOLD,
)
from backend.services.agent_intent_router import LLM_CONFIDENCE_FLOOR


OUTPUT_PATH = Path("Data/evals/rag/latest_rag_threshold_calibration.json")


# ─── Probe set: 12 (sentence, supporting-context, expected-status) cases ──
#
# Hand-curated so the expected status under the default thresholds is
# known and stable.  Each case is independent — no patient data.
PROBES: list[dict[str, Any]] = [
    # Clearly-supported claims (high overlap with context).
    {
        "id": "P-001",
        "sentence": "Trastuzumab is generally given every three weeks during HER2-positive therapy.",
        "chunks": [
            "Trastuzumab is typically administered every three weeks for HER2-positive disease during chemotherapy.",
        ],
        "expected_default": "supported",
    },
    {
        "id": "P-002",
        "sentence": "Doxorubicin can lower white blood cell counts after each cycle.",
        "chunks": [
            "Doxorubicin is associated with white blood cell suppression after each chemotherapy cycle.",
        ],
        "expected_default": "supported",
    },
    {
        "id": "P-003",
        "sentence": "Tumor markers like CA 15-3 alone are not enough to confirm recurrence.",
        "chunks": [
            "A single tumor marker measurement such as CA 15-3 is not sufficient to confirm cancer recurrence; imaging and clinical context are required.",
        ],
        "expected_default": "supported",
    },
    # Weakly supported — partial overlap.
    {
        "id": "P-004",
        "sentence": "Patients sometimes ask about supplements before chemotherapy.",
        "chunks": [
            "Some patients ask their oncology team about supplements during cancer therapy.",
        ],
        "expected_default": "weakly_supported",
    },
    {
        "id": "P-005",
        "sentence": "Anemia can make daily tasks feel harder during treatment.",
        "chunks": [
            "Anemia may cause fatigue and reduced exercise tolerance in patients receiving cancer therapy.",
        ],
        "expected_default": "weakly_supported",
    },
    # Unsupported — no overlap.
    {
        "id": "P-006",
        "sentence": "A daily walk after dinner improves sleep quality.",
        "chunks": [
            "Pegfilgrastim is administered subcutaneously after dose-dense chemotherapy.",
        ],
        "expected_default": "unsupported",
    },
    {
        "id": "P-007",
        "sentence": "Patients should drink eight glasses of water per day.",
        "chunks": [
            "Anthracycline regimens are monitored with cardiac function assessments before and during therapy.",
        ],
        "expected_default": "unsupported",
    },
    {
        "id": "P-008",
        "sentence": "Yoga is often discussed in patient support groups.",
        "chunks": [
            "Granulocyte colony-stimulating factor reduces neutropenia incidence during dose-dense chemotherapy.",
        ],
        "expected_default": "unsupported",
    },
    # Near-boundary — sits close to SUPPORTED_THRESHOLD.
    {
        "id": "P-009",
        "sentence": "MRI imaging is one of the modalities used for breast cancer response assessment.",
        "chunks": [
            "Magnetic resonance imaging is part of breast cancer response assessment in many treatment protocols.",
        ],
        "expected_default": "supported",
    },
    {
        "id": "P-010",
        "sentence": "Neutropenia can require dose adjustments in some cycles.",
        "chunks": [
            "Significant neutropenia during chemotherapy may prompt the care team to consider dose adjustments.",
        ],
        "expected_default": "supported",
    },
    # Near-boundary — sits close to WEAKLY_SUPPORTED_THRESHOLD.
    {
        "id": "P-011",
        "sentence": "Genetic counseling can be helpful for some families.",
        "chunks": [
            "Patients with a strong family history may benefit from genetic counseling referrals.",
        ],
        "expected_default": "weakly_supported",
    },
    {
        "id": "P-012",
        "sentence": "Side effects vary across regimens.",
        "chunks": [
            "Adverse events differ between chemotherapy combinations and across individual patients.",
        ],
        "expected_default": "weakly_supported",
    },
]


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "is", "are",
    "be", "been", "for", "with", "as", "at", "by", "this", "that", "it",
    "from", "may", "can", "do", "does", "during", "such", "some", "after",
})


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def _overlap(sentence: str, chunks: Iterable[str]) -> float:
    s = _tokens(sentence)
    if not s:
        return 0.0
    best = 0.0
    for c in chunks:
        c_toks = _tokens(c)
        if not c_toks:
            continue
        ov = len(s & c_toks) / max(1, len(s))
        if ov > best:
            best = ov
    return best


def _classify(score: float, supported: float, weak: float) -> str:
    if score >= supported:
        return "supported"
    if score >= weak:
        return "weakly_supported"
    return "unsupported"


# ─── Sweep ───────────────────────────────────────────────────────────────

@dataclass
class SweepPoint:
    value: float
    distribution: dict[str, int]
    agreement_with_default: float
    flips_from_default: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "distribution": self.distribution,
            "agreement_with_default": round(self.agreement_with_default, 4),
            "flips_from_default": self.flips_from_default,
        }


def _classify_all(
    supported: float,
    weak: float,
) -> tuple[list[str], dict[str, int]]:
    statuses: list[str] = []
    dist = {"supported": 0, "weakly_supported": 0, "unsupported": 0}
    for case in PROBES:
        score = _overlap(case["sentence"], case["chunks"])
        s = _classify(score, supported, weak)
        statuses.append(s)
        dist[s] += 1
    return statuses, dist


def _sweep_constant(
    name: str,
    values: list[float],
    default_value: float,
    other_value: float,
) -> list[SweepPoint]:
    # Compute default statuses once.
    if name == "SUPPORTED_THRESHOLD":
        default_statuses, _ = _classify_all(default_value, other_value)
    else:
        default_statuses, _ = _classify_all(other_value, default_value)

    out: list[SweepPoint] = []
    for v in values:
        if name == "SUPPORTED_THRESHOLD":
            statuses, dist = _classify_all(v, other_value)
        else:
            statuses, dist = _classify_all(other_value, v)
        flips = sum(1 for a, b in zip(default_statuses, statuses) if a != b)
        agreement = 1 - flips / len(PROBES)
        out.append(SweepPoint(value=v, distribution=dist, agreement_with_default=agreement, flips_from_default=flips))
    return out


def _confidence_floor_sweep(values: list[float]) -> list[dict[str, Any]]:
    # The router accepts an LLM intent iff confidence >= floor.  We
    # don't sweep the live LLM here (it would burn budget) — instead
    # we report the *fraction of a fixed synthetic confidence
    # distribution* that would be accepted at each floor.  The fixed
    # distribution mimics the LLM's empirical histogram in
    # Data/evals/* router probes (skewed high with a long tail).
    fixed_confidences: list[float] = [
        0.05, 0.12, 0.20, 0.31, 0.39, 0.48, 0.55, 0.61, 0.66, 0.70,
        0.72, 0.74, 0.77, 0.81, 0.84, 0.86, 0.88, 0.90, 0.92, 0.95,
    ]
    return [
        {
            "value": v,
            "accept_fraction": round(
                sum(1 for c in fixed_confidences if c >= v) / len(fixed_confidences),
                4,
            ),
        }
        for v in values
    ]


# ─── Verdict ────────────────────────────────────────────────────────────

def _verdict_label(min_agree: float) -> str:
    if min_agree >= 0.95:
        return "plateau"
    if min_agree >= 0.80:
        return "soft_slope"
    return "cliff"


def build_calibration_report() -> dict[str, Any]:
    supported_values = [0.20, 0.25, 0.30, 0.35, 0.40]
    weak_values      = [0.06, 0.10, 0.12, 0.16, 0.20]
    floor_values     = [0.60, 0.66, 0.72, 0.78, 0.84]

    supported_sweep = _sweep_constant(
        "SUPPORTED_THRESHOLD", supported_values, SUPPORTED_THRESHOLD, WEAKLY_SUPPORTED_THRESHOLD,
    )
    weak_sweep = _sweep_constant(
        "WEAKLY_SUPPORTED_THRESHOLD", weak_values, WEAKLY_SUPPORTED_THRESHOLD, SUPPORTED_THRESHOLD,
    )
    floor_sweep = _confidence_floor_sweep(floor_values)

    sup_min = min(p.agreement_with_default for p in supported_sweep)
    weak_min = min(p.agreement_with_default for p in weak_sweep)

    return {
        "schema_version": "1.0",
        "status": "informational",
        "label": "internal_engineering_eval_threshold_sensitivity",
        "claim_boundary": (
            "Threshold sensitivity is an engineering signal only. Confirms whether "
            "the current defaults sit on a plateau (small changes do not flip "
            "outcomes) or on a cliff. It does not validate the absolute values or "
            "establish any clinical property."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_probes": len(PROBES),
        "constants": {
            "SUPPORTED_THRESHOLD": {
                "default": SUPPORTED_THRESHOLD,
                "swept_values": supported_values,
                "sweep": [p.to_dict() for p in supported_sweep],
                "verdict": _verdict_label(sup_min),
            },
            "WEAKLY_SUPPORTED_THRESHOLD": {
                "default": WEAKLY_SUPPORTED_THRESHOLD,
                "swept_values": weak_values,
                "sweep": [p.to_dict() for p in weak_sweep],
                "verdict": _verdict_label(weak_min),
            },
            "LLM_CONFIDENCE_FLOOR": {
                "default": LLM_CONFIDENCE_FLOOR,
                "swept_values": floor_values,
                "sweep": floor_sweep,
                "verdict": (
                    "informational — accept_fraction monotone in floor by construction; "
                    "report shape over fixed synthetic confidence distribution"
                ),
            },
        },
        "guidance": (
            "If any verdict is 'cliff', the default value sits on a steep boundary "
            "and small changes substantially alter outcomes — consider tightening "
            "the probe set and re-running before changing the default."
        ),
    }


def write_calibration_report(output_path: Path = OUTPUT_PATH) -> Path:
    report = build_calibration_report()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "OUTPUT_PATH",
    "PROBES",
    "build_calibration_report",
    "write_calibration_report",
]
