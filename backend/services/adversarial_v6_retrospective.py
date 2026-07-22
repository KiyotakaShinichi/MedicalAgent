"""Retrospective disclosure for the internally authored adversarial v6 bank.

This module never evaluates the bank. It records that maintainers inspected its
failures, which makes subsequent classifier work tuning-informed rather than
held-out evidence.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("Data/evals/safety/latest_adversarial_holdout_v6_baseline.json")
DEFAULT_OUTPUT = Path("Data/evals/safety/latest_adversarial_v6_retrospective.json")


def build_retrospective(input_path: str | Path = DEFAULT_INPUT) -> dict[str, Any]:
    baseline = json.loads(Path(input_path).read_text(encoding="utf-8"))
    failures = list(baseline.get("failures") or [])
    category_counts = Counter(str(row.get("category") or "unknown") for row in failures)
    observed_routes = Counter(str(row.get("observed_route") or "unknown") for row in failures)
    missed = sum(1 for row in failures if row.get("observed_family") in {None, "none"})
    safe_conflicts = sum(1 for row in failures if row.get("safe_negative"))
    return {
        "schema_version": "adversarial_v6_retrospective_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "tuning_informed_not_held_out",
        "source_artifact": str(input_path).replace("\\", "/"),
        "source_bank_sha256": baseline.get("bank_sha256"),
        "source_total_n": baseline.get("total_n"),
        "source_pass_rate": baseline.get("pass_rate"),
        "source_unsafe_leakage_rate": baseline.get("unsafe_leakage_rate"),
        "failed_n": len(failures),
        "missed_unsafe_intent_n": missed,
        "safe_negative_conflict_n": safe_conflicts,
        "failure_categories": dict(sorted(category_counts.items())),
        "observed_failure_routes": dict(sorted(observed_routes.items())),
        "representative_case_ids": [str(row.get("case_id")) for row in failures[:12]],
        "was_used_for_tuning": True,
        "internal_vs_external": "internal_author_retrospective_tuning_diagnostic",
        "frozen_bank_was_rerun": False,
        "allowed_reading": "The v6 baseline identified generalization weaknesses and informed development mutations.",
        "blocked_readings": [
            "independent held-out evidence after failure inspection",
            "external red-team evidence",
            "clinical safety validation",
            "proof that safety is solved",
        ],
        "next_independent_step": "Use a separately authored no-read external adversarial bank; do not tune on it.",
        "clinical_validation": False,
        "claim_boundary": "Retrospective engineering disclosure only; not clinical validation or real-world safety evidence.",
    }


def write_retrospective(output_path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_retrospective()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = ["build_retrospective", "write_retrospective"]
