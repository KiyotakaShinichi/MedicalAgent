"""Tuning-informed regression over the previously inspected v6 bank.

The historical v6 baseline remains immutable. Once its failures were inspected,
any later execution is development regression evidence, never held-out evidence.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.adversarial_holdout_v6 import (
    DEFAULT_BANK_PATH,
    DEFAULT_MANIFEST_PATH,
    evaluate_holdout_v6,
)


DEFAULT_BASELINE_PATH = Path(
    "Data/evals/safety/latest_adversarial_holdout_v6_baseline.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/safety/latest_adversarial_v6_tuning_regression.json"
)


def build_v6_tuning_regression(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    baseline = json.loads(DEFAULT_BASELINE_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="nlcare-v6-regression-") as directory:
        scratch = Path(directory) / "v6.json"
        current = evaluate_holdout_v6(
            bank_path=DEFAULT_BANK_PATH,
            manifest_path=DEFAULT_MANIFEST_PATH,
            output_path=scratch,
        )
    before = float(baseline.get("pass_rate") or 0.0)
    after = float(current.get("pass_rate") or 0.0)
    payload = {
        "schema_version": "adversarial_v6_tuning_regression_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if after >= before else "needs_attention",
        "source_bank_sha256": current.get("bank_sha256"),
        "baseline_artifact": DEFAULT_BASELINE_PATH.as_posix(),
        "historical_baseline_preserved": True,
        "baseline_pass_rate": before,
        "tuning_regression_pass_rate": after,
        "pass_rate_delta": round(after - before, 6),
        "total_n": current.get("total_n"),
        "pass_count": current.get("pass_count"),
        "fail_count": current.get("fail_count"),
        "unsafe_leakage_rate": current.get("unsafe_leakage_rate"),
        "over_refusal_rate": current.get("over_refusal_rate"),
        "by_category": current.get("by_category"),
        "failures": current.get("failures"),
        "was_used_for_tuning": True,
        "internal_vs_external": "internal_tuning_regression_after_failure_inspection",
        "independent_holdout_evidence": False,
        "external_author_eval_completed": False,
        "clinical_validation": False,
        "allowed_reading": (
            "The generalized classifier changed performance on a previously "
            "inspected internal bank."
        ),
        "blocked_readings": [
            "independent held-out generalization",
            "external red-team evidence",
            "proof that safety is solved",
            "clinical validation",
        ],
        "claim_boundary": (
            "Tuning-informed internal regression only; not held-out, external, "
            "clinical, or real-world safety evidence."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = ["build_v6_tuning_regression"]
