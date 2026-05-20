"""Run the held-out adversarial variants (4 hardened categories only).

Reads ``Data/evals/safety/adversarial_safety_holdout_variants.jsonl``
and writes ``Data/evals/safety/latest_adversarial_safety_holdout.json``.

The held-out result is reported **alongside** the in-sample bank's
score in the eval drift artifact so a reviewer can see whether
hardening generalized or merely memorized.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("ONCOTRACK_FAST_MODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_adversarial_safety_regression import _evaluate_case  # noqa: E402


DEFAULT_BANK = Path("Data/evals/safety/adversarial_safety_holdout_variants.jsonl")
DEFAULT_OUTPUT = Path("Data/evals/safety/latest_adversarial_safety_holdout.json")


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["passed"]:
            by_cat[r["category"]]["passed"] += 1
    return {
        "schema_version": "1.0",
        "status": "informational",
        "label": "held_out_adversarial_variants_post_hardening",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_n": total,
        "pass_count": passed,
        "fail_count": total - passed,
        "skipped_count": 0,
        "overall_attack_block_rate": (passed / total) if total else 0.0,
        "by_category": {
            cat: {
                "total_n": d["total"],
                "pass_count": d["passed"],
                "fail_count": d["total"] - d["passed"],
                "attack_block_rate": (d["passed"] / d["total"]) if d["total"] else 0.0,
            }
            for cat, d in sorted(by_cat.items())
        },
        "contamination_note": (
            "These queries were NOT used to tune any deterministic pattern. "
            "They share the four hardened categories with the original bank "
            "but use fresh wording.  A held-out rate noticeably below the "
            "in-sample bank rate is evidence of bank-tuning, not generalization."
        ),
        "claim_boundary": (
            "Adversarial pass rates are engineering safety-regression signals only. "
            "They do not establish clinical safety or real-world patient benefit."
        ),
        "results": results,
    }


def run(bank_path: Path = DEFAULT_BANK, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if not bank_path.exists():
        raise FileNotFoundError(f"holdout bank missing: {bank_path}")
    results: list[dict[str, Any]] = []
    with bank_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            results.append(_evaluate_case(case))
    summary = _summarize(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default=str(DEFAULT_BANK))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    summary = run(Path(args.bank), Path(args.output))
    print(
        f"holdout cases={summary['total_n']}  "
        f"passed={summary['pass_count']}  "
        f"attack_block_rate={summary['overall_attack_block_rate']:.4f}"
    )
    for cat, d in summary["by_category"].items():
        print(f"  {cat:42s} {d['pass_count']}/{d['total_n']}  rate={d['attack_block_rate']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
