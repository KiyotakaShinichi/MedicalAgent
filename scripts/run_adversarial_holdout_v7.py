"""Freeze or execute the one-pass internal adversarial holdout v7."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_holdout_v7 import (
    evaluate_holdout_v7,
    freeze_holdout_v7,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("freeze", "evaluate"),
        help="Freeze first, then evaluate exactly once after all tuning is closed.",
    )
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze_holdout_v7()
        print(
            f"frozen total_n={result['total_n']} sha256={result['sha256']}"
        )
    else:
        result = evaluate_holdout_v7()
        print(
            f"status={result['status']} pass_rate={result['pass_rate']} "
            f"unsafe_leakage_rate={result['unsafe_leakage_rate']} "
            f"over_refusal_rate={result['over_refusal_rate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
