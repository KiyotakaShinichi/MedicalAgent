"""Freeze or evaluate the one-pass prototype-independent prompt bank v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.prototype_independent_prompt_bank_v2 import (
    DEFAULT_BANK_PATH,
    DEFAULT_MANIFEST_PATH,
    freeze_prompt_bank_v2,
)
from backend.services.prototype_independent_prompt_eval_v2 import (
    evaluate_frozen_prompt_bank_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze_prompt_bank_v2()
        print(f"Frozen {result['total_n']} cases: {result['sha256']}")
        return
    result = evaluate_frozen_prompt_bank_v2(DEFAULT_BANK_PATH, DEFAULT_MANIFEST_PATH)
    print(
        "Prototype-independent v2: "
        f"{result['pass_count']}/{result['total_n']} "
        f"({result['pass_rate']:.4f}), "
        f"unsafe leakage={result['unsafe_leakage_rate']:.4f}, "
        f"over-refusal={result['over_refusal_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
