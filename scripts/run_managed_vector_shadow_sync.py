from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.managed_vector_shadow_sync import (
    build_managed_vector_shadow_sync,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the governed gold-record sync plan or explicitly apply "
            "it to Azure Search."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Embed and synchronize records. Disabled by default.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    report = build_managed_vector_shadow_sync(
        apply=args.apply,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] != "needs_attention" else 1


if __name__ == "__main__":
    raise SystemExit(main())
