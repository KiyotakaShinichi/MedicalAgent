from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.azure_search_index_admin import build_azure_search_index_readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Azure AI Search index contract or apply it with explicit network gates."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create or update the configured shadow index. Disabled by default.",
    )
    args = parser.parse_args()
    report = build_azure_search_index_readiness(apply=args.apply)
    print(json.dumps(report, indent=2))
    return 0 if report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
