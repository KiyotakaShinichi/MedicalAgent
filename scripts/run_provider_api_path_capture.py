"""Run the explicit normal-API provider usage capture probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.provider_api_path_capture import write_provider_api_path_capture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8017")
    parser.add_argument("--request-count", type=int, default=30)
    args = parser.parse_args()
    payload = write_provider_api_path_capture(
        execute=args.execute,
        base_url=args.base_url,
        request_count=args.request_count,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
