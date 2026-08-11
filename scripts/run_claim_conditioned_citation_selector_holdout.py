from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.claim_conditioned_citation_selector_holdout import (  # noqa: E402
    build_selector_holdout_eval,
    freeze_selector_holdout,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("ONCOTRACK_FAST_MODE", "1")
    freeze = freeze_selector_holdout(overwrite=args.force) if args.freeze else None
    report = build_selector_holdout_eval()
    print(json.dumps({"freeze": freeze, "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
