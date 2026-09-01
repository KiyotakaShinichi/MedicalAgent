"""Seed the deterministic, disposable synthetic buyer demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.buyer.demo import seed_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database")
    args = parser.parse_args()
    print(json.dumps(seed_demo(args.database), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
