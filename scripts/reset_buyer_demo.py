"""Safely remove only a marked synthetic buyer-demo database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.buyer.demo import reset_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database")
    args = parser.parse_args()
    reset_demo(args.database)
    print("buyer demo reset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
