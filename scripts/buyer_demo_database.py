"""Backup or restore the disposable synthetic buyer-demo SQLite database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.buyer.demo import backup_demo, restore_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backup = subparsers.add_parser("backup")
    backup.add_argument("--database")
    backup.add_argument("--output", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--database")
    restore.add_argument("--backup", required=True)
    args = parser.parse_args()
    if args.command == "backup":
        result = {"backup": str(backup_demo(args.output, args.database))}
    else:
        result = restore_demo(args.backup, args.database)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
