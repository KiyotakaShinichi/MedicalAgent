"""Validate or build a deterministic NLCare buyer-candidate archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.buyer.contracts import ROOT, git_sha
from scripts.buyer.package import build_archive, selected_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dry_run or args.output is None:
        files = selected_files()
        result = {
            "status": "dry_run_passed",
            "candidate_type": "BUYER_CANDIDATE",
            "source_sha": git_sha(),
            "file_count": len(files) + 1,
            "archive_written": False,
        }
    else:
        output = args.output
        if not output.is_absolute():
            output = ROOT / output
        result = build_archive(output.resolve())
        result["status"] = "archive_built"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
