"""Append the current eval snapshot to ``Data/evals/history/eval_history.jsonl``
and refresh ``latest_eval_drift_report.json``.

Usage::

    python scripts/update_eval_history.py
    python scripts/update_eval_history.py --release-id 2026-05-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.eval_drift_tracker import EvalDriftTracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default=None)
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Refresh the drift report without appending a new history row.",
    )
    args = parser.parse_args()

    tracker = EvalDriftTracker()
    if not args.no_append:
        record = tracker.append_record(release_id=args.release_id)
        print(f"appended history row: commit={record['commit_hash']}  release_id={record['release_id']}")
        if record["missing_sources"]:
            print(f"  missing sources: {record['missing_sources']}")
    report = tracker.write_drift_report()
    print(f"wrote drift report: {tracker.report_path}")
    print(f"  records={report['n_records']}  regressions={report.get('regression_count', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
