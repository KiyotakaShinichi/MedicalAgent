"""Run the source-alias coverage diagnostic.

Writes ``Data/evals/rag/latest_source_alias_coverage.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.rag_source_alias_coverage import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_alias_coverage_report,
    write_alias_coverage_report,
)


def main() -> int:
    out = write_alias_coverage_report(DEFAULT_OUTPUT_PATH)
    report = build_alias_coverage_report()
    print(f"wrote: {out}")
    print(
        f"  goldset cases={report['n_goldset_cases']}  "
        f"kb chunks={report['n_kb_chunks']}  "
        f"alias keys={report['n_alias_keys_demanded_by_goldset']}  "
        f"uncovered={report['n_alias_keys_uncovered']}  "
        f"proposed={report['n_proposed_additions_total']}"
    )
    for entry in report["per_alias"]:
        if entry["proposed_additions_by_content_match"]:
            n = len(entry["proposed_additions_by_content_match"])
            print(f"  {entry['alias_key']:36s}  +{n} proposed (demand={entry['goldset_demand_count']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
