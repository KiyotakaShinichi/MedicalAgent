"""Build the source-filter-drop adjudication packet + readiness artifact.

Writes:
  Data/evals/rag/source_filter_drop_adjudication_packet.json
  Data/evals/rag/latest_goldset_adjudication_readiness.json

Both artifacts are draft state: every item's ``reviewer_decision`` is
``null`` and ``completed`` is ``false``.  No goldset case is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.rag_goldset_adjudication import (  # noqa: E402
    PACKET_OUTPUT_PATH,
    READINESS_OUTPUT_PATH,
    build_packet,
    write_packet,
    write_readiness_report,
)


def main() -> int:
    packet_path = write_packet(PACKET_OUTPUT_PATH)
    readiness_path = write_readiness_report(READINESS_OUTPUT_PATH)

    packet = build_packet()
    print(f"wrote packet: {packet_path}")
    print(f"  n_drop_cases: {packet['n_drop_cases']}")
    print(f"  status:       {packet['status']}")
    print(f"  completed:    {packet['completed']}")
    print(f"wrote readiness: {readiness_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
