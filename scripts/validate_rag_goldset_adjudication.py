"""Validate the source-filter-drop adjudication packet.

Reads ``Data/evals/rag/source_filter_drop_adjudication_packet.json``,
runs the validator, refreshes
``Data/evals/rag/latest_goldset_adjudication_readiness.json``, and
returns a non-zero exit code if any validation issue would block the
packet from being trusted by a downstream consumer.

The script never mutates the goldset and never auto-applies any
adjudicated decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.rag_goldset_adjudication import (  # noqa: E402
    PACKET_OUTPUT_PATH,
    READINESS_OUTPUT_PATH,
    GOLDSET_PATH,
    packet_did_not_mutate_goldset,
    validate_packet,
    write_readiness_report,
)


def main() -> int:
    if not PACKET_OUTPUT_PATH.exists():
        print(f"packet not found: {PACKET_OUTPUT_PATH}")
        return 1

    packet = json.loads(PACKET_OUTPUT_PATH.read_text(encoding="utf-8"))
    issues = validate_packet(packet)
    mutation_ok = packet_did_not_mutate_goldset(packet, goldset_path=GOLDSET_PATH)
    write_readiness_report(READINESS_OUTPUT_PATH)

    print(f"packet:    {PACKET_OUTPUT_PATH}")
    print(f"items:     {len(packet.get('items') or [])}")
    print(f"goldset unmodified since packet build: {mutation_ok}")
    print(f"validation issues: {len(issues)}")
    for issue in issues[:10]:
        print(f"  - [{issue.case_id or '-'}] {issue.issue}")

    return 0 if (mutation_ok and not issues) else 1


if __name__ == "__main__":
    sys.exit(main())
