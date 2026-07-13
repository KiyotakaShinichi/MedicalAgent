from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.automation_control_plane import build_automation_control_plane  # noqa: E402


if __name__ == "__main__":
    report = build_automation_control_plane()
    print(
        json.dumps(
            {
                "status": report["status"],
                "event_candidate_count": report["event_candidate_count"],
                "accepted_event_count": report["accepted_event_count"],
                "commands_executed": report["commands_executed"],
                "webhooks_sent": report["webhooks_sent"],
            },
            indent=2,
        )
    )
