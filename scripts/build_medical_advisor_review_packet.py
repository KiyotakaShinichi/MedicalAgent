from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.services.medical_advisor_packet import build_medical_advisor_review_packet


if __name__ == "__main__":
    packet = build_medical_advisor_review_packet()
    print(json.dumps({"status": packet["status"], "interaction_rules": packet["interaction_rule_count"]}, indent=2))
