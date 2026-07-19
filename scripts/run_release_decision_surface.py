import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.release_decision_surface import build_release_decision_surface


if __name__ == "__main__":
    result = build_release_decision_surface()
    print(f"decision={result['engineering_release_decision']} blockers={result['hard_blocker_count']} warnings={result['warning_count']}")
