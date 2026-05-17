from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.services.toxicity_model_metadata import build_toxicity_model_metadata


if __name__ == "__main__":
    metadata = build_toxicity_model_metadata()
    print(json.dumps({
        "status": metadata["status"],
        "model_exists": metadata["model_exists"],
        "shortcut_status": metadata["known_shortcut_risk"]["status"],
    }, indent=2))
