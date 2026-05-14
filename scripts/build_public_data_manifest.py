from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.public_data_manifest import DEFAULT_OUTPUT_PATH, build_public_data_manifest


def main() -> int:
    manifest = build_public_data_manifest(output_path=DEFAULT_OUTPUT_PATH)
    print(json.dumps({
        "status": manifest["status"],
        "sources": len(manifest["sources"]),
        "feature_needs": len(manifest["feature_feasibility"]),
        "manifest_hash": manifest["manifest_hash"],
        "output_path": DEFAULT_OUTPUT_PATH,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
