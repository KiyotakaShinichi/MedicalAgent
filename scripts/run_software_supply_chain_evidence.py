from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.software_supply_chain_evidence import (  # noqa: E402
    build_software_supply_chain_evidence,
)


if __name__ == "__main__":
    result = build_software_supply_chain_evidence()
    print(json.dumps({
        "status": result["status"],
        "components": result["sbom"]["component_count"],
        "secret_findings": result["secret_scan"]["finding_count"],
        "container_scan_executed": result["container_scan"]["executed"],
    }, indent=2))
