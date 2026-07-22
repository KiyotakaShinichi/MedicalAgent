from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.duke_tcia_tabular_bridge import run_duke_tcia_external_stress


if __name__ == "__main__":
    result = run_duke_tcia_external_stress()
    print(
        f"status={result['status']} rows={result['canonical_export']['row_count']} "
        f"clinical_validation={result['clinical_validation']}"
    )
