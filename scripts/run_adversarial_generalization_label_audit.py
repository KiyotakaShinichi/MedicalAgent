from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_generalization_label_audit import run_label_audit


if __name__ == "__main__":
    result = run_label_audit()
    print(
        f"Audited {result['original_failure_count']} failures; "
        f"suspected label conflicts={result['machine_suspected_safe_negative_conflict_n']}; "
        "human adjudication remains incomplete."
    )
