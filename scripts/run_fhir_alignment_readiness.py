from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.fhir_like_mapper import build_fhir_alignment_readiness


if __name__ == "__main__":
    payload = build_fhir_alignment_readiness()
    print({
        "status": payload["status"],
        "mapping_coverage": payload["mapping_coverage"],
        "unmapped_field_count": payload["unmapped_field_count"],
    })
