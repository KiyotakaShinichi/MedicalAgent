import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.complete_synthetic_xai import generate_complete_synthetic_xai


if __name__ == "__main__":
    result = generate_complete_synthetic_xai()
    print(
        f"patients_explained={result['patients_explained']} "
        f"method={result['method']} shap_available={result['shap_available']}"
    )
