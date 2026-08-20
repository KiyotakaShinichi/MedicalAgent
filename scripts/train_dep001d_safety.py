from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001d_training import train_dep001d_models


if __name__ == "__main__":
    result = train_dep001d_models()
    print(f"DEP-001D training status: {result['status']}")
    print(result["input_safety"]["validation"]["policy"])
    print(result["output_actionability"]["validation"])
