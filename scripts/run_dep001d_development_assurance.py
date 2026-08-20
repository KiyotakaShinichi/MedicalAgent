from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001d_development_assurance import build_dep001d_development_assurance


if __name__ == "__main__":
    report = build_dep001d_development_assurance()
    print(f"DEP-001D development assurance: {report['status']}")
    print(report["integrated_metrics"])
