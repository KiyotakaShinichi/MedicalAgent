from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001d_overlap_audit import run_dep001d_overlap_audit


if __name__ == "__main__":
    result = run_dep001d_overlap_audit()
    print(f"DEP-001D overlap audit: {result['status']}")
    print(f"new cases: {result['new_case_n']}")
