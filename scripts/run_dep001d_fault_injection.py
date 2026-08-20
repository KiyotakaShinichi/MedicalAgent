from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001d_fault_injection import run_dep001d_fault_injection


if __name__ == "__main__":
    report = run_dep001d_fault_injection()
    print(f"DEP-001D fault injection: {report['status']} ({report['passed_n']}/{report['total_n']})")
