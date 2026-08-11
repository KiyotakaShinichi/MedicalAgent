import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.tenant_isolation_security_eval import build_tenant_isolation_security_eval  # noqa: E402


if __name__ == "__main__":
    report = build_tenant_isolation_security_eval()
    print(
        f"tenant isolation: {report['status']} "
        f"({report['passed_count']}/{report['total_case_count']}, "
        f"leakage={report['cross_tenant_leakage_count']})"
    )
