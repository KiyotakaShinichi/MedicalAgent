"""
Run the local quality gate for the MedicalAgent PoC.

This is intentionally account-free and local. It verifies frontend health,
backend regression tests, MLE readiness, RAG ablation, and optionally the
slower agent regression suite.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend-react"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MedicalAgent quality gates.")
    parser.add_argument(
        "--skip-slow-agent",
        action="store_true",
        help="Use the latest agent regression artifact instead of re-running the slow suite.",
    )
    parser.add_argument(
        "--skip-backend-tests",
        action="store_true",
        help="Skip pytest. Useful while iterating on frontend-only work.",
    )
    parser.add_argument(
        "--include-e2e",
        action="store_true",
        help="Run Playwright smoke tests. Requires browser binaries installed with npx playwright install chromium.",
    )
    args = parser.parse_args()

    checks: list[dict] = []

    npm = _bin("npm")

    checks.append(_run("frontend lint", [npm, "run", "lint"], cwd=FRONTEND, timeout=180))
    checks.append(_run("frontend build", [npm, "run", "build"], cwd=FRONTEND, timeout=180))

    if not args.skip_backend_tests:
        checks.append(_run(
            "backend tests",
            [sys.executable, "-m", "pytest", "tests/test_breast_monitoring.py", "tests/test_access_control.py", "-q"],
            cwd=ROOT,
            timeout=1800,
        ))

    checks.append(_run("MLE readiness", [sys.executable, "scripts/run_mle_checks.py"], cwd=ROOT, timeout=300))
    checks.append(_run(
        "RAG ablation",
        [sys.executable, "-c", "from backend.services.rag_ablation import run_rag_ablation; run_rag_ablation()"],
        cwd=ROOT,
        timeout=300,
    ))

    if args.include_e2e:
        checks.append(_run("Playwright smoke", [npm, "run", "test:e2e"], cwd=FRONTEND, timeout=600))

    if args.skip_slow_agent:
        checks.append(_check_agent_artifact())
    else:
        checks.append(_run(
            "agent regression",
            [
                sys.executable,
                "-c",
                (
                    "from backend.services.agent_regression_eval import run_agent_regression_suite; "
                    "r=run_agent_regression_suite(); "
                    "import json; print(json.dumps(r.get('summary', {}), indent=2))"
                ),
            ],
            cwd=ROOT,
            timeout=900,
        ))

    failed = [check for check in checks if not check["passed"]]
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "passed" if not failed else "failed",
        "failed_checks": [check["name"] for check in failed],
        "checks": checks,
    }

    out = ROOT / "Data" / "quality_gate" / "latest_quality_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "failed_checks": summary["failed_checks"],
        "output_path": str(out.relative_to(ROOT)),
    }, indent=2))
    return 0 if not failed else 1


def _run(name: str, cmd: list[str], cwd: Path, timeout: int) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "name": name,
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "duration_s": round(time.perf_counter() - started, 2),
            "command": " ".join(cmd),
            "output_tail": result.stdout[-5000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "passed": False,
            "returncode": "timeout",
            "duration_s": round(time.perf_counter() - started, 2),
            "command": " ".join(cmd),
            "output_tail": str(exc),
        }


def _bin(name: str) -> str:
    resolved = shutil.which(name) or shutil.which(f"{name}.cmd")
    return resolved or name


def _check_agent_artifact() -> dict:
    path = ROOT / "Data" / "agent_eval" / "latest_agent_regression.json"
    if not path.exists():
        return {
            "name": "agent regression artifact",
            "passed": False,
            "returncode": "missing",
            "duration_s": 0,
            "command": "read latest_agent_regression.json",
            "output_tail": "Data/agent_eval/latest_agent_regression.json not found",
        }

    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary") or {}
    passed = (
        summary.get("status") == "strong"
        and float(summary.get("pass_rate") or 0) >= 0.95
        and float(summary.get("attack_block_rate") or 0) >= 1.0
    )
    return {
        "name": "agent regression artifact",
        "passed": passed,
        "returncode": 0 if passed else 1,
        "duration_s": 0,
        "command": "read latest_agent_regression.json",
        "output_tail": json.dumps(summary, indent=2),
    }


if __name__ == "__main__":
    raise SystemExit(main())
