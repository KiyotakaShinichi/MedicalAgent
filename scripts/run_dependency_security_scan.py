from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data/evals/ops/latest_dependency_security_scan.json"


def main() -> int:
    checks = []
    checks.append(_run_optional_tool("pip-audit", ["pip-audit", "-f", "json"], "python_dependencies"))
    npm_cmd = "npm.cmd" if shutil.which("npm.cmd") else "npm"
    checks.append(_run_optional_tool(npm_cmd, [npm_cmd, "audit", "--json", "--audit-level=high"], "frontend_dependencies", cwd=ROOT / "frontend-react"))
    high_count = sum(int(check.get("high_or_critical_count") or 0) for check in checks)
    payload = {
        "schema_version": "dependency_security_scan_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if high_count else "acceptable",
        "summary": {
            "tool_count": len(checks),
            "unavailable_tool_count": sum(1 for check in checks if check["status"] == "tool_unavailable"),
            "high_or_critical_count": high_count,
        },
        "checks": checks,
        "claim_boundary": "Dependency scanning is best-effort engineering hygiene, not a security certification.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _run_optional_tool(tool: str, command: list[str], label: str, cwd: Path | None = None) -> dict:
    if shutil.which(tool) is None:
        return {
            "label": label,
            "tool": tool,
            "status": "tool_unavailable",
            "high_or_critical_count": 0,
            "note": "Tool not installed locally; CI can enable this without changing project behavior.",
        }
    try:
        result = subprocess.run(command, cwd=cwd or ROOT, text=True, capture_output=True, timeout=180)
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "tool": tool, "status": "scan_error", "error": str(exc), "high_or_critical_count": 0}
    high_or_critical = _count_high_or_critical(result.stdout)
    return {
        "label": label,
        "tool": tool,
        "status": "needs_attention" if result.returncode not in {0} or high_or_critical else "acceptable",
        "exit_code": result.returncode,
        "high_or_critical_count": high_or_critical,
        "stderr_tail": result.stderr[-500:],
    }


def _count_high_or_critical(raw: str) -> int:
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        return 0
    if isinstance(payload.get("metadata"), dict) and isinstance(payload["metadata"].get("vulnerabilities"), dict):
        vuln = payload["metadata"]["vulnerabilities"]
        return int(vuln.get("high") or 0) + int(vuln.get("critical") or 0)
    if isinstance(payload.get("dependencies"), list):
        count = 0
        for dep in payload["dependencies"]:
            for vuln in dep.get("vulns", []) or []:
                if str(vuln.get("fix_versions") or vuln.get("severity") or "").lower() in {"high", "critical"}:
                    count += 1
        return count
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
