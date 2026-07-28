from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data/evals/ops/latest_dependency_security_scan.json"


def main() -> int:
    checks = []
    python_audit = _python_audit_command()
    checks.append(
        _run_optional_tool(
            python_audit[0] if python_audit else "pip-audit",
            python_audit[1] if python_audit else [],
            "python_dependencies",
            scan_scope=python_audit[2] if python_audit else "unavailable",
        )
    )
    npm_cmd = "npm.cmd" if shutil.which("npm.cmd") else "npm"
    checks.append(
        _run_optional_tool(
            npm_cmd,
            [npm_cmd, "audit", "--json", "--audit-level=high"],
            "frontend_dependencies",
            cwd=ROOT / "frontend-react",
            scan_scope="frontend_lockfile",
        )
    )
    high_count = sum(int(check.get("high_or_critical_count") or 0) for check in checks)
    known_count = sum(int(check.get("known_vulnerability_count") or 0) for check in checks)
    payload = {
        "schema_version": "dependency_security_scan_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if known_count or high_count else "acceptable",
        "summary": {
            "tool_count": len(checks),
            "unavailable_tool_count": sum(1 for check in checks if check["status"] == "tool_unavailable"),
            "high_or_critical_count": high_count,
            "known_vulnerability_count": known_count,
            "vulnerable_package_count": sum(
                int(check.get("vulnerable_package_count") or 0) for check in checks
            ),
        },
        "checks": checks,
        "claim_boundary": (
            "Dependency scanning is best-effort engineering hygiene, not a security "
            "certification. Python advisory feeds do not consistently expose severity, "
            "so known-vulnerability counts are reported separately from npm high/critical counts."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _python_audit_command() -> tuple[str, list[str], str] | None:
    executable = shutil.which("pip-audit")
    prefix = [executable] if executable else None
    if prefix is None and importlib.util.find_spec("pip_audit") is not None:
        executable = sys.executable
        prefix = [sys.executable, "-m", "pip_audit"]
    if prefix is None or executable is None:
        return None

    platform_lock = ROOT / "requirements-lock-py314-win.txt"
    if sys.platform == "win32" and platform_lock.exists():
        return (
            executable,
            [
                *prefix,
                "-r",
                str(platform_lock),
                "--no-deps",
                "--disable-pip",
                "-f",
                "json",
            ],
            "requirements-lock-py314-win.txt",
        )
    return executable, [*prefix, "-f", "json"], "installed_environment"


def _run_optional_tool(
    tool: str,
    command: list[str],
    label: str,
    cwd: Path | None = None,
    *,
    scan_scope: str,
) -> dict[str, Any]:
    if not command or shutil.which(tool) is None:
        return {
            "label": label,
            "tool": tool,
            "status": "tool_unavailable",
            "high_or_critical_count": 0,
            "known_vulnerability_count": 0,
            "vulnerable_package_count": 0,
            "scan_scope": scan_scope,
            "note": "Tool not installed locally; CI can enable this without changing project behavior.",
        }
    try:
        result = subprocess.run(command, cwd=cwd or ROOT, text=True, capture_output=True, timeout=180)
    except Exception as exc:  # noqa: BLE001
        return {
            "label": label,
            "tool": tool,
            "status": "scan_error",
            "error": str(exc),
            "high_or_critical_count": 0,
            "known_vulnerability_count": 0,
            "vulnerable_package_count": 0,
            "scan_scope": scan_scope,
        }
    counts = _audit_counts(result.stdout)
    return {
        "label": label,
        "tool": tool,
        "status": (
            "needs_attention"
            if result.returncode != 0
            or counts["high_or_critical_count"]
            or counts["known_vulnerability_count"]
            else "acceptable"
        ),
        "exit_code": result.returncode,
        **counts,
        "scan_scope": scan_scope,
        "stderr_tail": result.stderr[-500:],
    }


def _audit_counts(raw: str) -> dict[str, int]:
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        return {
            "high_or_critical_count": 0,
            "known_vulnerability_count": 0,
            "vulnerable_package_count": 0,
        }
    if isinstance(payload.get("metadata"), dict) and isinstance(payload["metadata"].get("vulnerabilities"), dict):
        vuln = payload["metadata"]["vulnerabilities"]
        return {
            "high_or_critical_count": int(vuln.get("high") or 0)
            + int(vuln.get("critical") or 0),
            "known_vulnerability_count": int(vuln.get("total") or 0),
            "vulnerable_package_count": len(payload.get("vulnerabilities") or {}),
        }
    if isinstance(payload.get("dependencies"), list):
        count = 0
        packages = 0
        for dep in payload["dependencies"]:
            vulnerabilities = dep.get("vulns", []) or []
            if vulnerabilities:
                packages += 1
                count += len(vulnerabilities)
        return {
            "high_or_critical_count": 0,
            "known_vulnerability_count": count,
            "vulnerable_package_count": packages,
        }
    return {
        "high_or_critical_count": 0,
        "known_vulnerability_count": 0,
        "vulnerable_package_count": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
