from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
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
    checks[-1] = _apply_frontend_risk_acceptance(checks[-1])
    high_count = sum(int(check.get("high_or_critical_count") or 0) for check in checks)
    known_count = sum(int(check.get("known_vulnerability_count") or 0) for check in checks)
    unaccepted_count = sum(
        int(check.get("unaccepted_known_vulnerability_count") or 0)
        for check in checks
    )
    accepted_count = max(0, known_count - unaccepted_count)
    payload = {
        "schema_version": "dependency_security_scan_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if unaccepted_count or high_count else "acceptable",
        "summary": {
            "tool_count": len(checks),
            "unavailable_tool_count": sum(1 for check in checks if check["status"] == "tool_unavailable"),
            "high_or_critical_count": high_count,
            "known_vulnerability_count": known_count,
            "accepted_known_vulnerability_count": accepted_count,
            "unaccepted_known_vulnerability_count": unaccepted_count,
            "vulnerable_package_count": sum(
                int(check.get("vulnerable_package_count") or 0) for check in checks
            ),
        },
        "checks": checks,
        "claim_boundary": (
            "Dependency scanning is best-effort engineering hygiene, not a security "
            "certification. Python advisory feeds do not consistently expose severity, "
            "so known-vulnerability counts are reported separately from npm high/critical counts. "
            "Accepted residual risks remain visible and expire; they are not treated as vulnerability removal."
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
    findings = _audit_findings(result.stdout)
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
        "findings": findings,
        "unaccepted_known_vulnerability_count": counts["known_vulnerability_count"],
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


def _audit_findings(raw: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        return []
    findings: list[dict[str, Any]] = []
    if isinstance(payload.get("dependencies"), list):
        for dependency in payload["dependencies"]:
            for vulnerability in dependency.get("vulns", []) or []:
                findings.append(
                    {
                        "package": dependency.get("name"),
                        "version": dependency.get("version"),
                        "advisory_id": vulnerability.get("id"),
                        "fix_versions": vulnerability.get("fix_versions") or [],
                        "severity": None,
                    }
                )
        return findings
    for package, item in (payload.get("vulnerabilities") or {}).items():
        for vulnerability in item.get("via", []) or []:
            if not isinstance(vulnerability, dict):
                continue
            url = str(vulnerability.get("url") or "")
            advisory_id = url.rstrip("/").rsplit("/", 1)[-1] if url else str(vulnerability.get("source") or "")
            findings.append(
                {
                    "package": package,
                    "version_range": vulnerability.get("range"),
                    "advisory_id": advisory_id,
                    "title": vulnerability.get("title"),
                    "severity": vulnerability.get("severity"),
                }
            )
    return findings


def _apply_frontend_risk_acceptance(check: dict[str, Any]) -> dict[str, Any]:
    output = dict(check)
    config_path = ROOT / "config/dependency_risk_acceptance.json"
    if not config_path.exists():
        return output
    try:
        acceptance = json.loads(config_path.read_text(encoding="utf-8"))
        expires_on = date.fromisoformat(str(acceptance.get("expires_on")))
    except (ValueError, TypeError, json.JSONDecodeError):
        return output

    accepted_ids = {
        str(item.get("advisory_id"))
        for item in acceptance.get("accepted_advisories", [])
        if item.get("advisory_id")
    }
    observed_ids = {
        str(item.get("advisory_id"))
        for item in output.get("findings", [])
        if item.get("advisory_id")
    }
    controls = _frontend_controls()
    unaccepted_ids = sorted(observed_ids - accepted_ids)
    current = expires_on >= datetime.now(timezone.utc).date()
    can_accept = bool(
        observed_ids
        and not unaccepted_ids
        and int(output.get("high_or_critical_count") or 0) == 0
        and current
        and controls["passed"]
    )
    output["risk_acceptance"] = {
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "current": current,
        "expires_on": expires_on.isoformat(),
        "observed_advisory_ids": sorted(observed_ids),
        "unaccepted_advisory_ids": unaccepted_ids,
        "controls": controls,
        "accepted": can_accept,
        "claim_boundary": acceptance.get("claim_boundary"),
    }
    if can_accept:
        output["status"] = "accepted_residual_risk"
        output["unaccepted_known_vulnerability_count"] = 0
    return output


def _frontend_controls() -> dict[str, Any]:
    source_root = ROOT / "frontend-react/src"
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in source_root.rglob("*")
        if path.suffix in {".ts", ".tsx"}
    )
    forbidden_runtime_markers = (
        "createStaticRouter",
        "StaticRouterProvider",
        "HydratedRouter",
        "ServerRouter",
        "react-server-dom",
    )
    observed = [marker for marker in forbidden_runtime_markers if marker in source_text]
    package = json.loads((ROOT / "frontend-react/package.json").read_text(encoding="utf-8"))
    scripts = package.get("scripts") or {}
    vite_client = "vite" in str(scripts.get("dev") or "") and "vite build" in str(scripts.get("build") or "")
    return {
        "passed": bool(vite_client and not observed),
        "vite_client_spa": vite_client,
        "ssr_or_rsc_markers_found": observed,
        "navigation_target_review": "repository-defined routes only; enforced by test",
    }


if __name__ == "__main__":
    raise SystemExit(main())
