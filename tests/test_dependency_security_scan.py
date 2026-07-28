from __future__ import annotations

import json

from scripts.run_dependency_security_scan import _audit_counts


def test_counts_npm_severity_and_packages() -> None:
    payload = {
        "vulnerabilities": {"react-router": {}, "react-router-dom": {}},
        "metadata": {
            "vulnerabilities": {
                "info": 0,
                "low": 0,
                "moderate": 0,
                "high": 2,
                "critical": 1,
                "total": 3,
            }
        },
    }

    assert _audit_counts(json.dumps(payload)) == {
        "high_or_critical_count": 3,
        "known_vulnerability_count": 3,
        "vulnerable_package_count": 2,
    }


def test_counts_pip_audit_findings_without_inventing_severity() -> None:
    payload = {
        "dependencies": [
            {"name": "safe", "vulns": []},
            {"name": "one", "vulns": [{"id": "A"}, {"id": "B"}]},
            {"name": "two", "vulns": [{"id": "C"}]},
        ]
    }

    assert _audit_counts(json.dumps(payload)) == {
        "high_or_critical_count": 0,
        "known_vulnerability_count": 3,
        "vulnerable_package_count": 2,
    }


def test_invalid_output_is_zeroed_instead_of_crashing() -> None:
    assert _audit_counts("not-json") == {
        "high_or_critical_count": 0,
        "known_vulnerability_count": 0,
        "vulnerable_package_count": 0,
    }
