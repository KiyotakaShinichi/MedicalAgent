from __future__ import annotations

import json

from scripts.run_dependency_security_scan import (
    _apply_frontend_risk_acceptance,
    _audit_counts,
    _audit_findings,
    _frontend_controls,
)


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


def test_extracts_npm_advisory_ids_for_review() -> None:
    payload = {
        "vulnerabilities": {
            "react-router": {
                "via": [
                    {
                        "url": "https://github.com/advisories/GHSA-2j2x-hqr9-3h42",
                        "title": "redirect",
                        "severity": "moderate",
                        "range": "<6.30.4",
                    }
                ]
            }
        }
    }
    assert _audit_findings(json.dumps(payload))[0]["advisory_id"] == "GHSA-2j2x-hqr9-3h42"


def test_frontend_risk_acceptance_requires_current_controls_and_known_ids() -> None:
    accepted = _apply_frontend_risk_acceptance(
        {
            "status": "needs_attention",
            "known_vulnerability_count": 1,
            "high_or_critical_count": 0,
            "findings": [{"advisory_id": "GHSA-2j2x-hqr9-3h42"}],
        }
    )
    assert accepted["status"] == "accepted_residual_risk"
    assert accepted["unaccepted_known_vulnerability_count"] == 0
    assert accepted["risk_acceptance"]["accepted"] is True
    assert _frontend_controls()["passed"] is True


def test_unknown_frontend_advisory_cannot_be_accepted() -> None:
    rejected = _apply_frontend_risk_acceptance(
        {
            "status": "needs_attention",
            "known_vulnerability_count": 1,
            "high_or_critical_count": 0,
            "findings": [{"advisory_id": "GHSA-unknown"}],
        }
    )
    assert rejected["status"] == "needs_attention"
    assert rejected["risk_acceptance"]["accepted"] is False
