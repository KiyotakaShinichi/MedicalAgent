"""Deployment readiness, SaaS foundation, and supply-chain/security scans.

Extracted from ``scripts/ship.py`` as part of splitting a 477-line
``_build_steps``. Step definitions are relocated verbatim: the command, working
directory, environment, and timeout of every step are unchanged, and the order
within and across these modules reproduces the original list exactly.
"""

from __future__ import annotations

import sys

from scripts.ship_steps.common import Step

__all__ = ["readiness_and_security_steps"]


def readiness_and_security_steps() -> list[Step]:
    return [
                Step(
                    name="Patient enrichment background eval",
                    command=[sys.executable, "scripts/run_patient_enrichment_background_eval.py"],
                ),
                Step(
                    name="High-risk conversation alert eval",
                    command=[sys.executable, "scripts/run_high_risk_conversation_alert_eval.py"],
                ),
                Step(
                    name="ML logic/safety alignment audit",
                    command=[sys.executable, "scripts/run_ml_logic_safety_alignment.py"],
                ),
                Step(
                    name="Synthetic statistical sensitivity audit",
                    command=[
                        sys.executable,
                        "scripts/run_synthetic_prediction_statistical_audit.py",
                    ],
                ),
                Step(
                    name="Patient XAI implementation-readability audit",
                    command=[
                        sys.executable,
                        "scripts/run_patient_xai_readability_dossier.py",
                    ],
                ),
                Step(
                    name="Restricted synthetic SaaS foundation readiness",
                    command=[sys.executable, "scripts/run_saas_foundation_readiness.py"],
                ),
                Step(
                    name="OIDC browser PKCE readiness",
                    command=[sys.executable, "scripts/run_oidc_browser_pkce_readiness.py"],
                ),
                Step(
                    name="Deployment profile and fail-closed matrix",
                    command=[sys.executable, "scripts/validate_deployment_env.py"],
                ),
                Step(
                    name="Dependency security scan",
                    command=[sys.executable, "scripts/run_dependency_security_scan.py"],
                ),
                Step(
                    name="Container security evidence",
                    command=[sys.executable, "scripts/run_container_security_scan.py"],
                ),
                Step(
                    name="Lock-derived SBOM and repository secret scan",
                    command=[sys.executable, "scripts/run_software_supply_chain_evidence.py"],
                ),
    ]
