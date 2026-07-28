"""Executable, evidence-linked improvement program for NLCare.

The program does not assign clinical maturity scores. It translates the
canonical release surface into domain-specific engineering acceptance criteria
and separates locally controllable work from external blockers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SURFACE = Path("Data/evals/governance/latest_release_decision_surface.json")
DEFAULT_OUTPUT = Path("Data/evals/governance/latest_constraint_aware_improvement_program.json")
DEFAULT_DOC = Path("docs/constraint_aware_improvement_program.md")

DOMAIN_PLANS: dict[str, dict[str, Any]] = {
    "aie": {
        "objective": "Prefer the simplest governed retrieval and routing policy that survives frozen evaluation.",
        "implemented_controls": [
            "dense and sparse baselines with RRF comparison",
            "source-tier and allowed-use filtering",
            "claim and post-generation validation",
            "frozen adversarial and metamorphic evaluations",
        ],
        "acceptance_criteria": [
            "critical safety regression remains passing",
            "source-tier correctness remains 1.0 on patient-facing routes",
            "raw retrieval superiority is claimed only when frozen comparison proves it",
            "stage latency and marginal quality are reported together",
        ],
        "next_internal_action": "Increase independent route sample counts and preserve BM25 as the explicit simple baseline.",
        "external_blocker": "No-read RAG holdout and external-authored adversarial cases remain incomplete.",
    },
    "mle": {
        "objective": "Make synthetic ML statistically defensible as engineering evidence while preventing clinical interpretation.",
        "implemented_controls": [
            "patient-level temporal splitting and leakage audits",
            "simple-baseline and multi-seed stress comparisons",
            "calibration, abstention, subgroup, missingness, and shortcut audits",
            "promotion policy fixed to HOLD or review-only use",
        ],
        "acceptance_criteria": [
            "patient overlap and temporal leakage remain zero",
            "simple baselines are always reported beside nonlinear models",
            "selective risk versus coverage is reported for every prediction head",
            "synthetic confidence is never described as patient outcome probability",
        ],
        "next_internal_action": "Keep saturated metrics out of headline surfaces and expand ambiguous-label sensitivity analyses.",
        "external_blocker": "Simulator-built labels cannot establish real-patient transfer or clinical calibration.",
    },
    "swe": {
        "objective": "Make the system reproducible, reviewable, secure by default, and boring to operate in controlled demos.",
        "implemented_controls": [
            "modular FastAPI and typed React surfaces",
            "migrations, role isolation, request IDs, tests, and release gates",
            "freshness-aware canonical release decision surface",
            "evidence-backed service health aggregation",
        ],
        "acceptance_criteria": [
            "full ship workflow passes from a clean checkout",
            "hard blockers cannot be diluted by informational artifacts",
            "canonical evidence is refreshed after its dependencies",
            "dependency and container scans expose unavailable tools as gaps",
        ],
        "next_internal_action": "Reduce stale artifact volume and add a Linux reproducibility lane with SBOM and container scan evidence.",
        "external_blocker": "No production traffic, penetration test, SOC 2, HITRUST, or healthcare security review exists.",
    },
    "data_engineering": {
        "objective": "Operate a lineage-complete, idempotent non-patient knowledge pipeline with explicit recovery behavior.",
        "implemented_controls": [
            "content-addressed bronze and governed silver/gold layers",
            "contract checks, quarantine, fingerprints, and lineage",
            "idempotency, migration, tombstone, backfill, and fallback drills",
        ],
        "acceptance_criteria": [
            "all gold records retain governance metadata",
            "duplicate identifiers and hard data-quality failures remain zero",
            "replay produces the same KB fingerprint",
            "no patient data or raw chat enters managed-vector payloads",
        ],
        "next_internal_action": "Add partition-scale benchmarks before introducing Databricks or Data Factory.",
        "external_blocker": "No cloud lakehouse execution or real-patient data-quality evidence exists.",
    },
    "infrastructure": {
        "objective": "Keep infrastructure reproducible, private by default, cost-bounded, and disposable.",
        "implemented_controls": [
            "compile-checked Bicep with cost-bearing resources default off",
            "managed identity, RBAC, Key Vault references, private endpoints, budgets, and alerts",
            "non-patient Azure Search shadow contracts",
        ],
        "acceptance_criteria": [
            "Bicep compilation has zero diagnostics",
            "public network and cost-bearing resources remain opt-in",
            "no secrets are committed",
            "deployment evidence distinguishes compile, what-if, deploy, restore, and teardown",
        ],
        "next_internal_action": "Run an authenticated what-if in a disposable non-patient development subscription.",
        "external_blocker": "No cloud deployment, private-connectivity exercise, restore drill, or cost measurement exists.",
    },
    "medical": {
        "objective": "Preserve clinician authority and make uncertainty, missingness, and non-diagnostic boundaries understandable.",
        "implemented_controls": [
            "deterministic medical claim boundaries",
            "minimum evidence and abstention rules",
            "urgent, distress, genetics/VUS, tumor-marker, and supplement routing boundaries",
            "unreviewed role-specific review packets",
        ],
        "acceptance_criteria": [
            "diagnosis, prognosis, treatment, dosage, and genetic/tumor-marker conclusions remain blocked",
            "patient explanations include meaning, calculation, missingness, limitation, and safe review action",
            "delivery receipts are never described as clinician acknowledgement",
            "all review packets remain labelled unreviewed until actually completed",
        ],
        "next_internal_action": "Run non-clinical comprehension and overtrust testing with synthetic scenarios.",
        "external_blocker": "No clinician, nurse, genetic counselor, pharmacist, or patient review has been completed.",
    },
    "automation": {
        "objective": "Automate redacted engineering workflows with durable delivery semantics and explicit human acknowledgement boundaries.",
        "implemented_controls": [
            "database leases, heartbeats, retries, dead letters, and idempotency",
            "signed redacted webhooks and signed receipt validation",
            "fault-injection scenarios and dry-run n8n templates",
            "service-health metrics for automation control and live-delivery gaps",
        ],
        "acceptance_criteria": [
            "duplicate enqueue and replay remain idempotent",
            "PHI and raw chat are rejected from external payloads",
            "delivery receipt and human acknowledgement remain separate fields",
            "patient-facing or clinical action automation stays disabled",
        ],
        "next_internal_action": "Exercise one synthetic test-recipient channel and record receipt latency without claiming clinical coverage.",
        "external_blocker": "No provider reliability, operator rota, emergency coverage, or clinician acknowledgement evidence exists.",
    },
    "deployment": {
        "objective": "Support controlled non-clinical deployment profiles without implying healthcare production readiness.",
        "implemented_controls": [
            "Docker, readiness probes, migrations, CORS and security-header checks",
            "RS256 OIDC bearer validation",
            "S256 PKCE transaction creation, bounded storage, expiry, and replay rejection",
            "strict profile validation and local recovery drills",
        ],
        "acceptance_criteria": [
            "demo auth and localhost CORS are disabled in strict profiles",
            "OIDC issuer, audience, HTTPS JWKS, and exact role mapping fail closed",
            "PKCE callback transactions are single-use and verifier material is not disclosed",
            "production healthcare readiness remains false",
        ],
        "next_internal_action": "Add a shared encrypted PKCE transaction store, reviewed token exchange, and provider logout before any live browser login claim.",
        "external_blocker": "No live identity provider, managed database recovery, production SLO history, or healthcare deployment review exists.",
    },
}


def build_improvement_program(
    *,
    surface_path: str | Path = DEFAULT_SURFACE,
    output_path: str | Path = DEFAULT_OUTPUT,
    doc_path: str | Path = DEFAULT_DOC,
) -> dict[str, Any]:
    surface = _read_json(_resolve(surface_path))
    states = {
        str(row.get("domain")): str(row.get("state"))
        for row in surface.get("domains") or []
        if row.get("domain")
    }
    checks_by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in surface.get("checks") or []:
        checks_by_domain.setdefault(str(row.get("domain")), []).append(row)

    domains = []
    priorities = []
    for domain, plan in DOMAIN_PLANS.items():
        checks = checks_by_domain.get(domain, [])
        state = states.get(domain, "missing_canonical_evidence")
        attention = [
            row.get("id")
            for row in checks
            if row.get("decision") in {"attention", "missing", "invalid"}
            or row.get("evidence_state") == "stale"
        ]
        domains.append({
            "domain": domain,
            "evidence_state": state,
            "canonical_check_count": len(checks),
            "attention_check_ids": attention,
            **plan,
        })
        priorities.append({
            "priority": "P0" if any(
                row.get("tier") == "hard_blocker" and row.get("decision") != "pass"
                for row in checks
            ) else "P1" if attention else "P2",
            "domain": domain,
            "action": plan["next_internal_action"],
            "acceptance_criteria": plan["acceptance_criteria"],
            "blocked_by_external_evidence": plan["external_blocker"],
        })

    priorities.sort(key=lambda row: (row["priority"], row["domain"]))
    payload = {
        "schema_version": "constraint_aware_improvement_program_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if surface.get("hard_blocker_count") else "needs_attention",
        "engineering_release_decision": surface.get("engineering_release_decision"),
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "real_patient_data_used": False,
        "domain_count": len(domains),
        "domains": domains,
        "ranked_priorities": priorities,
        "implemented_in_this_pass": [
            "freshness-aware canonical evidence surface across all engineering domains",
            "evidence-backed operational health metrics instead of null placeholders",
            "single-use, expiring PKCE transaction lifecycle with replay rejection",
            "ship ordering that refreshes dependent artifacts before final release summaries",
        ],
        "things_internal_engineering_cannot_prove": [
            "clinical validation or patient benefit",
            "real-world safety or medical correctness",
            "generalisation from synthetic ML to real patients",
            "managed-cloud reliability without deployment evidence",
            "human acknowledgement from an automation delivery receipt",
            "production healthcare readiness",
        ],
        "claim_boundary": (
            "Constraint-aware engineering plan only. Completion means stronger internal "
            "engineering evidence, not clinical validation, clinician approval, real-world "
            "patient safety, or production healthcare readiness."
        ),
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Constraint-Aware Improvement Program",
        "",
        payload["claim_boundary"],
        "",
        "## Canonical Priorities",
        "",
        "| Priority | Domain | Evidence state | Next internal action |",
        "| --- | --- | --- | --- |",
    ]
    states = {row["domain"]: row["evidence_state"] for row in payload["domains"]}
    for row in payload["ranked_priorities"]:
        lines.append(
            f"| {row['priority']} | {row['domain']} | {states[row['domain']]} | {row['action']} |"
        )
    lines.extend(["", "## Domain Acceptance Criteria", ""])
    for domain in payload["domains"]:
        lines.append(f"### {domain['domain'].replace('_', ' ').title()}")
        lines.append("")
        lines.append(domain["objective"])
        lines.append("")
        for criterion in domain["acceptance_criteria"]:
            lines.append(f"- {criterion}")
        lines.append(f"- External blocker: {domain['external_blocker']}")
        lines.append("")
    lines.extend(["## What This Cannot Prove", ""])
    lines.extend(f"- {item}" for item in payload["things_internal_engineering_cannot_prove"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


__all__ = ["DOMAIN_PLANS", "build_improvement_program"]
