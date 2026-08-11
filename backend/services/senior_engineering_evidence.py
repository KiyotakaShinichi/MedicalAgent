"""Concentrated, falsifiable engineering-evidence dossier.

The dossier does not award a senior title. It binds high-signal internal
evidence to source, test, artifact, and command provenance so a reviewer can
reproduce or falsify the claims without reading hundreds of artifacts.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/governance/latest_senior_engineering_evidence.json"
)
DEFAULT_DOC_PATH = Path("docs/senior_engineering_evidence.md")

EVIDENCE_TRIANGLES: dict[str, dict[str, str]] = {
    "agent_execution": {
        "source": "backend/services/agent_execution_policy.py",
        "test": "tests/test_agent_execution_policy.py",
        "artifact": "Data/evals/agentic_tool_use/latest_agent_execution_policy_eval.json",
    },
    "cross_domain_assurance": {
        "source": "backend/services/cross_domain_assurance_eval.py",
        "test": "tests/test_cross_domain_assurance_eval.py",
        "artifact": "Data/evals/governance/latest_cross_domain_assurance_eval.json",
    },
    "automation": {
        "source": "backend/services/automation_fault_injection_eval.py",
        "test": "tests/test_automation_fault_injection.py",
        "artifact": "Data/evals/ops/latest_automation_fault_injection.json",
    },
    "rag_degradation": {
        "source": "backend/services/rag_degradation_resilience_eval.py",
        "test": "tests/test_rag_degradation_resilience_eval.py",
        "artifact": "Data/evals/rag/latest_rag_degradation_resilience.json",
    },
    "synthetic_ml_statistics": {
        "source": "backend/services/synthetic_prediction_statistical_audit.py",
        "test": "tests/test_synthetic_prediction_statistical_audit.py",
        "artifact": "Data/evals/models/latest_synthetic_prediction_statistical_audit.json",
    },
    "xai_implementation": {
        "source": "backend/services/patient_xai_readability_dossier.py",
        "test": "tests/test_patient_xai_readability_dossier.py",
        "artifact": "Data/evals/governance/latest_patient_xai_readability_dossier.json",
    },
    "data_reliability": {
        "source": "backend/services/data_platform_reliability_eval.py",
        "test": "tests/test_data_platform_reliability_eval.py",
        "artifact": "Data/evals/ops/latest_data_platform_reliability_eval.json",
    },
    "infrastructure": {
        "source": "backend/services/cloud_infrastructure_readiness.py",
        "test": "tests/test_cloud_infrastructure_readiness.py",
        "artifact": "Data/evals/ops/latest_cloud_infrastructure_readiness.json",
    },
}

SUPPORTING_ARTIFACTS = (
    "Data/evals/ops/latest_ship_run.json",
    "Data/evals/governance/latest_release_decision_surface.json",
    "Data/evals/governance/latest_focused_release_summary.json",
    "Data/evals/rag/latest_rag_baseline_comparison.json",
    "Data/evals/safety/latest_adversarial_holdout_v6_baseline.json",
    "Data/evals/ops/latest_dependency_security_scan.json",
    "Data/evals/ops/latest_software_supply_chain_evidence.json",
    "Data/evals/ops/latest_deployment_recovery_drill.json",
    "Data/evals/ops/latest_trace_envelope_v2_eval.json",
)


def _ship_manifest_passed_with_timeouts(ship: dict[str, Any]) -> bool:
    steps = ship.get("steps") or []
    return bool(
        ship.get("status") == "passed"
        and steps
        and all(
            row.get("status") in {"passed", "cached_pass"}
            and int(row.get("timeout_seconds") or 0) >= 30
            for row in steps
        )
    )


def build_senior_engineering_evidence(
    *,
    root_dir: str | Path = ROOT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    """Build a reviewer-first evidence dossier with explicit downgrade rules."""

    root = Path(root_dir)
    triangles = [
        _triangle(root, domain, paths)
        for domain, paths in EVIDENCE_TRIANGLES.items()
    ]
    supporting = [_artifact_record(root, path) for path in SUPPORTING_ARTIFACTS]
    ship = _read_json(root / "Data/evals/ops/latest_ship_run.json")
    cross = _read_json(
        root / "Data/evals/governance/latest_cross_domain_assurance_eval.json"
    )
    release = _read_json(
        root / "Data/evals/governance/latest_release_decision_surface.json"
    )
    rag = _read_json(root / "Data/evals/rag/latest_rag_baseline_comparison.json")
    adversarial = _read_json(
        root / "Data/evals/safety/latest_adversarial_holdout_v6_baseline.json"
    )
    ml = _read_json(
        root / "Data/evals/models/latest_synthetic_prediction_statistical_audit.json"
    )
    cloud = _read_json(
        root / "Data/evals/ops/latest_cloud_infrastructure_readiness.json"
    )

    ship_source = (root / "scripts/ship.py").read_text(
        encoding="utf-8", errors="replace"
    )
    architecture_fitness = [
        _check(
            "all_evidence_triangles_complete",
            all(row["complete"] for row in triangles),
            "Every selected claim has source, focused test, and generated artifact.",
        ),
        _check(
            "cross_domain_assurance_green",
            cross.get("status") == "strong_internal_assurance"
            and cross.get("failed_count") == 0,
            "Composed offline control boundaries agree.",
        ),
        _check(
            "ship_manifest_passed_with_timeouts",
            _ship_manifest_passed_with_timeouts(ship),
            "The recorded release run completed with bounded step timeouts.",
            mandatory=False,
        ),
        _check(
            "new_evidence_is_wired_into_ship",
            "run_cross_domain_assurance_eval.py" in ship_source
            and "run_senior_engineering_evidence.py" in ship_source
            and "test_cross_domain_assurance_eval.py" in ship_source
            and "test_senior_engineering_evidence.py" in ship_source,
            "The evidence regenerates and its contracts execute during ship.",
        ),
        _check(
            "negative_results_are_not_promoted",
            rag.get("summary", {}).get("improvement_proven_vs_bm25") is False
            and ml.get("promotion_decision") == "hold_synthetic_only"
            and adversarial.get("status") == "needs_attention",
            "Known RAG, adversarial, and synthetic-ML limitations remain binding.",
        ),
        _check(
            "cloud_evidence_is_not_deployment_evidence",
            cloud.get("cloud_deployment_completed") is False
            and cloud.get("healthcare_production_ready") is False,
            "Compiled infrastructure is not represented as a live deployment.",
        ),
        _check(
            "release_surface_never_false_clean",
            release.get("engineering_release_decision") in {
                "BLOCK",
                "PROCEED_WITH_WARNINGS",
            }
            and (
                int(release.get("hard_blocker_count") or 0) >= 1
                or int(release.get("warning_count") or 0) >= 1
            ),
            "The canonical release decision preserves blockers or warnings instead of silently becoming clean.",
        ),
    ]
    mandatory_checks = [
        row for row in architecture_fitness if row.get("mandatory") is True
    ]
    mandatory_passed = all(row["passed"] for row in mandatory_checks)
    prior_ship_passed = next(
        row["passed"]
        for row in architecture_fitness
        if row["check_id"] == "ship_manifest_passed_with_timeouts"
    )
    hashes_complete = all(row["sha256"] for row in supporting if row["exists"])
    reproducibility = {
        "ship_status": ship.get("status"),
        "ship_completed_step_count": ship.get("completed_step_count"),
        "recorded_commands": [
            {
                "name": row.get("name"),
                "command": row.get("command"),
                "cwd": row.get("cwd"),
                "duration_seconds": row.get("duration_seconds"),
                "timeout_seconds": row.get("timeout_seconds"),
            }
            for row in ship.get("steps") or []
        ],
        "source_artifact_hashes_complete": bool(hashes_complete),
        "toolchain": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "repository_state": _repository_state(root),
    }
    negative_results = {
        "rag_improvement_proven_vs_bm25": rag.get("summary", {}).get(
            "improvement_proven_vs_bm25"
        ),
        "rag_full_stack_recall_at_10": rag.get("summary", {}).get(
            "full_stack_recall_at_10"
        ),
        "rag_bm25_recall_at_10": rag.get("summary", {}).get(
            "bm25_recall_at_10"
        ),
        "frozen_adversarial_v6_pass_rate": adversarial.get("pass_rate"),
        "frozen_adversarial_v6_unsafe_leakage_rate": adversarial.get(
            "unsafe_leakage_rate"
        ),
        "synthetic_ml_promotion_decision": ml.get("promotion_decision"),
        "cloud_deployment_completed": cloud.get("cloud_deployment_completed"),
        "independent_review_completed": False,
    }
    falsification_criteria = [
        {
            "id": "cross_domain_control_failure",
            "effect": "downgrade_to_needs_attention",
            "condition": "Any composed assurance scenario fails.",
        },
        {
            "id": "unsafe_patient_route_leakage",
            "effect": "block_engineering_release",
            "condition": "Critical patient-facing unsafe leakage is non-zero.",
        },
        {
            "id": "unbound_or_replayed_write",
            "effect": "block_engineering_release",
            "condition": "A write executes without a fresh patient- and payload-bound confirmation.",
        },
        {
            "id": "promotion_boundary_erased",
            "effect": "block_engineering_release",
            "condition": "Synthetic ML or unproven retrieval is promoted as clinical or production evidence.",
        },
        {
            "id": "negative_result_hidden",
            "effect": "downgrade_to_needs_attention",
            "condition": "Frozen adversarial, RAG, dependency, or external-review warnings disappear without new evidence.",
        },
        {
            "id": "artifact_source_mismatch",
            "effect": "downgrade_to_needs_attention",
            "condition": "A selected claim loses its source-test-artifact triangle or provenance hash.",
        },
    ]
    if not mandatory_passed or not hashes_complete:
        status = "needs_attention"
    elif prior_ship_passed:
        status = "strong_internal_engineering_evidence"
    else:
        status = "provisional_pending_current_ship"
    report = {
        "schema_version": "nlcare_senior_engineering_evidence_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "evidence_maturity": (
            "advanced_internal_candidate"
            if status == "strong_internal_engineering_evidence"
            else "provisional_internal_candidate"
            if status == "provisional_pending_current_ship"
            else "incomplete_internal_evidence"
        ),
        "senior_title_awarded_by_artifact": False,
        "independent_reproduction_completed": False,
        "external_reviewer_completed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "architecture_fitness": architecture_fitness,
        "architecture_fitness_pass_rate": round(
            sum(int(row["passed"]) for row in mandatory_checks)
            / len(mandatory_checks),
            6,
        ),
        "all_architecture_checks_pass_rate": round(
            sum(int(row["passed"]) for row in architecture_fitness)
            / len(architecture_fitness),
            6,
        ),
        "evidence_triangles": triangles,
        "supporting_artifact_provenance": supporting,
        "reproducibility": reproducibility,
        "negative_results": negative_results,
        "falsification_criteria": falsification_criteria,
        "remaining_senior_evidence_gaps": [
            "Independent engineer reproduction from a clean clone is not completed.",
            "External no-read safety and RAG evaluation is not completed.",
            "No clinician or genetic-counselor review is completed.",
            "No live managed-cloud load, failover, restore, cost, or delivery evidence exists.",
            "No real patient data, clinician-reviewed labels, IRB, or clinical validation exists.",
        ],
        "claim_boundary": (
            "This dossier demonstrates advanced internal engineering discipline under "
            "synthetic and offline constraints. It does not award professional seniority, "
            "prove independent reproducibility, establish clinical validation, or show "
            "production healthcare readiness."
        ),
    }
    output = _resolve(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_doc(_resolve(root, doc_path), report)
    return report


def _triangle(
    root: Path,
    domain: str,
    paths: dict[str, str],
) -> dict[str, Any]:
    records = {
        kind: _file_record(root, path)
        for kind, path in paths.items()
    }
    artifact = _read_json(root / paths["artifact"])
    return {
        "domain": domain,
        **records,
        "artifact_status": artifact.get("status"),
        "artifact_clinical_validation": artifact.get("clinical_validation"),
        "complete": all(record["exists"] for record in records.values())
        and artifact.get("clinical_validation") is False,
    }


def _artifact_record(root: Path, relative_path: str) -> dict[str, Any]:
    record = _file_record(root, relative_path)
    payload = _read_json(root / relative_path)
    return {
        **record,
        "status": payload.get("status"),
        "clinical_validation": payload.get("clinical_validation"),
        "healthcare_production_ready": payload.get(
            "healthcare_production_ready",
            payload.get("production_healthcare_ready"),
        ),
    }


def _file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path.replace("\\", "/"),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _repository_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        return {
            "git_available": True,
            "commit": commit,
            "worktree_dirty": bool(dirty),
            "dirty_path_count": len(dirty),
            "note": (
                "Hashes in this dossier bind the evaluated working tree even when "
                "it is not a clean commit."
            ),
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "git_available": False,
            "commit": None,
            "worktree_dirty": None,
            "dirty_path_count": None,
            "note": "Git state was unavailable; per-file hashes remain authoritative.",
        }


def _check(
    check_id: str,
    passed: bool,
    description: str,
    *,
    mandatory: bool = True,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "mandatory": bool(mandatory),
        "description": description,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _write_doc(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = "\n".join(
        f"- `{'PASS' if row['passed'] else 'FAIL'}` "
        f"`{'mandatory' if row['mandatory'] else 'observational'}` "
        f"{row['check_id']}: "
        f"{row['description']}"
        for row in report["architecture_fitness"]
    )
    gaps = "\n".join(
        f"- {item}" for item in report["remaining_senior_evidence_gaps"]
    )
    criteria = "\n".join(
        f"- `{item['id']}` -> `{item['effect']}`: {item['condition']}"
        for item in report["falsification_criteria"]
    )
    path.write_text(
        "\n".join(
            [
                "# Senior Engineering Evidence Under Constraints",
                "",
                f"Status: `{report['status']}`",
                "",
                "This is an internal engineering-evidence dossier. It does not award",
                "professional seniority and is not clinical or production-healthcare proof.",
                "",
                "## Architecture fitness",
                checks,
                "",
                "## Falsification criteria",
                criteria,
                "",
                "## Remaining gaps",
                gaps,
                "",
                "## Claim boundary",
                report["claim_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )


__all__ = ["build_senior_engineering_evidence"]
