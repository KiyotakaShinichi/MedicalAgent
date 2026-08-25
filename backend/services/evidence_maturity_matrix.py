"""Cross-domain evidence maturity and maintainability budget.

This is deliberately not an average score. A large number of internally
generated artifacts must not compensate for missing independent, clinical, or
real-traffic evidence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/governance/latest_evidence_maturity_matrix.json"
)
DEFAULT_DOC_PATH = Path("docs/evidence_maturity_matrix.md")

EVIDENCE_TIERS = {
    0: "unmeasured_or_missing",
    1: "scaffold_or_contract_only",
    2: "internal_self_test",
    3: "frozen_internal_with_contamination_controls",
    4: "independent_external_engineering_evidence",
    5: "institutional_or_clinical_evidence",
}

ARTIFACTS = {
    "rag": "Data/evals/rag/latest_rag_paired_statistical_comparison.json",
    "citation_selector": "Data/evals/rag/latest_claim_conditioned_citation_selector_eval.json",
    "citation_selector_holdout": "Data/evals/rag/latest_claim_conditioned_citation_selector_holdout.json",
    "research_paper_rag": "Data/evals/rag/latest_research_paper_retrieval_eval.json",
    "adversarial": "Data/evals/safety/latest_adversarial_holdout_v7_baseline.json",
    "cost": "Data/evals/ops/latest_cost_latency_report.json",
    "ml": "Data/evals/models/latest_synthetic_prediction_statistical_audit.json",
    "xai": "Data/evals/models/latest_xai_reliability_gate.json",
    "finetune": "Data/evals/models/latest_finetune_hardening_assurance.json",
    "finetune_adjudication": "Data/evals/models/latest_finetune_contamination_adjudication_readiness.json",
    "ship": "Data/evals/ops/latest_ship_run.json",
    "automation": "Data/evals/ops/latest_automation_reliability_dossier.json",
    "cloud": "Data/evals/ops/latest_cloud_infrastructure_readiness.json",
    "deployment": "Data/evals/ops/latest_deployment_readiness.json",
    "synthetic_staging": "Data/evals/ops/latest_disposable_synthetic_staging_readiness.json",
    "ml_perturbation": "Data/evals/models/latest_synthetic_model_perturbation_retrain_eval.json",
    "medical_review": "Data/evals/medical/latest_medical_advisor_review_packet.json",
    "external_review": "Data/evals/governance/latest_external_review_readiness.json",
    "data_platform": "Data/lakehouse/manifests/latest_pipeline_run.json",
}

CLAIM_BOUNDARY = (
    "This matrix rates evidence provenance, not product quality or clinical "
    "readiness. Internal engineering evidence cannot be averaged into clinical "
    "validation, real-world safety, patient benefit, or production healthcare readiness."
)


def _research_paper_sentence(research_paper_rag: dict) -> str:
    """Describe the paper suite as it stands in *this* run.

    The case count used to be written into the sentence as a constant, which
    read as a current measurement even on a checkout where the optional corpus
    is absent and nothing was measured at all. The count now comes from the
    artifact, and when there is no current evaluation the sentence says so
    instead of quoting a number.
    """
    status = research_paper_rag.get("status")
    if research_paper_rag.get("evaluated") is False:
        return (
            "The corpus-derived research-paper suite was not evaluated in this run "
            f"(status={status}); its optional local corpus is unavailable, so no "
            "current PMCID identity, section retrieval, or provenance measurement exists."
        )
    case_count = research_paper_rag.get("case_count")
    scale = f"{case_count}-case " if isinstance(case_count, int) else ""
    return (
        f"A separate {scale}corpus-derived research-paper suite measures PMCID identity, "
        f"section retrieval, provenance, and no-evidence behavior (status={status})."
    )


def build_evidence_maturity_matrix(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    artifacts = {name: _read(path) for name, path in ARTIFACTS.items()}
    dimensions = _dimensions(artifacts)
    architecture = _architecture_budget()
    tier_counts = {
        str(tier): sum(item["current_evidence_tier"] == tier for item in dimensions)
        for tier in EVIDENCE_TIERS
    }
    payload = {
        "schema_version": "evidence_maturity_matrix_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "needs_attention"
            if any(item["current_evidence_tier"] < 2 for item in dimensions)
            or architecture["budget_status"] == "needs_attention"
            else "acceptable"
        ),
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "scoring_policy": {
            "aggregate_score_emitted": False,
            "reason": (
                "Averages hide blocking dimensions and let artifact volume "
                "compensate for missing independent evidence."
            ),
            "tiers": EVIDENCE_TIERS,
            "tier_counts": tier_counts,
        },
        "dimensions": dimensions,
        "architecture_maintainability": architecture,
        "cross_domain_blockers": [
            "No independent clean-clone reproduction.",
            "No completed no-read external RAG/adversarial evaluation.",
            "No clinician or genetic-counselor review.",
            "No real traffic, managed-cloud failover, or audited provider billing evidence.",
            "No real patient data, IRB/ethics approval, or clinical validation.",
        ],
        "highest_roi_internal_actions": [
            "Adjudicate fine-tune semantic-similarity flags before any candidate run.",
            "Collect provider-reported token usage on 30+ representative calls.",
            "Keep the claim-conditioned citation selector offline after its negative frozen holdout; improve upstream evidence selection on a separate development bank.",
            "Use non-frozen mutation cases to address the remaining research-paper boundary escape, then rerun the internal bank with tuning disclosure.",
            "Keep unstable XAI rank order hidden and expose grouped factors only.",
            "Hold the oversized-file ratchet at nine or lower and continue replacing concentrated ownership surfaces before adding feature breadth.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write(output_path, payload)
    _write_doc(doc_path, payload)
    return payload


def _dimensions(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rag_headline = artifacts["rag"].get("headline") or {}
    citation_selector = artifacts["citation_selector"]
    citation_selector_holdout = artifacts["citation_selector_holdout"]
    research_paper_rag = artifacts["research_paper_rag"]
    research_summary = research_paper_rag.get("summary") or {}
    adversarial = artifacts["adversarial"]
    cost_summary = artifacts["cost"].get("summary") or {}
    provider = cost_summary.get("provider_reported_usage") or {}
    fine_summary = artifacts["finetune"].get("summary") or {}
    finetune_adjudication = artifacts["finetune_adjudication"]
    xai_policy = artifacts["xai"].get("patient_display_policy") or {}
    automation = artifacts["automation"]
    cloud = artifacts["cloud"]
    deployment = artifacts["deployment"]
    synthetic_staging = artifacts["synthetic_staging"]
    ml_perturbation = artifacts["ml_perturbation"]
    ship = artifacts["ship"]
    data_platform = artifacts["data_platform"]
    return [
        _dimension(
            "AIE/RAG",
            3,
            "Frozen internal case-level comparisons, paired bootstrap intervals, "
            "multiple-comparison correction, source governance, negative results, and "
            f"an offline claim-conditioned citation candidate status={citation_selector.get('status')}. "
            f"Its frozen generated-answer holdout decision={citation_selector_holdout.get('promotion_decision')}. "
            f"{_research_paper_sentence(research_paper_rag)}",
            (
                "Full governed-stack raw Recall@10 superiority over BM25 is not proven; "
                f"current headline={rag_headline.get('full_stack_improvement_proven_vs_bm25')}. "
                "The citation selector regressed citation precision/support on its frozen generated-answer holdout and remains offline. "
                f"The paper suite is not independent and its boundary-route correctness is "
                f"{research_summary.get('boundary_route_correctness')}; its boundary-gated escape rate is "
                f"{research_summary.get('boundary_gated_no_evidence_escape_rate')}."
            ),
            "Improve upstream evidence selection on a separate development bank, preserve the negative selector result, "
            "and complete an independent no-read holdout.",
        ),
        _dimension(
            "AIE/adversarial safety",
            3,
            "One-pass frozen internal v7 with explicit author-contamination disclosure, "
            "plus separate tuning-only development and safe-negative controls.",
            (
                f"V7 unsafe leakage={adversarial.get('unsafe_leakage_rate')} and "
                f"over-refusal={adversarial.get('over_refusal_rate')}; "
                "the result is weak and external authorship is absent."
            ),
            "External mutation/red-team bank passes predeclared thresholds without tuning.",
        ),
        _dimension(
            "LLM token/latency observability",
            2,
            "Request totals, local route samples, stage timings, token estimates, and provider usage fields.",
            f"Provider usage coverage is {provider.get('coverage_rate')}; cost is not billing reconciliation.",
            "30+ representative calls with >=80% provider usage plus staged cloud latency/cost reconciliation.",
        ),
        _dimension(
            "MLE/statistics",
            2,
            "Patient-level temporal splits, leakage/shortcut audits, bootstrap uncertainty, "
            "paired tests, calibration, train-only constant and linear baselines, "
            "coverage-performance curves, synthetic perturbation sensitivity, and repeated patient-split sensitivity "
            f"across {(ml_perturbation.get('repeated_patient_split_stability') or {}).get('split_count')} fixed seeds.",
            "All outcomes and uncertainty remain simulator-bounded; transportability is unproven.",
            (
                "Resolve documented synthetic stress failures "
                f"(current count={len(ml_perturbation.get('stress_failures') or [])}) "
                "and use a task-aligned external cohort, or retain synthetic engineering-only positioning."
            ),
        ),
        _dimension(
            "XAI",
            2,
            "Mechanical additivity, bootstrap set stability, retraining stability, and fail-closed display policy.",
            f"Patient display mode is {xai_policy.get('mode')}; exact feature rank stability is not established.",
            "Stable retraining evidence plus human comprehension review; causality remains blocked.",
        ),
        _dimension(
            "Fine-tuning",
            1,
            "Behavior-only synthetic dataset, immutable revision contracts, promotion tripwires, "
            "and semantic-similarity screening.",
            f"Promotion={artifacts['finetune'].get('promotion_decision')}; blockers={fine_summary.get('blocking_gap_count')}; "
            f"human adjudication unresolved={finetune_adjudication.get('unresolved_count')} with critical dual-review required.",
            "Pinned runtime, matched generations, lineage, memorization audit, dual-reviewed contamination flags, and paired lift.",
        ),
        _dimension(
            "SWE/release discipline",
            2,
            f"Integrated ship manifest status={ship.get('status')} with repeatable tests and release gates.",
            "Evidence is owner-run; nine oversized modules remain even though the declared ratchet is met, and no independent clean-clone reproduction exists.",
            "Independent clean-clone reproduction and continued module reduction without expanding the service/artifact surface.",
        ),
        _dimension(
            "Automation",
            2,
            (
                f"Local redacted outbox/retry/dead-letter/idempotency contracts status={automation.get('status')}; "
                "a loopback n8n import and synthetic MailHog receipt are executable."
            ),
            "External delivery is disabled by default and no live clinician acknowledgement workflow is proven.",
            "Independent operator drill and approved external sandbox receipt; real use still requires governance.",
        ),
        _dimension(
            "Infrastructure/deployment",
            (
                2
                if synthetic_staging.get("runtime_healthchecks_completed") is True
                and synthetic_staging.get("postgres_restore_drill_completed") is True
                else 1
            ),
            (
                f"Reference cloud architecture status={cloud.get('status')}; disposable loopback runtime="
                f"{synthetic_staging.get('runtime_healthchecks_completed')} and Postgres restore="
                f"{synthetic_staging.get('postgres_restore_drill_completed')}."
            ),
            (
                f"Deployment readiness={deployment.get('status')}; local containers do not prove authenticated "
                "cloud deployment, managed failover, external traffic, secret rotation, or cloud cost."
            ),
            "Managed non-patient staging with authenticated deployment, failover, restore, load, secret rotation, and cost evidence.",
        ),
        _dimension(
            "Data engineering",
            2,
            f"Non-patient bronze/silver/gold lineage pipeline status={data_platform.get('status')}.",
            "No governed real-patient pipeline, managed deletion proof, or healthcare interoperability evidence.",
            "Managed non-patient shadow pipeline proves idempotency, deletion, replay, quality quarantine, and recovery.",
        ),
        _dimension(
            "Medical/human factors",
            1,
            "Deterministic boundaries, evidence policies, escalation language, overtrust warnings, and review packets.",
            "No clinician, nurse, pharmacist, genetic counselor, or real-user review has been completed.",
            "Qualified reviewers complete case-linked logs; this still does not equal clinical validation.",
        ),
    ]


def _dimension(
    name: str,
    tier: int,
    proven: str,
    not_proven: str,
    next_evidence: str,
) -> dict[str, Any]:
    return {
        "dimension": name,
        "current_evidence_tier": tier,
        "tier_label": EVIDENCE_TIERS[tier],
        "what_is_proven": proven,
        "what_is_not_proven": not_proven,
        "next_falsifiable_evidence": next_evidence,
    }


def _architecture_budget() -> dict[str, Any]:
    roots = [
        (ROOT_DIR / "backend", ".py", 1000, 1800),
        (ROOT_DIR / "scripts", ".py", 1000, 1800),
        (ROOT_DIR / "frontend-react" / "src", (".ts", ".tsx"), 800, 1200),
    ]
    rows = []
    for root, suffixes, warning, critical in roots:
        allowed = {suffixes} if isinstance(suffixes, str) else set(suffixes)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in allowed:
                continue
            if "generated-openapi" in path.name:
                continue
            try:
                line_count = sum(
                    1 for _ in path.open("r", encoding="utf-8", errors="ignore")
                )
            except OSError:
                continue
            severity = (
                "critical"
                if line_count >= critical
                else "warning"
                if line_count >= warning
                else "within_budget"
            )
            if severity != "within_budget":
                rows.append(
                    {
                        "path": path.relative_to(ROOT_DIR).as_posix(),
                        "line_count": line_count,
                        "severity": severity,
                        "warning_threshold": warning,
                        "critical_threshold": critical,
                    }
                )
    rows.sort(key=lambda item: item["line_count"], reverse=True)
    service_count = len(list((ROOT_DIR / "backend" / "services").glob("*.py")))
    test_count = len(list((ROOT_DIR / "tests").glob("test_*.py")))
    latest_artifact_count = len(
        list((ROOT_DIR / "Data" / "evals").rglob("latest_*.json"))
    )
    oversized_baseline = 9
    critical_count = sum(item["severity"] == "critical" for item in rows)
    within_ratchet = len(rows) <= oversized_baseline and critical_count == 0
    return {
        "budget_status": "acceptable" if within_ratchet else "needs_attention",
        "oversized_file_count": len(rows),
        "critical_file_count": critical_count,
        "backend_service_file_count": service_count,
        "test_file_count": test_count,
        "test_to_service_file_ratio": round(test_count / max(service_count, 1), 4),
        "ratchet": {
            "oversized_file_baseline": oversized_baseline,
            "oversized_file_count_within_baseline": len(rows) <= oversized_baseline,
            "critical_file_ceiling": 0,
            "critical_file_count_within_ceiling": not any(
                item["severity"] == "critical" for item in rows
            ),
        },
        "artifact_budget": {
            "latest_json_artifact_count": latest_artifact_count,
            "warning_threshold": 200,
            "status": (
                "needs_consolidation"
                if latest_artifact_count > 200
                else "within_budget"
            ),
            "target": (
                "Consolidate superseded latest_* artifacts below 200 without "
                "deleting audit history."
            ),
        },
        "largest_violations": rows[:20],
        "policy": (
            "Do not add a new service or artifact solely to increase feature count. "
            "New modules should close a measured gap, replace an older surface, or "
            "come with a deletion/consolidation plan."
        ),
    }


def _read(path: str | Path) -> dict[str, Any]:
    full = ROOT_DIR / Path(path)
    if not full.exists():
        return {}
    try:
        payload = json.loads(full.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    lines = [
        "# Evidence Maturity Matrix",
        "",
        "No aggregate score is emitted. Evidence volume cannot compensate for a "
        "blocking domain or missing independent review.",
        "",
        "| Dimension | Tier | Proven | Not proven |",
        "|---|---:|---|---|",
    ]
    for row in payload["dimensions"]:
        lines.append(
            f"| {row['dimension']} | {row['current_evidence_tier']} "
            f"({row['tier_label']}) | {row['what_is_proven']} | "
            f"{row['what_is_not_proven']} |"
        )
    architecture = payload["architecture_maintainability"]
    lines.extend(
        [
            "",
            "## Architecture Budget",
            "",
            f"- Status: `{architecture['budget_status']}`",
            f"- Oversized files: `{architecture['oversized_file_count']}`",
            f"- Critical files: `{architecture['critical_file_count']}`",
            f"- Backend service files: `{architecture['backend_service_file_count']}`",
            "",
            architecture["policy"],
            "",
            "## Boundary",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["build_evidence_maturity_matrix"]
