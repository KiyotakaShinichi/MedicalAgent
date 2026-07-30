"""Canonical registry for evidence gaps that green internal gates cannot erase."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/governance/latest_credibility_gap_registry.json"
DEFAULT_DOC_PATH = "docs/credibility_gap_registry.md"

CLAIM_BOUNDARY = (
    "This registry tracks engineering evidence gaps. Closing an internal gap does "
    "not establish clinical validation, patient benefit, clinician approval, IRB "
    "approval, production healthcare readiness, or generalisation to real patients."
)


def build_credibility_gap_registry(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    cost = _read("Data/evals/ops/latest_cost_latency_report.json")
    finetune = _read("Data/evals/models/latest_finetune_hardening_assurance.json")
    finetune_semantic = _read(
        "Data/evals/models/latest_finetune_semantic_contamination.json"
    )
    rag = _read("Data/evals/rag/latest_rag_baseline_comparison.json")
    rag_paired = _read(
        "Data/evals/rag/latest_rag_paired_statistical_comparison.json"
    )
    senior = _read("Data/evals/governance/latest_senior_engineering_evidence.json")
    automation = _read("Data/evals/ops/latest_automation_reliability_dossier.json")
    adversarial = _read("Data/evals/safety/latest_adversarial_v6_tuning_regression.json")

    cost_summary = cost.get("summary") or {}
    provider_usage = cost_summary.get("provider_reported_usage") or {}
    latency = cost_summary.get("overall_latency_ms") or {}
    local_probe = cost.get("local_probe_stage_latency") or {}
    finetune_summary = finetune.get("summary") or {}
    semantic_summary = finetune_semantic.get("summary") or {}
    automation_summary = automation.get("summary") or {}

    gaps = [
        _gap(
            "provider_token_usage_coverage",
            "AIE/observability",
            "medium",
            _state(
                float(provider_usage.get("coverage_rate") or 0.0) >= 0.8
                and int(latency.get("sample_count") or 0) >= 30
            ),
            True,
            False,
            ["Data/evals/ops/latest_cost_latency_report.json"],
            {
                "coverage_rate": provider_usage.get("coverage_rate"),
                "latency_sample_count": latency.get("sample_count"),
            },
            "Capture provider-reported usage on at least 80% of 30+ representative requests.",
            "python scripts/run_cost_latency_report.py",
            "AI platform owner",
            "Token totals are partly estimated; do not present them as provider billing truth.",
        ),
        _gap(
            "tail_latency_evidence",
            "SWE/infra",
            "high",
            _state(
                latency.get("percentile_credibility")
                in {"directional_internal_sample", "stable_internal_sample"}
            ),
            True,
            False,
            ["Data/evals/ops/latest_cost_latency_report.json"],
            {
                "sample_count": latency.get("sample_count"),
                "p95_ms": latency.get("p95"),
                "credibility": latency.get("percentile_credibility"),
                "local_probe_stage_sample_count": local_probe.get(
                    "measured_stage_sample_count"
                ),
                "local_probe_environment": local_probe.get("environment"),
            },
            "Collect 100+ representative requests and pass route-specific p95 budgets without hiding cold starts; reconcile the local probe against staged cloud traffic.",
            "python scripts/run_cost_latency_report.py",
            "SWE/infra owner",
            "Current p95 is internal and sample-dependent, not a production SLO.",
        ),
        _gap(
            "fine_tune_runtime_and_candidate",
            "MLE/fine-tuning",
            "high",
            _state(
                finetune.get("status") == "strong"
                and finetune.get("promotion_decision") == "PROMOTE"
            ),
            True,
            False,
            [
                "Data/evals/models/latest_finetune_runtime_preflight.json",
                "Data/evals/models/latest_finetune_hardening_assurance.json",
                "Data/evals/models/latest_finetune_promotion_gate.json",
            ],
            {
                "promotion_decision": finetune.get("promotion_decision"),
                "blocking_gap_ids": finetune_summary.get("blocking_gap_ids"),
            },
            "Pass pinned runtime, lineage, memorization, per-behavior, paired-statistical, safety, and output-length gates.",
            "python scripts/run_finetune_hardening_assurance.py",
            "MLE owner",
            "Fine-tuning is scaffolded only; no adapter improvement is proven.",
        ),
        _gap(
            "fine_tune_semantic_contamination",
            "MLE/evaluation",
            "medium",
            _state(
                finetune_summary.get("semantic_similarity_screen_completed") is True
                and finetune_summary.get(
                    "semantic_adjudication_cleared_for_candidate"
                )
                is True
                and int(
                    finetune_summary.get("semantic_unresolved_pair_count") or 0
                )
                == 0
            ),
            True,
            False,
            [
                "Data/evals/models/latest_finetune_semantic_contamination.json",
                "Data/finetune/prepared/dataset_card.json",
            ],
            {
                "semantic_similarity_proxy_completed": (
                    (finetune_semantic.get("screen") or {}).get(
                        "semantic_similarity_proxy_completed"
                    )
                ),
                "flagged_pair_count": semantic_summary.get("flagged_pair_count"),
                "unresolved_pair_count": semantic_summary.get(
                    "unresolved_pair_count"
                ),
                "review_completed": semantic_summary.get("review_completed"),
                "adjudication_cleared_for_candidate": semantic_summary.get(
                    "adjudication_cleared_for_candidate"
                ),
                "artifact_flag_rows_capped": semantic_summary.get(
                    "artifact_flag_rows_capped"
                ),
                "semantic_contamination_absence_proven": semantic_summary.get(
                    "semantic_contamination_absence_proven"
                ),
            },
            "Run semantic/paraphrase contamination detection with reviewer adjudication of flagged pairs.",
            "python scripts/run_finetune_semantic_contamination.py",
            "MLE evaluation owner",
            "TF-IDF is a lexical-semantic proxy; even a completed review does not prove semantic independence.",
        ),
        _gap(
            "rag_improvement_over_bm25",
            "AIE/RAG",
            "high",
            _state(_rag_improvement_proven(rag)),
            True,
            False,
            [
                "Data/evals/rag/latest_rag_baseline_comparison.json",
                "Data/evals/rag/latest_rag_paired_statistical_comparison.json",
            ],
            {
                "improvement_proven_vs_bm25": _rag_improvement_proven(rag),
                "comparison_scope": "internal frozen goldset",
                "paired_headline": rag_paired.get("headline"),
            },
            "Demonstrate a predeclared improvement on an independent no-read holdout, or retain governance-first positioning.",
            "python scripts/run_rag_holdout_baseline_comparison.py",
            "RAG evaluation owner",
            "The complex stack is governance-oriented; raw retrieval superiority over BM25 is not proven.",
        ),
        _gap(
            "frozen_adversarial_generalization",
            "AIE/safety",
            "high",
            _state(_adversarial_passed(adversarial)),
            True,
            False,
            ["Data/evals/safety/latest_adversarial_v6_tuning_regression.json"],
            {"status": adversarial.get("status"), "summary": adversarial.get("summary")},
            "Meet predeclared frozen-bank thresholds without using the bank for tuning and preserve safe-negative performance.",
            "python scripts/run_adversarial_v6_tuning_regression.py",
            "AI safety owner",
            "Safety is not solved; frozen adversarial weaknesses remain visible.",
        ),
        _gap(
            "independent_clean_clone_reproduction",
            "SWE/reproducibility",
            "high",
            "blocked_external",
            False,
            True,
            ["Data/evals/governance/latest_senior_engineering_evidence.json"],
            {
                "independent_reproduction_completed": senior.get(
                    "independent_reproduction_completed"
                )
            },
            "A reviewer with no project involvement reproduces setup, tests, artifacts, and demo from a clean clone.",
            "python scripts/ship.py",
            "Independent peer engineer",
            "The owner has internal reproducibility evidence, not independent reproduction.",
        ),
        _gap(
            "external_no_read_evaluation",
            "Evaluation governance",
            "high",
            "blocked_external",
            False,
            True,
            [
                "Data/evals/rag/latest_rag_holdout_baseline_comparison.json",
                "docs/evals/no_read_rag_goldset_protocol.md",
            ],
            {"external_author_eval_completed": False},
            "An eligible external author completes the no-read RAG and adversarial protocols with attestation.",
            "python scripts/run_rag_holdout_baseline_comparison.py",
            "External evaluation author",
            "Prepared external evaluation is not completed external evidence.",
        ),
        _gap(
            "clinician_and_genetics_review",
            "Medical governance",
            "critical",
            "blocked_external",
            False,
            True,
            [
                "docs/review_packets/nurse_or_clinician_safety_review_packet.md",
                "docs/review_packets/genetic_counselor_vus_review_packet.md",
            ],
            {
                "clinician_review_completed": False,
                "genetic_counselor_review_completed": False,
            },
            "Qualified reviewers complete dated, case-linked review logs; this still does not equal clinical approval.",
            "manual external review",
            "External clinician and genetic counselor",
            "Medical wording and boundaries are unreviewed by clinicians.",
        ),
        _gap(
            "live_cloud_and_delivery_evidence",
            "Infra/automation",
            "high",
            _state(
                bool(automation_summary.get("live_delivery_receipt_verified"))
                and bool(automation_summary.get("live_failover_drill_completed"))
            ),
            True,
            False,
            [
                "Data/evals/ops/latest_automation_reliability_dossier.json",
                "Data/evals/ops/latest_cloud_infrastructure_readiness.json",
            ],
            {
                "live_delivery_receipt_verified": automation_summary.get(
                    "live_delivery_receipt_verified"
                ),
                "live_failover_drill_completed": automation_summary.get(
                    "live_failover_drill_completed"
                ),
            },
            "Run staged live delivery, retry, duplicate suppression, restore, failover, load, and cost reconciliation drills.",
            "python scripts/run_automation_reliability_dossier.py",
            "Infra/automation owner",
            "Local/synthetic automation readiness is not live cloud reliability.",
        ),
        _gap(
            "real_data_irb_clinical_validation",
            "Clinical evidence",
            "critical",
            "blocked_institutional",
            False,
            True,
            ["README.md"],
            {
                "real_patient_data": False,
                "irb_or_ethics_approval": False,
                "clinical_validation": False,
            },
            "Requires institutionally governed real data, ethics/IRB review, clinical protocol, and qualified oversight.",
            "not self-certifiable",
            "Clinical institution",
            "No clinical readiness, real-world safety, patient benefit, or healthcare-production claim is allowed.",
        ),
    ]

    internally_closed = [item for item in gaps if item["current_status"] == "complete_internal"]
    self_controllable = [item for item in gaps if item["controllable_now"]]
    payload = {
        "schema_version": "credibility_gap_registry_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if len(internally_closed) < len(gaps) else "strong",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "summary": {
            "gap_count": len(gaps),
            "internally_closed_count": len(internally_closed),
            "open_or_external_count": len(gaps) - len(internally_closed),
            "self_controllable_count": len(self_controllable),
            "cannot_be_self_certified_count": sum(
                1 for item in gaps if item["cannot_be_self_certified"]
            ),
            "internal_evidence_closure_rate": round(
                len(internally_closed) / len(gaps), 4
            ),
            "score_interpretation": (
                "Evidence-completion bookkeeping only; not a quality, safety, or clinical-readiness score."
            ),
        },
        "gaps": gaps,
        "next_three_controllable_actions": [
            "Capture 30+ instrumented provider calls and refresh token/latency telemetry.",
            "Provision the pinned offline PEFT runtime and generate matched baseline/candidate outputs with manifests.",
            "Run semantic fine-tune contamination review and frozen adversarial regression without tuning on holdouts.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write(output_path, payload)
    _write_doc(doc_path, payload)
    return payload


def _gap(
    identifier: str,
    domain: str,
    severity: str,
    current_status: str,
    controllable_now: bool,
    cannot_be_self_certified: bool,
    evidence_artifacts: list[str],
    evidence_snapshot: dict[str, Any],
    completion_criteria: str,
    verification_command: str,
    owner: str,
    honest_claim_until_closed: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "domain": domain,
        "severity": severity,
        "current_status": current_status,
        "controllable_now": controllable_now,
        "cannot_be_self_certified": cannot_be_self_certified,
        "evidence_artifacts": evidence_artifacts,
        "evidence_snapshot": evidence_snapshot,
        "completion_criteria": completion_criteria,
        "verification_command": verification_command,
        "owner": owner,
        "honest_claim_until_closed": honest_claim_until_closed,
    }


def _state(passed: bool) -> str:
    return "complete_internal" if passed else "open"


def _rag_improvement_proven(payload: dict[str, Any]) -> bool:
    if payload.get("improvement_proven_vs_bm25") is not None:
        return bool(payload.get("improvement_proven_vs_bm25"))
    summary = payload.get("summary") or {}
    return bool(
        summary.get("improvement_proven_vs_bm25")
        or summary.get("complex_stack_improvement_proven")
    )


def _adversarial_passed(payload: dict[str, Any]) -> bool:
    return payload.get("status") in {"strong", "acceptable"} and bool(
        (payload.get("summary") or {}).get("no_frozen_regression")
    )


def _read(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    if not full.exists():
        return {}
    try:
        parsed = json.loads(full.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Credibility Gap Registry",
        "",
        "This registry prevents green internal tests from being mistaken for external or clinical evidence.",
        "",
        f"- Open or external gaps: `{payload['summary']['open_or_external_count']}`",
        f"- Cannot be self-certified: `{payload['summary']['cannot_be_self_certified_count']}`",
        "",
        "## Gaps",
        "",
    ]
    for item in payload["gaps"]:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- Domain: `{item['domain']}`",
                f"- Status: `{item['current_status']}`",
                f"- Severity: `{item['severity']}`",
                f"- Owner: `{item['owner']}`",
                f"- Completion: {item['completion_criteria']}",
                f"- Until closed: {item['honest_claim_until_closed']}",
                "",
            ]
        )
    full.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "build_credibility_gap_registry",
]
