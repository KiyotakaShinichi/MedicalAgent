"""Negative-results gallery.

An explicit catalogue of findings that did *not* work: retrieval
configurations that lost to BM25, models held back from promotion, safety
categories still below threshold. Publishing these alongside the positive
results is what stops the evidence base from reading as advertising.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NEGATIVE_RESULTS_PATH = Path("Data/evals/governance/latest_negative_results_gallery.json")

#: Statuses meaning the artifact behind an entry was not produced by this run.
NON_RESULT_STATUSES = frozenset({"not_evaluated_optional_corpus"})


def _artifact_status(relative_path: str) -> str | None:
    try:
        payload = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("status")


def _mark_evidence_currency(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Say plainly that every recorded metric is from the run that found it.

    The metric values in this gallery are literals: they were written down when
    the finding was made and are never recomputed. That is deliberate - a
    negative result must not be quietly revised - but it also means they are
    historical by construction, and nothing here should be read as this run's
    numbers.

    The research-paper entry makes that concrete. Its metrics describe a
    nine-paper, thirty-two-case corpus that no longer exists; the current suite
    is larger, and on a checkout without the optional corpus there is no current
    suite at all. So each entry also carries the *current* status of the
    artifact it cites, which is how a reader tells "still true" from "recorded
    once".
    """
    marked = []
    for item in items:
        entry = dict(item)
        entry["evidence_currency"] = "historical"
        entry["metric_value_is_current_run"] = False
        artifact = entry.get("evidence_artifact")
        if artifact:
            status = _artifact_status(str(artifact))
            entry["current_artifact_status"] = status
            if status in NON_RESULT_STATUSES:
                entry["current_run_evaluated"] = False
                entry["currency_note"] = (
                    "The cited artifact reports no evaluation for the current run, so "
                    "the metrics above are historical only and must not be quoted as "
                    "current evidence."
                )
            elif status is not None:
                entry["current_run_evaluated"] = True
        marked.append(entry)
    return marked


def build_negative_results_gallery() -> dict[str, Any]:
    items = [
        {
            "title": "Full source-governed RAG stack does not beat BM25 on raw Recall@10",
            "evidence_artifact": "Data/evals/rag/latest_rag_baseline_comparison.json",
            "metric_value": {
                "bm25_recall_at_10": 0.8041,
                "full_stack_recall_at_10": 0.7838,
                "improvement_proven_vs_bm25": False,
            },
            "why_it_matters": (
                "The source-governed stack is valuable for safety and source-tier "
                "governance, but it does NOT deliver a raw-recall lift on the "
                "frozen internal goldset.  Quoting full-stack recall as a model "
                "improvement is overclaiming."
            ),
            "decision_taken": (
                "Stack retained for governance value.  Reporting wording switched "
                "to 'governance/safety tradeoff' rather than 'retrieval improvement'."
            ),
            "what_was_not_claimed": [
                "raw retrieval improvement over BM25",
                "production-grade retrieval superiority",
                "clinical validation",
            ],
            "next_action": (
                "Complete held-out v2 RAG eval under the no-read protocol; "
                "re-run baseline comparison; report side-by-side."
            ),
            "clinical_validation": False,
        },
        {
            "title": "Research-paper KB benchmark remains corpus-derived and BM25-competitive",
            "evidence_artifact": "Data/evals/rag/latest_research_paper_retrieval_eval.json",
            "metric_value": {
                "paper_count": 9,
                "case_count": 32,
                "bm25_recall_at_10": 1.0,
                "full_stack_recall_at_10": 0.963,
                "paired_delta": -0.037,
                "bootstrap_ci95": [-0.111111, 0.0],
                "exact_sign_test_p": 1.0,
                "boundary_route_correctness": 0.6,
                "improvement_proven_vs_bm25": False,
            },
            "why_it_matters": (
                "The paper suite verifies PMCID identity and provenance, but it was "
                "authored from the same narrow nine-paper corpus. BM25 retrieves every "
                "positive case, and two of five no-evidence premises miss the intended "
                "pre-retrieval boundary route."
            ),
            "decision_taken": (
                "Keep the suite as an untuned internal regression with status "
                "needs_attention. Do not tune retrieval or safety rules on this run."
            ),
            "what_was_not_claimed": [
                "paper retrieval improvement over BM25",
                "independent literature generalization",
                "broad oncology knowledge coverage",
                "clinical validation",
            ],
            "next_action": (
                "Use a separately authored no-read paper-query set and resolve generalized "
                "no-evidence routing gaps on a development-only bank."
            ),
            "clinical_validation": False,
        },
        {
            "title": "Citation context pruner regressed citation precision",
            "evidence_artifact": "Data/evals/rag/latest_rag_baseline_comparison.json",
            "metric_value": {
                "full_stack_citation_precision": 0.5243,
                "full_stack_plus_pruner_citation_precision": 0.4275,
                "citation_precision_delta": -0.0968,
                "recall_at_5_delta": +0.0743,
                "promoted_to_live_agent": False,
            },
            "why_it_matters": (
                "The pruner improves top-5 recall but trades that off against "
                "citation precision.  The brief was explicit: do not optimize for "
                "recall when citation precision worsens.  Live agent stays "
                "unchanged."
            ),
            "decision_taken": (
                "Pruner remains eval-path only; documented as experimental; not "
                "wired into apply_intent_aware_rag_layer."
            ),
            "what_was_not_claimed": [
                "retrieval improvement",
                "deployment readiness",
                "promotion to live agent",
            ],
            "next_action": (
                "Leave pruner experimental; revisit only after held-out v2 result."
            ),
            "clinical_validation": False,
        },
        {
            "title": "Cross-encoder reranker lift not proven",
            "evidence_artifact": "Data/evals/rag/latest_reranker_ablation.json",
            "metric_value": {
                "improvement_proven": "needs_attention",
                "promoted_to_default": False,
            },
            "why_it_matters": (
                "Cross-encoder reranking was scaffolded as opt-in via "
                "RAG_ENABLE_CROSS_ENCODER=true, but the ablation artifact has "
                "not shown a clean lift without higher unsupported-context risk."
            ),
            "decision_taken": (
                "Cross-encoder kept off by default.  No claim of retrieval lift."
            ),
            "what_was_not_claimed": [
                "reranking improves the retrieval stack",
                "production reranking readiness",
            ],
            "next_action": (
                "Re-evaluate cross-encoder only after held-out v2 and after the "
                "source-filter-drop adjudication is filled."
            ),
            "clinical_validation": False,
        },
        {
            "title": "Held-out adversarial generalization is weak on hardened categories",
            "evidence_artifact": "Data/evals/safety/latest_adversarial_safety_holdout.json",
            "metric_value": {
                "in_sample_attack_block_rate_per_hardened_category": 1.0,
                "held_out_v1_attack_block_rate_overall": 0.0625,
                "promoted_to_release_gate_blocker": False,
            },
            "why_it_matters": (
                "In-sample 1.0 across the four hardened categories is bank-tuned. "
                "Held-out v1 generalisation is ~6%.  The gap is recorded openly "
                "and the held-out result is informational only."
            ),
            "decision_taken": (
                "Held-out v1 stays informational.  Anti-contamination test enforces "
                "no held-out query appears in the original bank.  See ADR 0005."
            ),
            "what_was_not_claimed": [
                "the adversarial bank generalises",
                "the safety vocabulary is robust to paraphrase",
                "clinical safety validation",
            ],
            "next_action": (
                "Author external adversarial cases (15+) under attestation; revisit."
            ),
            "clinical_validation": False,
        },
        {
            "title": "Frozen internal adversarial v6 regressed sharply",
            "evidence_artifact": "Data/evals/safety/latest_adversarial_holdout_v6_baseline.json",
            "metric_value": {
                "total_n": 162,
                "pass_rate": 0.518519,
                "unsafe_leakage_rate": 0.560606,
                "over_refusal_rate": 0.133333,
                "was_used_for_tuning": False,
            },
            "why_it_matters": (
                "The newest one-pass internally authored holdout is materially worse than v5. "
                "Cross-patient, VUS, genetics, and diagnosis variants remain especially fragile, "
                "so green development-set results do not generalise reliably."
            ),
            "decision_taken": (
                "V6 is locked as a warning and is not used for further tuning in this pass. "
                "The classifier is unchanged after the one-pass evaluation."
            ),
            "what_was_not_claimed": [
                "adversarial safety is solved",
                "generalized unsafe-intent detection is robust",
                "clinical safety validation",
            ],
            "next_action": (
                "Use a new development-only mutation bank for generalized research, then freeze a newer "
                "holdout before any future classifier change; prioritize external-author cases."
            ),
            "clinical_validation": False,
        },
        {
            "title": "Source_filter_drop is mostly a goldset/governance mismatch",
            "evidence_artifact": "Data/evals/rag/latest_rag_stage_oracle_diagnostic.json",
            "metric_value": {
                "source_filter_drop_cases": 9,
                "total_failure_cases": 14,
                "oracle_gap": 0.0540,
                "source_filter_retention_rate": 0.8378,
            },
            "why_it_matters": (
                "The dominant failure stage is the source-tier filter doing what "
                "patient-facing governance requires.  The retrievers are fine; the "
                "goldset's expected_source_ids include clinician-facing sources."
            ),
            "decision_taken": (
                "Adjudication packet built; source-tier filter NOT weakened.  See "
                "docs/evals/rag_goldset_adjudication.md."
            ),
            "what_was_not_claimed": [
                "retrieval failure",
                "filter is too aggressive",
                "system is broken at retrieval",
            ],
            "next_action": (
                "Reviewer fills the 9-case adjudication packet under attestation."
            ),
            "clinical_validation": False,
        },
        {
            "title": "Toxicity signal is shortcut-prone (synthetic structural leakage)",
            "evidence_artifact": "Data/evals/models/latest_toxicity_feature_audit.json",
            "metric_value": {
                "toxicity_auc_synthetic": 0.9989,
                "shortcut_risk_documented": True,
                "monitor_only_policy": True,
            },
            "why_it_matters": (
                "The synthetic generator's label-construction rules leak structural "
                "signal into toxicity targets.  The headline AUC is not a clinical "
                "performance claim and the audit makes that explicit."
            ),
            "decision_taken": (
                "Toxicity signal retained as monitor-only.  Promotion to "
                "treatment influence is blocked by the promotion-policy artifact."
            ),
            "what_was_not_claimed": [
                "toxicity prediction validity",
                "clinical decision support",
                "treatment selection authority",
            ],
            "next_action": (
                "Noisier synthetic v2 with label noise; see "
                "docs/noisier_synthetic_v2_plan.md."
            ),
            "clinical_validation": False,
        },
        {
            "title": "All synthetic ML metrics are engineering self-tests, not clinical evidence",
            "evidence_artifact": "Data/evals/realism/latest_synthetic_data_quality.json",
            "metric_value": {
                "label_disclaimer_enforced_by_test": True,
                "is_clinical_realism_measure": False,
            },
            "why_it_matters": (
                "Every synthetic metric in the repo (AUC, calibration coverage, "
                "subgroup accuracy, conformal coverage) describes the synthetic "
                "distribution.  None of them establishes clinical predictive "
                "validity."
            ),
            "decision_taken": (
                "Synthetic data quality artifact self-labels as "
                "'synthetic_generator_quality_proxy' with a test-enforced "
                "disclaimer.  Headline metrics are always shown alongside the "
                "audit footnote."
            ),
            "what_was_not_claimed": [
                "clinical realism",
                "distribution similarity to a real cohort",
                "predictive validity",
            ],
            "next_action": (
                "Build noisier synthetic v2 so metrics differentiate; do not "
                "promote any synthetic metric to a clinical claim."
            ),
            "clinical_validation": False,
        },
        {
            "title": "External / no-read RAG holdout v2 is prepared but not completed",
            "evidence_artifact": "Data/evals/rag/latest_rag_holdout_baseline_comparison.json",
            "metric_value": {
                "status": "ready_for_external_authoring",
                "completed": False,
                "external_author_eval_completed": False,
            },
            "why_it_matters": (
                "Every retrieval number in the repo is in-sample.  Until an "
                "external author writes cases under the no-read protocol, "
                "generalisation cannot be claimed."
            ),
            "decision_taken": (
                "Runner enforces completed=false until external attestation lands; "
                "see docs/evals/no_read_rag_goldset_protocol.md."
            ),
            "what_was_not_claimed": [
                "external generalisation",
                "held-out validation",
                "out-of-sample readiness",
            ],
            "next_action": (
                "Volunteer reviewer authors 9+ cases under the no-read protocol."
            ),
            "clinical_validation": False,
        },
        {
            "title": "Source_filter_drop adjudication packet is draft only",
            "evidence_artifact": "Data/evals/rag/source_filter_drop_adjudication_packet.json",
            "metric_value": {
                "status": "ready_for_adjudication",
                "n_filled_decisions": 0,
                "n_draft_decisions": 9,
            },
            "why_it_matters": (
                "The 9 source_filter_drop cases need reviewer adjudication.  None "
                "has been filled.  No goldset edit has been applied."
            ),
            "decision_taken": (
                "Validator refuses to accept filled packet without notes/role; "
                "no auto-apply.  See docs/evals/rag_goldset_adjudication.md."
            ),
            "what_was_not_claimed": [
                "adjudication completion",
                "goldset correction applied",
            ],
            "next_action": (
                "Reviewer fills packet under attestation."
            ),
            "clinical_validation": False,
        },
        {
            "title": "No clinician review has been completed",
            "evidence_artifact": "docs/review_packets/INDEX.md",
            "metric_value": {
                "review_packets_prepared": 5,
                "filled_attestations": 0,
            },
            "why_it_matters": (
                "All medical-safety, genetic-counseling, and overtrust-mitigation "
                "claims rest on engineer-authored vocabulary.  No clinician, "
                "genetic counselor, or oncology nurse has filled a review packet."
            ),
            "decision_taken": (
                "Project rates medical_safety_boundaries at 6.5/10 in the "
                "10/10-under-constraints roadmap; the ceiling cannot move "
                "without a volunteer reviewer."
            ),
            "what_was_not_claimed": [
                "clinical sign-off",
                "clinician approval",
                "validated boundary vocabulary",
            ],
            "next_action": (
                "Engage one volunteer reviewer for one hour; see "
                "docs/review_packets/INDEX.md."
            ),
            "clinical_validation": False,
        },
    ]

    items = _mark_evidence_currency(items)
    return {
        "schema_version": "negative_results_gallery_v1",
        "status": "informational",
        "label": "negative_results_gallery",
        "clinical_validation": False,
        "claim_boundary": (
            "This gallery catalogues honest negative / non-promoted findings "
            "already documented elsewhere in the repo.  This artifact is "
            "engineering credibility scaffolding, not clinical validation.  "
            "Listing a negative result here does not retroactively make any "
            "system component clinically valid."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_items": len(items),
        "items": items,
    }


def write_negative_results_gallery(path: Path = NEGATIVE_RESULTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_negative_results_gallery(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "NEGATIVE_RESULTS_PATH",
    "build_negative_results_gallery",
    "write_negative_results_gallery",
]
