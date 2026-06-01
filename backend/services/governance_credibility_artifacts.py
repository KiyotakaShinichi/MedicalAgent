"""Four governance/credibility artifacts.

Each artifact is pure-data: a single function builds the JSON payload,
a sibling write_* function persists it to disk.  No retrieval, ML,
safety, or live-agent behaviour is changed.  Every artifact carries
``clinical_validation: false`` and is gated as informational in the
release gate.

Artifacts
~~~~~~~~~
1. ``build_negative_results_gallery`` — explicit catalogue of negative
   / non-promoted findings already documented elsewhere in the repo.
2. ``build_portfolio_claim_safety_check`` — banned-phrase / allowed-
   phrase guardrails for CV, LinkedIn, README, recruiter, and
   senior-engineer wording.
3. ``build_eval_contamination_harmonization`` — maps every eval
   artifact category (used-for-tuning / frozen / external / synthetic
   / live / informational) and assigns allowed claim strength.
4. ``build_noisier_synthetic_v2_readiness`` — readiness scaffold for
   a noisier synthetic-data v2.  Status is ``scaffold_only``; no
   model is retrained, no clinical behaviour changes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ─── Output paths ────────────────────────────────────────────────────────

NEGATIVE_RESULTS_PATH = Path("Data/evals/governance/latest_negative_results_gallery.json")
PORTFOLIO_PATH = Path("Data/evals/governance/latest_portfolio_claim_safety_check.json")
CONTAMINATION_PATH = Path("Data/evals/governance/latest_eval_contamination_harmonization.json")
NOISIER_V2_PATH = Path("Data/evals/models/latest_noisier_synthetic_v2_readiness.json")


# Anti-overclaim invariant: every artifact's claim_boundary must
# include the verbatim phrase "not clinical validation" (lowercase
# match in tests).  Tests enforce this and an additional list of
# banned-phrase patterns for the portfolio artifact.
REQUIRED_CLAIM_BOUNDARY_PHRASE = "not clinical validation"


# ─── 1. Negative-results gallery ─────────────────────────────────────────


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


# ─── 2. Portfolio / CV safe-wording template ─────────────────────────────


# Banned phrases — must NOT appear as bare affirmative claims in the
# portfolio/CV doc.  The test suite enforces this on the artifact AND
# the companion markdown.
BANNED_AFFIRMATIVE_PHRASES: tuple[str, ...] = (
    "clinically validated",
    "production healthcare ready",
    "patient benefit",
    "diagnostic system",
    "treatment recommender",
    "proven safe",
    "clinician-approved",
    "fhir compliant",
    "hospital interoperable",
    "fda approved",
    "fda cleared",
    "ce marked",
    "hipaa compliant",
    "real-world evidence",
)

ALLOWED_PHRASES: tuple[str, ...] = (
    "engineering prototype",
    "synthetic-only ML signals",
    "not clinically validated",
    "non-diagnostic",
    "monitor-only",
    "intended for clinician review",
    "source-governed retrieval",
    "claim-level citation validation",
    "release-gate-enforced",
    "in-sample only",
    "improvement not proven",
    "informational artifact only",
)


def build_portfolio_claim_safety_check() -> dict[str, Any]:
    samples = [
        {
            "audience": "linkedin_one_line",
            "safe_version": (
                "Built an engineering prototype of a safety-first breast cancer "
                "monitoring agent with source-governed RAG, claim-level citation "
                "validation, and release-gate-enforced negative-result reporting; "
                "synthetic-only data, not clinically validated."
            ),
            "unsafe_version": (
                "Built a clinically validated, production-ready AI doctor that "
                "diagnoses breast cancer using FHIR-compliant patient data."
            ),
            "why_unsafe": (
                "claims clinical validation, diagnosis authority, FHIR compliance, "
                "and production readiness — none of which are true"
            ),
        },
        {
            "audience": "recruiter_short",
            "safe_version": (
                "Designed and shipped a synthetic-data oncology monitoring agent: "
                "hybrid RAG, source-tier governance, adversarial safety bank with "
                "held-out generalisation reported honestly, and a 120-artifact "
                "release gate with explicit anti-overclaim tests."
            ),
            "unsafe_version": (
                "Built an AI cancer agent that improves patient outcomes and "
                "supports clinical decision-making in hospitals."
            ),
            "why_unsafe": (
                "asserts patient outcomes and clinical decision support without any "
                "real-data evidence or clinician sign-off"
            ),
        },
        {
            "audience": "senior_engineer_technical",
            "safe_version": (
                "RAG architecture with 5 intent-aware source-governed modes, "
                "hybrid dense+sparse RRF, query rewriting, claim-level citation "
                "validation (heuristic by default, NLI opt-in), uncertainty-aware "
                "answerability routing, per-turn trace with chain-of-thought "
                "deny-list, stage-wise oracle diagnostic.  Source-governed stack "
                "does not exceed BM25 on raw recall on the in-sample goldset; "
                "negative results documented; held-out v2 prepared but not "
                "completed."
            ),
            "unsafe_version": (
                "RAG architecture that outperforms baselines on retrieval and is "
                "clinically validated for oncology decision support."
            ),
            "why_unsafe": (
                "'outperforms baselines' is false on the frozen goldset; clinical "
                "validation has not happened"
            ),
        },
        {
            "audience": "readme_summary_paragraph",
            "safe_version": (
                "MedicalAgent is a safety-first, non-diagnostic breast cancer "
                "monitoring engineering prototype.  It combines source-governed "
                "dense/sparse RAG, claim-level citation validation, deterministic "
                "pre-generation safety gates, adversarial safety regression with "
                "held-out generalisation reporting, and release-gate-enforced "
                "negative-result publication.  All ML signals are synthetic and "
                "monitor-only.  No clinician sign-off, no IRB, no real patient "
                "data."
            ),
            "unsafe_version": (
                "MedicalAgent is a clinically validated breast cancer monitoring "
                "system used in hospitals to improve patient outcomes."
            ),
            "why_unsafe": (
                "every clause is unverifiable under current constraints"
            ),
        },
        {
            "audience": "cv_bullet",
            "safe_version": (
                "Engineering prototype of a non-diagnostic oncology monitoring "
                "agent on synthetic data; documented negative results (pruner "
                "regression, held-out adversarial gap, full-stack not exceeding "
                "BM25); test-locked anti-overclaim invariants."
            ),
            "unsafe_version": (
                "Shipped clinically validated AI for breast cancer diagnosis used "
                "by oncologists."
            ),
            "why_unsafe": (
                "no clinician engagement, no validation, no diagnostic authority"
            ),
        },
    ]

    return {
        "schema_version": "portfolio_claim_safety_check_v1",
        "status": "informational",
        "label": "portfolio_claim_safety_check",
        "clinical_validation": False,
        "claim_boundary": (
            "This artifact is wording guidance only.  It is not clinical "
            "validation, not clinician sign-off, not IRB approval, and not "
            "proof of patient benefit.  The samples below are templates the "
            "project owner can adapt while staying inside the project's hard "
            "constraints (synthetic-only, no clinician, no IRB, no real patient "
            "data)."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "banned_affirmative_phrases": list(BANNED_AFFIRMATIVE_PHRASES),
        "allowed_phrases": list(ALLOWED_PHRASES),
        "audience_samples": samples,
        "guidance": (
            "If a sentence about the project would be false to say in a courtroom "
            "or in front of a regulator, it must not be said in a CV, README, "
            "LinkedIn, or recruiter blurb either.  When in doubt, use 'engineering "
            "prototype' and 'synthetic-only'."
        ),
    }


def write_portfolio_claim_safety_check(path: Path = PORTFOLIO_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_portfolio_claim_safety_check(), indent=2), encoding="utf-8")
    return path


# ─── 3. Contamination harmonisation ──────────────────────────────────────


_HARMONISATION_CATEGORIES = (
    "internal_used_for_tuning",
    "internal_frozen_not_used_for_tuning",
    "external_no_read_prepared_incomplete",
    "external_completed",
    "synthetic_generated",
    "live_agent_internal",
    "informational_only",
)


def build_eval_contamination_harmonization() -> dict[str, Any]:
    # Hand-curated map: each eval artifact assigned to one harmonisation
    # category with allowed claim strength and release-gate tier.  Pulled
    # from the eval landscape that already exists in the repo.
    rows = [
        {
            "path": "Data/evals/rag/retrieval_goldset.jsonl",
            "authorship": "engineering_internal",
            "was_used_for_tuning": True,
            "frozen_status": "frozen_2026_05",
            "contamination_risk": "high",
            "allowed_claim_strength": "in-sample engineering signal only",
            "release_gate_tier": "informational_blocker_for_freshness",
            "harmonisation_category": "internal_used_for_tuning",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/rag/latest_rag_baseline_comparison.json",
            "authorship": "engineering_internal",
            "was_used_for_tuning": False,
            "frozen_status": "regenerated_per_release",
            "contamination_risk": "high",
            "allowed_claim_strength": "in-sample comparison; improvement_proven_vs_bm25=false",
            "release_gate_tier": "blocker",
            "harmonisation_category": "internal_used_for_tuning",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl",
            "authorship": "engineering_internal_template",
            "was_used_for_tuning": False,
            "frozen_status": "template_only",
            "contamination_risk": "n_a_template",
            "allowed_claim_strength": "template; not an eval set",
            "release_gate_tier": "informational",
            "harmonisation_category": "external_no_read_prepared_incomplete",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/rag/latest_rag_holdout_baseline_comparison.json",
            "authorship": "pending_external",
            "was_used_for_tuning": False,
            "frozen_status": "not_yet_authored",
            "contamination_risk": "low_when_completed",
            "allowed_claim_strength": "external generalisation (only after completed=true)",
            "release_gate_tier": "informational",
            "harmonisation_category": "external_no_read_prepared_incomplete",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/safety/adversarial_safety_regression_bank.jsonl",
            "authorship": "engineering_internal",
            "was_used_for_tuning": True,
            "frozen_status": "stable_ids",
            "contamination_risk": "high_post_2026_05_20_hardening",
            "allowed_claim_strength": "bank-tuned in-sample only",
            "release_gate_tier": "warn_on_regression",
            "harmonisation_category": "internal_used_for_tuning",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/safety/adversarial_safety_holdout_variants.jsonl",
            "authorship": "engineering_internal_post_hardening",
            "was_used_for_tuning": False,
            "frozen_status": "post_hardening_holdout_v1",
            "contamination_risk": "low",
            "allowed_claim_strength": "honest generalisation signal (held-out v1)",
            "release_gate_tier": "informational",
            "harmonisation_category": "internal_frozen_not_used_for_tuning",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/rag/latest_rag_stage_oracle_diagnostic.json",
            "authorship": "engineering_internal_diagnostic",
            "was_used_for_tuning": False,
            "frozen_status": "regenerated_per_release",
            "contamination_risk": "n_a_diagnostic",
            "allowed_claim_strength": "stage-wise attribution; not a score claim",
            "release_gate_tier": "informational",
            "harmonisation_category": "informational_only",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/rag/latest_citation_precision_failure_analysis.json",
            "authorship": "engineering_internal_diagnostic",
            "was_used_for_tuning": False,
            "frozen_status": "regenerated_per_release",
            "contamination_risk": "n_a_diagnostic",
            "allowed_claim_strength": "categorisation; not a score claim",
            "release_gate_tier": "informational",
            "harmonisation_category": "informational_only",
            "clinical_validation": False,
        },
        {
            "path": "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv",
            "authorship": "synthetic_generator",
            "was_used_for_tuning": True,
            "frozen_status": "frozen_per_release",
            "contamination_risk": "structural_label_leakage_documented",
            "allowed_claim_strength": "synthetic distribution only; not clinical",
            "release_gate_tier": "audit_footnote_required",
            "harmonisation_category": "synthetic_generated",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/realism/latest_synthetic_data_quality.json",
            "authorship": "engineering_internal",
            "was_used_for_tuning": False,
            "frozen_status": "regenerated_per_release",
            "contamination_risk": "n_a_quality_proxy",
            "allowed_claim_strength": "internal generator quality proxy; NOT realism",
            "release_gate_tier": "informational",
            "harmonisation_category": "synthetic_generated",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/rag/latest_live_rag_eval.json",
            "authorship": "live_agent_internal",
            "was_used_for_tuning": False,
            "frozen_status": "captured_per_release",
            "contamination_risk": "live_internal_only",
            "allowed_claim_strength": "live-agent internal behaviour signal",
            "release_gate_tier": "blocker",
            "harmonisation_category": "live_agent_internal",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/governance/latest_10_out_of_10_constraint_roadmap.json",
            "authorship": "engineering_internal_self_rating",
            "was_used_for_tuning": False,
            "frozen_status": "regenerated_per_release",
            "contamination_risk": "n_a_self_rating",
            "allowed_claim_strength": "engineering self-rating only",
            "release_gate_tier": "informational",
            "harmonisation_category": "informational_only",
            "clinical_validation": False,
        },
        {
            "path": "Data/evals/rag/source_filter_drop_adjudication_packet.json",
            "authorship": "engineering_internal_draft",
            "was_used_for_tuning": False,
            "frozen_status": "draft_packet",
            "contamination_risk": "n_a_draft",
            "allowed_claim_strength": "reviewer workflow only",
            "release_gate_tier": "informational",
            "harmonisation_category": "informational_only",
            "clinical_validation": False,
        },
    ]

    category_counts: dict[str, int] = {c: 0 for c in _HARMONISATION_CATEGORIES}
    for row in rows:
        category_counts[row["harmonisation_category"]] = category_counts.get(row["harmonisation_category"], 0) + 1

    return {
        "schema_version": "eval_contamination_harmonization_v1",
        "status": "informational",
        "label": "eval_contamination_harmonization",
        "clinical_validation": False,
        "claim_boundary": (
            "Harmonisation map of eval-artifact contamination categories.  Does "
            "not change any artifact's content or any release-gate decision.  "
            "Engineering credibility scaffolding only.  Not clinical validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "categories": list(_HARMONISATION_CATEGORIES),
        "category_counts": category_counts,
        "n_artifacts_mapped": len(rows),
        "artifacts": rows,
        "guidance": (
            "An artifact's 'allowed_claim_strength' is the strongest reading a "
            "reviewer should give it.  Pushing any artifact past its category "
            "(e.g. citing an 'internal_used_for_tuning' number as 'external "
            "generalisation') is overclaiming, regardless of how green the "
            "metric looks."
        ),
    }


def write_eval_contamination_harmonization(path: Path = CONTAMINATION_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_eval_contamination_harmonization(), indent=2), encoding="utf-8")
    return path


# ─── 4. Noisier synthetic v2 readiness scaffold ──────────────────────────


# Hard cap: this artifact's status field MUST be "scaffold_only" until
# a deliberate decision is made to start data generation.  The test
# suite enforces this.
ALLOWED_NOISIER_V2_STATUS: frozenset[str] = frozenset({
    "scaffold_only",
    "planned_not_trained",
})


def build_noisier_synthetic_v2_readiness() -> dict[str, Any]:
    noise_types = [
        {
            "name": "missingness_noise",
            "rationale": "Real cohorts have missing labs/imaging; current synthetic data has none.",
            "planned_distribution": "Bernoulli(p=0.1-0.3) per modality per cycle, with patient-block structure.",
        },
        {
            "name": "label_noise",
            "rationale": "Real outcome labels disagree across reviewers; current synthetic labels are deterministic.",
            "planned_distribution": "Symmetric noise rate eta in {0.05, 0.10, 0.15} for binary outcomes.",
        },
        {
            "name": "measurement_noise",
            "rationale": "Lab values have analytical variance; current synthetic values are exact.",
            "planned_distribution": "Multiplicative log-normal noise calibrated to assay CV bands.",
        },
        {
            "name": "date_jitter",
            "rationale": "Real records have +/- a few days of date drift around treatment events.",
            "planned_distribution": "Uniform jitter +/- 3 days per event, preserving ordering.",
        },
        {
            "name": "symptom_reporting_noise",
            "rationale": "Patient-reported severity is bursty and inconsistent.",
            "planned_distribution": "Per-patient over/under-reporting bias drawn once per patient.",
        },
        {
            "name": "imaging_report_ambiguity",
            "rationale": "Imaging reports have hedged language; current synthetic reports are crisp.",
            "planned_distribution": "Hedge-word injection rate in {0.1, 0.2}; impression vs body separation preserved.",
        },
        {
            "name": "treatment_delay_randomness",
            "rationale": "Real chemotherapy cycles slip due to non-clinical reasons.",
            "planned_distribution": "Per-cycle delay ~ Geometric(p) with p tuned to median 0 delay, p95 ~7 days.",
        },
        {
            "name": "subgroup_distribution_shift",
            "rationale": "Synthetic cohort is balanced by construction; real subgroups are not.",
            "planned_distribution": "Reweight subgroup priors per release using documented prior shifts.",
        },
    ]

    blocked_claims = [
        "this synthetic v2 represents real patients",
        "this synthetic v2 establishes clinical performance",
        "this synthetic v2 is FDA / IRB ready",
        "this synthetic v2 is sufficient for deployment",
        "this synthetic v2 replaces real-data validation",
    ]

    expected_evals_before_promotion = [
        "leakage audit re-run with patient-level temporal CV under noise",
        "subgroup metrics re-run under each noise type independently",
        "calibration + conformal coverage under noise",
        "shortcut audit re-run; toxicity AUC must drop below saturation",
        "synthetic data quality proxy with v2-specific disclaimer text",
        "release gate must continue to PASS with v2 artifacts at status: informational",
        "no metric promoted from monitor-only to treatment-influence",
    ]

    return {
        "schema_version": "noisier_synthetic_v2_readiness_v1",
        "status": "informational",
        "scaffold_status": "scaffold_only",
        "label": "noisier_synthetic_v2_readiness",
        "clinical_validation": False,
        "claim_boundary": (
            "Readiness scaffold only.  No noisier synthetic v2 data has been "
            "generated, no model has been retrained, and no live-agent behaviour "
            "has been changed by this artifact.  This is engineering planning "
            "infrastructure; it is not clinical validation, real-world readiness, "
            "or any kind of model promotion."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "why_current_synthetic_data_is_too_clean": (
            "Current temporal_ml_rows.csv has deterministic labels, no missingness, "
            "no measurement noise, no date jitter, no reporting bias, and a "
            "balanced subgroup distribution.  This saturates every metric in the "
            "MLE stack (toxicity AUC ~1.0, patient-temporal CV AUC ~0.9996) and "
            "prevents the ML and statistical-rigor dimensions of the "
            "10/10-under-constraints roadmap from moving."
        ),
        "planned_noise_types": noise_types,
        "blocked_clinical_claims": blocked_claims,
        "expected_evals_before_promotion": expected_evals_before_promotion,
        "why_this_remains_synthetic_only": (
            "Noisier synthetic v2 still has no real patient data, no clinician-"
            "reviewed labels, and no IRB.  It improves the *measurement surface* "
            "by removing saturation; it does NOT close the gap to real data."
        ),
    }


def write_noisier_synthetic_v2_readiness(path: Path = NOISIER_V2_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_noisier_synthetic_v2_readiness(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "ALLOWED_NOISIER_V2_STATUS",
    "ALLOWED_PHRASES",
    "BANNED_AFFIRMATIVE_PHRASES",
    "CONTAMINATION_PATH",
    "NEGATIVE_RESULTS_PATH",
    "NOISIER_V2_PATH",
    "PORTFOLIO_PATH",
    "REQUIRED_CLAIM_BOUNDARY_PHRASE",
    "build_eval_contamination_harmonization",
    "build_negative_results_gallery",
    "build_noisier_synthetic_v2_readiness",
    "build_portfolio_claim_safety_check",
    "write_eval_contamination_harmonization",
    "write_negative_results_gallery",
    "write_noisier_synthetic_v2_readiness",
    "write_portfolio_claim_safety_check",
]
