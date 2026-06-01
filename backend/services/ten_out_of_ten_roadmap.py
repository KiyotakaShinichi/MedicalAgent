"""10/10-under-constraints roadmap artifact.

Reading guide
~~~~~~~~~~~~~
"10/10 under constraints" means **as far as this project can credibly
go with no real patient data, no clinician sign-off, no IRB, and no
institutional partner**.  It does NOT mean clinically validated,
production healthcare ready, hospital deployable, or proven patient
benefit.

The dimension ``real_clinical_readiness`` is included specifically to
**stay low**.  Anyone reading the artifact must see that the project's
absolute clinical-readiness floor is dictated by the constraints, not
by the engineering effort.

The module is read-only.  No retrieval, ML, safety, or live-agent
behaviour is altered by this artifact; it is an engineering ratings
snapshot tied to the current repo state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/governance/latest_10_out_of_10_constraint_roadmap.json")


# ─── Dimension ratings ─────────────────────────────────────────────────


@dataclass
class DimensionRating:
    dimension: str
    side: str                          # AI / ML / Medical / SWE / Product
    current_score_out_of_10: float
    why_not_higher: str
    strongest_evidence: str
    weakest_evidence: str
    credibility_risk: str
    what_would_make_it_10_under_constraints: str
    what_cannot_be_solved_without_external_or_real_data_or_irb: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "side": self.side,
            "current_score_out_of_10": self.current_score_out_of_10,
            "why_not_higher": self.why_not_higher,
            "strongest_evidence": self.strongest_evidence,
            "weakest_evidence": self.weakest_evidence,
            "credibility_risk": self.credibility_risk,
            "what_would_make_it_10_under_constraints": self.what_would_make_it_10_under_constraints,
            "what_cannot_be_solved_without_external_or_real_data_or_irb": (
                self.what_cannot_be_solved_without_external_or_real_data_or_irb
            ),
        }


# Honest ratings.  Numbers are deliberately conservative: any number
# above 8/10 must be defensible from artifacts already in the repo.
# The real_clinical_readiness dimension is intentionally capped at
# 1.5/10 — the test suite enforces this.
DIMENSIONS: tuple[DimensionRating, ...] = (
    DimensionRating(
        dimension="ai_rag_architecture",
        side="ai_rag",
        current_score_out_of_10=8.0,
        why_not_higher=(
            "Live agent still uses heuristic claim validator by default; NLI is opt-in. "
            "Pruner experiment was a negative result and is not promoted."
        ),
        strongest_evidence=(
            "5 intent-aware RAG modes, source-governed retrieval, hybrid RRF, query "
            "rewriting, parent-child expansion, retrieval_confidence routing, per-turn "
            "trace with chain-of-thought scrub; documented in ADRs 0001-0009."
        ),
        weakest_evidence=(
            "Claim validator is heuristic token-overlap; held-out v2 RAG eval not "
            "completed; in-sample full-stack does not beat BM25 on raw recall."
        ),
        credibility_risk=(
            "Quoting in-sample full-stack recall as 'better than baseline' is "
            "overclaiming; the baseline comparison artifact explicitly says "
            "improvement_proven_vs_bm25 = false."
        ),
        what_would_make_it_10_under_constraints=(
            "Held-out v2 RAG eval completed by external author under no-read protocol; "
            "claim validator promoted to entailment-based as default with calibration; "
            "stage-wise oracle gap shrinks past goldset-adjudication outcome."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Real-world recall on actual patient queries; live multi-turn safety review "
            "by a clinician."
        ),
    ),
    DimensionRating(
        dimension="rag_evaluation_credibility",
        side="ai_rag",
        current_score_out_of_10=8.0,
        why_not_higher=(
            "Most numbers are in-sample on the 74-case internal goldset.  External "
            "no-read holdout v2 is prepared but not completed."
        ),
        strongest_evidence=(
            "improvement_proven_vs_bm25 = false reported honestly; pruner negative "
            "result documented; alias-coverage diagnostic; stage-wise oracle; "
            "citation-precision failure analysis; held-out v2 readiness artifact "
            "refuses fake completion."
        ),
        weakest_evidence=(
            "Citation_precision = 0.5243 (mediocre); held-out v2 has zero cases; "
            "alias additions were derived from the in-sample goldset."
        ),
        credibility_risk=(
            "Presenting alias-corrected post-2026-05-27 recall numbers without "
            "the BM25 figure next to them; that would inflate the apparent "
            "stack improvement."
        ),
        what_would_make_it_10_under_constraints=(
            "Held-out v2 goldset filled by external author under attestation; "
            "side-by-side BM25 vs full-stack on the held-out set; oracle gap "
            "explicitly recomputed."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Generalization to real patient queries from a real population; clinician "
            "agreement on what 'expected_source_ids' should be."
        ),
    ),
    DimensionRating(
        dimension="claim_validation_grounding",
        side="ai_rag",
        current_score_out_of_10=6.5,
        why_not_higher=(
            "Heuristic token-overlap is the default validator; NLI is opt-in via env "
            "var.  In-sample citation_precision is 0.5243."
        ),
        strongest_evidence=(
            "Claim-source alignment ledger; sentence-level claim status (supported / "
            "weakly_supported / unsupported); per-turn validator latency tracked."
        ),
        weakest_evidence=(
            "Citation precision 0.52 means ~half of cited chunks are not the expected "
            "gold sources; threshold-calibration sweep marked SUPPORTED_THRESHOLD as "
            "'soft_slope' rather than plateau."
        ),
        credibility_risk=(
            "Calling overlap-based 'support' a citation guarantee; the validator is "
            "*evidence of support*, not a proof of factual correctness."
        ),
        what_would_make_it_10_under_constraints=(
            "NLI validator on by default with calibration; citation_precision floor "
            "enforced post-adjudication; contradiction-trap eval expanded."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Real fact-checking on real-world generated answers; clinician judgement "
            "on contested claims."
        ),
    ),
    DimensionRating(
        dimension="agentic_workflow_safety",
        side="ai_rag",
        current_score_out_of_10=7.0,
        why_not_higher=(
            "Bounded tool surface is enforced but only by configuration; no kernel-"
            "level capability enforcement; multi-turn eval is in-sample."
        ),
        strongest_evidence=(
            "Bounded tool list, confirmation-before-write, forbidden tools blocked, "
            "multi-turn agent eval, agentic shadow-mode eval, adversarial tool-use eval."
        ),
        weakest_evidence=(
            "No external-author adversarial tool-use cases; trace stores decisions "
            "only (which is correct) but no real audit log of denied actions yet."
        ),
        credibility_risk=(
            "Treating in-sample agentic-workflow pass rates as proof of bounded-action "
            "safety in real workflows."
        ),
        what_would_make_it_10_under_constraints=(
            "External-author adversarial tool-use cases completed; per-action audit "
            "log with denial reasons; live-agent shadow trace fully wired with "
            "correlation_ids."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Real users attempting real edge cases; real workflow integration testing."
        ),
    ),
    DimensionRating(
        dimension="adversarial_safety",
        side="ai_rag",
        current_score_out_of_10=7.0,
        why_not_higher=(
            "Held-out v1 generalization was ~0.06 — the in-sample 1.0 is bank-tuned. "
            "Held-out v2 baseline is informational only."
        ),
        strongest_evidence=(
            "200-case stable-ID bank, 32 held-out v1 cases authored after the bank, "
            "anti-contamination test, four hardened categories at 1.0 in-sample with "
            "held-out reported separately."
        ),
        weakest_evidence=(
            "Held-out v1 0.06 is the truth-teller — generalization is poor.  Held-out "
            "v2 baseline shows the same gap pattern."
        ),
        credibility_risk=(
            "Quoting in-sample 1.0 without held-out 0.06 alongside; this would be "
            "memorisation, not generalisation."
        ),
        what_would_make_it_10_under_constraints=(
            "External-author adversarial cases (15+) authored under attestation; "
            "deterministic safety vocab extended only when held-out generalisation "
            "moves; entailment-based safety classifier."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Adversarial behaviour from real patients trying real prompts."
        ),
    ),
    DimensionRating(
        dimension="ml_mle_engineering",
        side="ml_mle",
        current_score_out_of_10=7.5,
        why_not_higher=(
            "Every numeric ML metric is saturated by synthetic-generator label "
            "consistency; the coverage floors are enforced but the headline numbers "
            "are not differentiating."
        ),
        strongest_evidence=(
            "Leakage audit, evidence-aware abstention, modality-dropout retraining, "
            "patient-level temporal CV with patient_overlap_pairs == 0, conformal "
            "calibration coverage >= 0.75, subgroup metrics promoted to required=true."
        ),
        weakest_evidence=(
            "Toxicity AUC ~1.0 acknowledged as structural leakage in the audit; "
            "patient-temporal CV AUC ~0.9996 for both protocols (saturation)."
        ),
        credibility_risk=(
            "Quoting a saturated AUC anywhere without the audit footnote next to it."
        ),
        what_would_make_it_10_under_constraints=(
            "Noisier synthetic v2 with label noise + missingness so numeric metrics "
            "differentiate; per-subgroup accuracy gated in release gate."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Real predictive validity on a real cohort."
        ),
    ),
    DimensionRating(
        dimension="ml_statistical_rigor",
        side="ml_mle",
        current_score_out_of_10=6.5,
        why_not_higher=(
            "Statistical evidence + robustness scripts exist; small-N caveats are "
            "documented; but multi-cohort prospective testing is not possible."
        ),
        strongest_evidence=(
            "ml_statistical_evidence.json and ml_statistical_robustness.json with CIs, "
            "paired model comparison with bootstrap, McNemar where appropriate."
        ),
        weakest_evidence=(
            "All comparisons sit on the same synthetic distribution; CIs are tight by "
            "construction."
        ),
        credibility_risk=(
            "Reading the tight CIs as evidence of model robustness rather than as "
            "evidence of synthetic homogeneity."
        ),
        what_would_make_it_10_under_constraints=(
            "Permutation tests across subgroups; multi-seed bootstrap; documented "
            "minimum-detectable-effect per metric."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Cross-cohort generalisation tests; prospective evaluation."
        ),
    ),
    DimensionRating(
        dimension="synthetic_data_governance",
        side="ml_mle",
        current_score_out_of_10=7.0,
        why_not_higher=(
            "Generator quality proxy is hand-curated sanity bands; not an empirical "
            "real-data comparison."
        ),
        strongest_evidence=(
            "synthetic_generator_quality_proxy artifact with enforced disclaimer; "
            "leakage audit; row-level prediction export manifest; promotion policy "
            "blocks clinical use."
        ),
        weakest_evidence=(
            "Quality proxy explicitly NOT a realism measure; missingness coverage and "
            "label-noise injection are not yet exercised."
        ),
        credibility_risk=(
            "Reading the quality proxy as evidence of realism."
        ),
        what_would_make_it_10_under_constraints=(
            "Noisier synthetic v2; documented generator changelog; label-noise eval."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Demonstrated distributional similarity to a real cohort."
        ),
    ),
    DimensionRating(
        dimension="external_data_readiness",
        side="ml_mle",
        current_score_out_of_10=6.0,
        why_not_higher=(
            "TCGA / METABRIC / BreastDCEDL bridges are stress tests on schema, not "
            "transfer learning."
        ),
        strongest_evidence=(
            "Canonical oncology schema; external_dataset_bridge_v2; common feature "
            "transfer stress; cBioPortal export; readiness checklist labelled "
            "'not_ready' honestly."
        ),
        weakest_evidence=(
            "No real-data scoring; no clinician-reviewed endpoints on external data."
        ),
        credibility_risk=(
            "Presenting schema readiness as transfer-learning readiness."
        ),
        what_would_make_it_10_under_constraints=(
            "External-author mapping review; FHIR conformance test against a sandbox; "
            "documented restricted-data-access packet (already exists)."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Actual data access; trained / scored models on the external cohort."
        ),
    ),
    DimensionRating(
        dimension="medical_safety_boundaries",
        side="medical",
        current_score_out_of_10=6.5,
        why_not_higher=(
            "Zero clinicians have reviewed any boundary template or refusal phrasing. "
            "Vocabulary is engineer-authored."
        ),
        strongest_evidence=(
            "Special-population boundary detector; medical claim boundary checker; "
            "multilingual safety vocab (EN + Taglish + Spanish); minimum evidence "
            "standards; advisor packet labelled 'unreviewed'."
        ),
        weakest_evidence=(
            "Genetic-counselor review packet exists but no genetic counselor has "
            "engaged; nurse safety review packet exists but no nurse has engaged."
        ),
        credibility_risk=(
            "Implying the boundary vocabulary has clinical authority because it is "
            "long."
        ),
        what_would_make_it_10_under_constraints=(
            "Volunteer clinician hour on refusal-template wording; volunteer genetic "
            "counselor hour on VUS handling; volunteer nurse hour on urgent triggers."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Regulatory framing (FDA / SaMD); real clinical sign-off."
        ),
    ),
    DimensionRating(
        dimension="human_factors_overtrust_mitigation",
        side="medical",
        current_score_out_of_10=6.5,
        why_not_higher=(
            "Banners and disclaimers exist and are tested; but no real-user usability "
            "test of overtrust under emotional load."
        ),
        strongest_evidence=(
            "Clinical boundary slim strip with full text always in DOM (ADR 0007); "
            "'Synthetic engineering signal · Not a clinical prediction' chips on KPI "
            "tiles; collapsible boundary banner with anti-overclaim wording."
        ),
        weakest_evidence=(
            "No real-user A/B of slim vs full banner; no eye-tracking; no patient "
            "interview."
        ),
        credibility_risk=(
            "Treating presence of disclaimer as evidence of correctly mitigated "
            "overtrust."
        ),
        what_would_make_it_10_under_constraints=(
            "5-user think-aloud usability test (engineering peers); documented "
            "qualitative overtrust failure modes; banner-blindness check."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Real-patient overtrust behaviour; longitudinal usability."
        ),
    ),
    DimensionRating(
        dimension="swe_architecture",
        side="swe",
        current_score_out_of_10=8.5,
        why_not_higher=(
            "Some modules carry getattr() degradation paths for not-yet-migrated "
            "columns; no admin-tier capability enforcement at OS level."
        ),
        strongest_evidence=(
            "15-module split of agent_rag.py (ADR 0001); FAST_MODE escape hatch "
            "(ADR 0003); no-CoT trace contract (ADR 0004); release-gate tier "
            "structure (ADR 0006); composer attachment popover (ADR 0008)."
        ),
        weakest_evidence=(
            "Frontend index.css is single ~71kB file; dead components were dropped "
            "but the next sweep is overdue."
        ),
        credibility_risk=(
            "Equating 'lots of ADRs' with 'well-architected'; ADRs document decisions, "
            "they don't validate them."
        ),
        what_would_make_it_10_under_constraints=(
            "CSS split; capability log written through `validate_trace_payload`; "
            "explicit deployment-readiness boundary doc."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Real production traffic shape; HIPAA / PHI architecture review."
        ),
    ),
    DimensionRating(
        dimension="ci_release_discipline",
        side="swe",
        current_score_out_of_10=8.0,
        why_not_higher=(
            "121-artifact release gate; with informational tiering it's manageable, "
            "but the long tail risks paperwork dilution."
        ),
        strongest_evidence=(
            "release_gate_thresholds.yaml with tiers; ship.py; eval drift tracker "
            "with regression detection; per-artifact accepted_status + "
            "max_age_days + metric_thresholds."
        ),
        weakest_evidence=(
            "Tier semantics are encoded in status fields + required flags rather "
            "than an explicit blocker/warn/informational column."
        ),
        credibility_risk=(
            "Reading 'gate passed' as deployment readiness rather than as engineering "
            "readiness."
        ),
        what_would_make_it_10_under_constraints=(
            "Explicit tier column in YAML; per-tier failure-counts in the printout; "
            "trim long tail."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Production CI/CD into an actual healthcare environment."
        ),
    ),
    DimensionRating(
        dimension="observability_traceability",
        side="swe",
        current_score_out_of_10=8.0,
        why_not_higher=(
            "Per-turn trace persists only on RAG path; cache and tool-use surfaces "
            "have correlation_ids but not full trace envelopes."
        ),
        strongest_evidence=(
            "agent_turn_trace with CoT deny-list; validate_trace_payload; "
            "trace_diagnostics_coverage artifact; runtime quality sentinel."
        ),
        weakest_evidence=(
            "trace_diagnostics_coverage is currently 'needs_attention' status; "
            "non-RAG turns have no equivalent envelope."
        ),
        credibility_risk=(
            "Calling per-turn trace 'auditable' without acknowledging the no-CoT "
            "policy is a contract, not a regulatory artifact."
        ),
        what_would_make_it_10_under_constraints=(
            "Trace coverage at 'strong'; non-RAG turn envelopes wired; trace replay "
            "admin UI fully populated."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Real audit logs with real patient consent; SOC 2 / HITRUST."
        ),
    ),
    DimensionRating(
        dimension="product_reviewer_credibility",
        side="product",
        current_score_out_of_10=7.0,
        why_not_higher=(
            "Review packets exist for 5 reviewer roles; no reviewer has engaged."
        ),
        strongest_evidence=(
            "Review packet INDEX with reviewer-time budget; nurse / genetic-counselor "
            "/ senior MLE / external-author / agentic-workflow packets; ADR folder; "
            "negative-result documentation (pruner, held-out 0.06)."
        ),
        weakest_evidence=(
            "Zero filled attestations in Data/evals/external_review/; goldset "
            "adjudication packet has 0 filled decisions."
        ),
        credibility_risk=(
            "Presenting 'reviewer packets prepared' as 'reviewer engagement'."
        ),
        what_would_make_it_10_under_constraints=(
            "One filled adjudication; one filled external-author attestation; one "
            "completed held-out v2 RAG eval."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Clinical authority for any system output."
        ),
    ),
    DimensionRating(
        dimension="overall_portfolio_strength",
        side="portfolio",
        current_score_out_of_10=7.5,
        why_not_higher=(
            "Breadth is real; every layer has a negative result honestly reported; "
            "but no external engagement has happened yet."
        ),
        strongest_evidence=(
            "12 ADRs; 8 review packet templates; 121 release-gate artifacts with "
            "tier semantics; negative results visible (pruner -9.7pp CP; held-out "
            "0.06; oracle gap 0.054)."
        ),
        weakest_evidence=(
            "Every honest negative still sits inside an in-sample frame."
        ),
        credibility_risk=(
            "Volume mistaken for validity."
        ),
        what_would_make_it_10_under_constraints=(
            "Held-out v2 done; one clinician hour; one filled adjudication; "
            "negative-results gallery doc."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Patient benefit; real-world safety; productionisation."
        ),
    ),
    DimensionRating(
        dimension="real_clinical_readiness",
        side="real_clinical",
        # Hard cap.  This dimension is the project's floor and the test
        # suite asserts <= 2.0.
        current_score_out_of_10=1.5,
        why_not_higher=(
            "No real patient data, no clinician sign-off, no IRB, no institutional "
            "partner, no validated clinical labels, no clinical evidence at all."
        ),
        strongest_evidence=(
            "Honest 'NOT validated' banners; production_readiness_boundary status "
            "'production_shaped_not_healthcare_production_ready'; every artifact "
            "carries clinical_validation: false."
        ),
        weakest_evidence=(
            "Everything else.  The project is engineering scaffolding for a future "
            "clinical evaluation, not the clinical evaluation itself."
        ),
        credibility_risk=(
            "Any sentence in the README, dashboard, or eval artifact that elides "
            "the 'engineering only' framing."
        ),
        what_would_make_it_10_under_constraints=(
            "Nothing — this dimension is bounded by what *cannot* be done under the "
            "constraints.  The 10/10 framing does NOT apply here."
        ),
        what_cannot_be_solved_without_external_or_real_data_or_irb=(
            "Literally everything: real patient data, IRB, clinician sign-off, "
            "prospective evaluation, regulatory clearance, real-world safety, "
            "patient benefit."
        ),
    ),
)


# ─── Highest-ROI ranked roadmap ────────────────────────────────────────


@dataclass
class RoadmapItem:
    rank: int
    item: str
    tier: str                          # A_implement_now | B_external_reviewer | C_real_data | D_irb_institution
    impact: str
    difficulty: str
    overclaim_risk: str
    controllable_now: bool
    artifact_when_done: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "item": self.item,
            "tier": self.tier,
            "impact": self.impact,
            "difficulty": self.difficulty,
            "overclaim_risk": self.overclaim_risk,
            "controllable_now": self.controllable_now,
            "artifact_when_done": self.artifact_when_done,
        }


ROADMAP_ITEMS: tuple[RoadmapItem, ...] = (
    RoadmapItem(
        rank=1, item="Negative results gallery doc",
        tier="A_implement_now", impact="high", difficulty="low",
        overclaim_risk="none — explicit anti-overclaim",
        controllable_now=True,
        artifact_when_done="docs/negative_results.md",
    ),
    RoadmapItem(
        rank=2, item="10/10-under-constraints roadmap artifact (this pass)",
        tier="A_implement_now", impact="high", difficulty="low",
        overclaim_risk="medium — must enforce cap on real_clinical_readiness",
        controllable_now=True,
        artifact_when_done="Data/evals/governance/latest_10_out_of_10_constraint_roadmap.json",
    ),
    RoadmapItem(
        rank=3, item="CV / portfolio safe-wording template",
        tier="A_implement_now", impact="medium", difficulty="low",
        overclaim_risk="low — template enforces explicit anti-clinical wording",
        controllable_now=True,
        artifact_when_done="docs/portfolio_safe_wording_template.md",
    ),
    RoadmapItem(
        rank=4, item="Held-out v2 RAG eval completed by external author",
        tier="B_external_reviewer", impact="very_high", difficulty="medium",
        overclaim_risk="low if author follows no-read protocol",
        controllable_now=False,
        artifact_when_done="Data/evals/rag/retrieval_goldset_holdout_v2.jsonl + completed=true comparison",
    ),
    RoadmapItem(
        rank=5, item="Source_filter_drop adjudication completed (9 cases)",
        tier="B_external_reviewer", impact="high", difficulty="low",
        overclaim_risk="low",
        controllable_now=False,
        artifact_when_done="Filled Data/evals/rag/source_filter_drop_adjudication_packet.json",
    ),
    RoadmapItem(
        rank=6, item="External-author adversarial cases (15+)",
        tier="B_external_reviewer", impact="high", difficulty="medium",
        overclaim_risk="low if held-out from training",
        controllable_now=False,
        artifact_when_done="Filled Data/evals/safety/external_author_adversarial_template.jsonl",
    ),
    RoadmapItem(
        rank=7, item="Clinician hour on refusal templates + urgent triggers",
        tier="B_external_reviewer", impact="very_high", difficulty="low",
        overclaim_risk="medium — must not be presented as clinical sign-off",
        controllable_now=False,
        artifact_when_done="Filled docs/review_packets/nurse_or_clinician_safety_review_packet.md",
    ),
    RoadmapItem(
        rank=8, item="Genetic counselor hour on VUS handling",
        tier="B_external_reviewer", impact="high", difficulty="low",
        overclaim_risk="medium — must not be presented as clinical sign-off",
        controllable_now=False,
        artifact_when_done="Filled docs/review_packets/genetic_counselor_vus_review_packet.md",
    ),
    RoadmapItem(
        rank=9, item="Noisier synthetic v2 (label noise + missingness)",
        tier="A_implement_now", impact="medium", difficulty="medium",
        overclaim_risk="low",
        controllable_now=True,
        artifact_when_done="Data/synthetic_v2/* + noise_eval extended",
    ),
    RoadmapItem(
        rank=10, item="Eval contamination registry harmonization",
        tier="A_implement_now", impact="medium", difficulty="low",
        overclaim_risk="none",
        controllable_now=True,
        artifact_when_done="Data/evals/governance/latest_eval_contamination_registry.json updated",
    ),
    RoadmapItem(
        rank=11, item="Real cohort external dataset bridge",
        tier="C_real_data", impact="very_high", difficulty="high",
        overclaim_risk="very_high",
        controllable_now=False,
        artifact_when_done="Schema mapping + restricted-access packet + signed DUA",
    ),
    RoadmapItem(
        rank=12, item="Real-world latency / cost shape",
        tier="C_real_data", impact="medium", difficulty="medium",
        overclaim_risk="low",
        controllable_now=False,
        artifact_when_done="Production-traffic latency profile",
    ),
    RoadmapItem(
        rank=13, item="Real recall@10 / citation_precision on real queries",
        tier="C_real_data", impact="very_high", difficulty="high",
        overclaim_risk="medium",
        controllable_now=False,
        artifact_when_done="Real-query baseline comparison artifact",
    ),
    RoadmapItem(
        rank=14, item="IRB submission for any patient-facing experiment",
        tier="D_irb_institution", impact="very_high", difficulty="very_high",
        overclaim_risk="not_applicable",
        controllable_now=False,
        artifact_when_done="IRB approval letter (out of repo scope)",
    ),
    RoadmapItem(
        rank=15, item="Clinician oversight committee / sign-off process",
        tier="D_irb_institution", impact="very_high", difficulty="very_high",
        overclaim_risk="not_applicable",
        controllable_now=False,
        artifact_when_done="Signed clinical advisory charter (out of repo scope)",
    ),
)


# ─── Build / write ──────────────────────────────────────────────────────


# Anti-overclaim disclaimer tokens that MUST appear (tests assert this).
REQUIRED_ANTI_OVERCLAIM_TOKENS: tuple[str, ...] = (
    "not clinical validation",
    "not production healthcare ready",
    "no clinician sign-off",
    "no irb",
    "no real patient data",
)


def build_roadmap() -> dict[str, Any]:
    dimensions = [d.to_dict() for d in DIMENSIONS]
    roadmap = [r.to_dict() for r in ROADMAP_ITEMS]
    avg = round(
        sum(d["current_score_out_of_10"] for d in dimensions if d["dimension"] != "real_clinical_readiness")
        / max(1, len([d for d in dimensions if d["dimension"] != "real_clinical_readiness"])),
        2,
    )
    expected_after = 8.0  # estimated weighted average after the safe A items below land
    return {
        "schema_version": "10_out_of_10_constraint_roadmap_v1",
        "status": "informational",
        "label": "10_out_of_10_under_constraints_roadmap",
        "clinical_validation": False,
        "claim_boundary": (
            "10/10 under current constraints is NOT clinical validation, is NOT "
            "production healthcare ready, is NOT hospital deployable, and is NOT "
            "proven patient benefit.  This artifact is engineering self-rating only.  "
            "No clinician sign-off, no IRB, no real patient data.  See "
            "`real_clinical_readiness` for the dimension that captures the floor "
            "created by these constraints."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "average_score_excluding_real_clinical_readiness": avg,
            "real_clinical_readiness_score": next(
                d["current_score_out_of_10"]
                for d in dimensions
                if d["dimension"] == "real_clinical_readiness"
            ),
            "expected_avg_after_a_items_land": expected_after,
            "n_dimensions": len(dimensions),
            "n_roadmap_items": len(roadmap),
            "n_implement_now": sum(1 for r in roadmap if r["tier"] == "A_implement_now"),
            "n_needs_external_reviewer": sum(1 for r in roadmap if r["tier"] == "B_external_reviewer"),
            "n_needs_real_data": sum(1 for r in roadmap if r["tier"] == "C_real_data"),
            "n_needs_irb_institution": sum(1 for r in roadmap if r["tier"] == "D_irb_institution"),
        },
        "dimensions": dimensions,
        "ranked_roadmap": roadmap,
        "anti_overclaim_invariants": list(REQUIRED_ANTI_OVERCLAIM_TOKENS),
        "things_we_cannot_claim_under_constraints": [
            "clinical validation",
            "real-world safety guarantee",
            "patient benefit",
            "diagnostic, treatment, or prognostic authority",
            "clinician sign-off",
            "IRB / ethics approval",
            "regulatory clearance",
            "production healthcare readiness",
            "generalisation from synthetic to real patients",
        ],
    }


def write_roadmap(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    payload = build_roadmap()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DIMENSIONS",
    "REQUIRED_ANTI_OVERCLAIM_TOKENS",
    "ROADMAP_ITEMS",
    "build_roadmap",
    "write_roadmap",
]
