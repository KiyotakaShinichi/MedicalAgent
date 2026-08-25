"""Consolidated failure-mode registry.

Pulls together every place the system already records how things can go
wrong:

  - `failure_case_gallery` — hand-authored narrative cases
  - `safety_red_team` — automated adversarial / urgent / refusal cases
  - `drift_report` — distribution shift findings
  - hardcoded "engineering risks" — things we know about the synthetic
    pipeline that aren't covered by any one artifact

Output schema (per entry):

  {
    "name":             "prompt_injection_to_treatment_advice",
    "category":         "adversarial",
    "example":          "...",
    "risk":             "Model could leak treatment advice when prompted",
    "detection":        "safety_red_team",
    "mitigation":       "Deterministic input + output guardrails",
    "benchmark_coverage": ["safety_red_team", "rag_eval"],
    "remaining_gap":    "Real-world prompt variations not yet enumerated",
    "severity":         "high",
  }

The registry is engineering provenance, not clinical risk assessment.
A populated registry is the floor for a defensible portfolio — it shows
the system knows where its blind spots are.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FAILURE_GALLERY_PATH = "Data/reports/failure_case_gallery.json"
DEFAULT_SAFETY_RED_TEAM_PATH = "Data/evals/safety/latest_safety_red_team.json"
DEFAULT_DRIFT_REPORT_PATH = "Data/mle_monitoring/latest_drift_report.json"
DEFAULT_OUTPUT_PATH = "Data/evals/safety/latest_failure_mode_registry.json"


# Hand-curated engineering risks not directly covered by any single eval.
# These describe known limitations of the *generator* and *pipeline* itself.
# Each entry must list at least one benchmark in `benchmark_coverage` that
# would surface a regression, even if that benchmark is indirect.

ENGINEERING_RISKS: tuple[dict[str, Any], ...] = (
    {
        "name": "synthetic_to_real_gap",
        "category": "data_quality",
        "example": "Champion classifier reports AUROC ≈ 0.95 on synthetic but external BreastDCEDL evaluation lands much lower.",
        "risk": "Performance numbers from synthetic data are not transferrable to real patients.",
        "detection": "Compare synthetic AUROC vs. external_breastdcedl_auroc in model_benchmark.",
        "mitigation": "Every patient-facing surface carries a non-diagnostic disclaimer; model recommendations are monitor_only.",
        "benchmark_coverage": ["model_benchmark", "synthetic_realism_candidate", "realism_candidate_ab_gate"],
        "remaining_gap": "No prospective validation; real-world calibration unknown.",
        "severity": "high",
    },
    {
        "name": "toxicity_classifier_tautological_features",
        "category": "data_pipeline",
        "example": "Toxicity classifier reports AUC ≈ 1.0 on synthetic; feature audit shows 8 features (intervention_count, nadir_anc, nadir_wbc, dose_delayed, pre_wbc, pre_anc, recovery_wbc, cycle) have label-separation gap ≥ 0.85.",
        "risk": "Headline AUC reflects synthetic-generator structure, not model skill. Quoting the AUC as 'how good the model is' without context is misleading.",
        "detection": "toxicity_feature_audit benchmark + strict no-proxy baseline (strip every near-label feature). When strict baseline AUC stays >0.90, the synthetic generator wires the label to too many features to remove cleanly.",
        "mitigation": "Audit artifact pinned to benchmark_registry; documented in model card; recommendations downstream are monitor_only.",
        "benchmark_coverage": ["toxicity_feature_audit"],
        "remaining_gap": "Synthetic generator design: toxicity labels are deterministic functions of CBC + intervention rows. Real-world toxicity prediction requires real-world labels — out of scope for this PoC.",
        "severity": "high",
    },
    {
        "name": "modality_overconfidence_when_imaging_present",
        "category": "model_behavior",
        "example": "Patient with full imaging gets a 90%+ probability that does not reflect uncertainty from the single-modality nature of the signal.",
        "risk": "Model could quote extreme probability on rows where imaging is the dominant signal.",
        "detection": "predict_with_abstention applies a confidence modifier on partial-evidence rows; modality_robustness_comparison reports per-scenario calibration.",
        "mitigation": "Confidence modifier shrinks probability toward prior when evidence is partial; abstention layer refuses scoring when response signal is absent.",
        "benchmark_coverage": ["evidence_abstention_eval", "modality_robustness_comparison", "calibration_eval"],
        "remaining_gap": "Modifier is rule-based, not learned; needs clinical advisor review.",
        "severity": "medium",
    },
    {
        "name": "label_proxy_leakage_in_features",
        "category": "data_pipeline",
        "example": "A future edit accidentally adds latent_response_strength or response_score_percent to NUMERIC_FEATURES.",
        "risk": "Inflated metrics with no real learning — model would memorise the label.",
        "detection": "leakage_audit runs as a hard CI gate against the production CSV.",
        "mitigation": "EXCLUDED_COLUMNS denylist + leakage_audit test in CI.",
        "benchmark_coverage": ["leakage_audit"],
        "remaining_gap": "Audit only covers known proxies; novel proxies in future generator versions need manual review.",
        "severity": "high",
    },
    {
        "name": "patient_id_overlap_between_splits",
        "category": "data_pipeline",
        "example": "train_test_split groups by cycle row instead of patient_id, putting cycles of the same patient on both sides.",
        "risk": "Inflated test metrics from intra-patient leakage; subgroup performance overestimated.",
        "detection": "leakage_audit asserts _patient_split is disjoint across multiple seeds + targets.",
        "mitigation": "All training code uses _patient_split; leakage_audit runs in CI.",
        "benchmark_coverage": ["leakage_audit"],
        "remaining_gap": None,
        "severity": "high",
    },
    {
        "name": "abstention_too_aggressive",
        "category": "model_behavior",
        "example": "Patient with CBC + symptoms but missing imaging gets `insufficient_evidence` when a cautious response signal would have been more useful.",
        "risk": "Over-cautious system that refuses to help — false_abstention_rate goes up.",
        "detection": "evidence_abstention_eval reports false_abstention_rate per scenario.",
        "mitigation": "Question-specific sufficiency rules tunable; toxicity and urgent intervention rules accept narrower evidence than response classification.",
        "benchmark_coverage": ["evidence_abstention_eval"],
        "remaining_gap": "Rules are defaults; clinical advisor sign-off needed before relaxing them.",
        "severity": "medium",
    },
    {
        "name": "tumor_marker_overclaim",
        "category": "clinical_safety",
        "example": "Patient sees a chat reply that says elevated CA 15-3 means cancer has returned.",
        "risk": "Standalone tumor-marker readings interpreted as diagnostic — dangerous overclaim.",
        "detection": "safety_red_team + genetic_counseling_eval probe tumor_marker_overclaim_rate.",
        "mitigation": "Safety gate refuses standalone interpretation; KB carries tumor-marker limitations sources.",
        "benchmark_coverage": ["safety_red_team", "genetic_counseling_eval"],
        "remaining_gap": "Multilingual variations of tumor-marker queries need wider coverage.",
        "severity": "high",
    },
    {
        "name": "vus_misinterpretation",
        "category": "clinical_safety",
        "example": "Patient asks 'My BRCA test says VUS — am I positive?' and the agent says yes.",
        "risk": "Genetic-variant misinterpretation can drive unnecessary preventive surgery.",
        "detection": "genetic_counseling_eval measures VUS_handling_correctness.",
        "mitigation": "Deterministic safety route refuses interpretation, defers to genetic counselor referral.",
        "benchmark_coverage": ["genetic_counseling_eval", "safety_red_team"],
        "remaining_gap": None,
        "severity": "high",
    },
    {
        "name": "hallucinated_citation",
        "category": "rag_quality",
        "example": "RAG answer cites a knowledge-base source that doesn't actually back the claim.",
        "risk": "False sense of grounding; patient trusts an unsupported claim.",
        "detection": "rag_regression citation_coverage + expected_source_hit_rate; llm_judge groundedness score (when available).",
        "mitigation": "Reciprocal-rank fusion + citation validator + safety-aware caching; output guardrail strips citations on refusal_type=safety_boundary.",
        "benchmark_coverage": ["rag_regression", "rag_gold", "llm_judge"],
        "remaining_gap": "Citation validator is heuristic; no semantic equivalence check yet.",
        "severity": "medium",
    },
    {
        "name": "urgent_symptom_not_escalated",
        "category": "clinical_safety",
        "example": "Patient says 'lagnat na lagnat, 39 degrees, ANC mababa' and the system gives education rather than urgent escalation.",
        "risk": "Delayed clinician contact for neutropenic fever or similar acute event.",
        "detection": "safety_red_team urgent_symptom_escalation category; multilingual_refusal_eval Taglish cases.",
        "mitigation": "Deterministic safety gate fires before RAG/LLM; urgent route bypasses retrieval and prompts clinician contact.",
        "benchmark_coverage": ["safety_red_team", "multilingual_refusal_eval"],
        "remaining_gap": "Coverage of code-switched urgent phrases beyond current Taglish set.",
        "severity": "high",
    },
)


def build_failure_mode_registry(
    *,
    failure_gallery_path: str = DEFAULT_FAILURE_GALLERY_PATH,
    safety_red_team_path: str = DEFAULT_SAFETY_RED_TEAM_PATH,
    drift_report_path: str = DEFAULT_DRIFT_REPORT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Aggregate the four source artifacts into one registry payload."""
    entries: list[dict[str, Any]] = []
    entries.extend(ENGINEERING_RISKS)
    entries.extend(_from_failure_gallery(failure_gallery_path))
    entries.extend(_from_safety_red_team(safety_red_team_path))
    entries.extend(_from_drift_report(drift_report_path))

    by_severity: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    by_category: dict[str, int] = {}
    for e in entries:
        sev = (e.get("severity") or "unknown").lower()
        by_severity[sev] = by_severity.get(sev, 0) + 1
        cat = e.get("category") or "uncategorised"
        by_category[cat] = by_category.get(cat, 0) + 1

    entries_with_unresolved_gap = sum(1 for e in entries if e.get("remaining_gap"))

    payload = {
        "schema_version": "failure_mode_registry_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _overall_status(by_severity, entries),
        "entry_count": len(entries),
        "summary": {
            "by_severity": by_severity,
            "by_category": by_category,
            "entries_with_unresolved_gap": entries_with_unresolved_gap,
        },
        "entries": entries,
        "sources": {
            "engineering_risks": "hand-curated, in source",
            "failure_case_gallery": failure_gallery_path,
            "safety_red_team": safety_red_team_path,
            "drift_report": drift_report_path,
        },
        "interpretation": (
            "Engineering provenance — describes known failure modes the "
            "system tries to detect and mitigate.  A populated registry "
            "is the floor for portfolio defensibility; it is not a "
            "guarantee that unknown failure modes don't exist."
        ),
        "claim_boundary": (
            "Engineering risk register, not clinical risk assessment.  "
            "Real-world failure modes outside this synthetic-data pipeline "
            "have not been enumerated."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_failure_mode_registry(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "failure_mode_registry_v1",
            "status": "missing",
            "message": (
                "Failure-mode registry has not been generated yet.  Run "
                "`scripts/run_failure_mode_registry.py` or POST to "
                "/admin/failure-mode-registry."
            ),
            "entries": [],
            "summary": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


# ─── Source adapters ─────────────────────────────────────────────────────────


def _safe_load(path: str) -> dict[str, Any] | list[Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _from_failure_gallery(path: str) -> list[dict[str, Any]]:
    payload = _safe_load(path)
    if not isinstance(payload, dict):
        return []
    out: list[dict[str, Any]] = []
    for case in payload.get("cases") or []:
        out.append({
            "name": case.get("id") or case.get("name") or "unnamed_gallery_case",
            "category": "narrative_case",
            "example": case.get("what_happened") or case.get("example") or "",
            "risk": case.get("why_risky") or "",
            "detection": case.get("system_response") or "manual review",
            "mitigation": case.get("mitigation") or "",
            "benchmark_coverage": ["failure_case_gallery"],
            "remaining_gap": case.get("unresolved") or None,
            "severity": case.get("severity") or "medium",
        })
    return out


def _from_safety_red_team(path: str) -> list[dict[str, Any]]:
    """Surface the *categories* of failed safety-red-team cases rather than
    every individual case — the registry stays scannable and the dashboard
    can drill into the underlying artifact for case-level detail."""
    payload = _safe_load(path)
    if not isinstance(payload, dict):
        return []
    summary = payload.get("summary") or {}
    failed_cases = summary.get("failed_cases") or []
    if not failed_cases:
        return []
    return [{
        "name": "safety_red_team_failures",
        "category": "adversarial",
        "example": f"{len(failed_cases)} safety case(s) failing: {', '.join(failed_cases[:5])}",
        "risk": "Adversarial inputs bypassed deterministic safety; review individual cases.",
        "detection": "safety_red_team benchmark",
        "mitigation": "Patch safety gate, expand category coverage, rerun benchmark.",
        "benchmark_coverage": ["safety_red_team"],
        "remaining_gap": "Each failed case ID is its own residual risk.",
        "severity": "high",
    }]


def _from_drift_report(path: str) -> list[dict[str, Any]]:
    """Surface drift findings as failure modes when status is not 'stable'."""
    payload = _safe_load(path)
    if not isinstance(payload, dict):
        return []
    out: list[dict[str, Any]] = []
    for section_key, label in (
        ("lab_distribution_shift",   "lab distribution shift"),
        ("symptom_frequency_shift",  "symptom frequency shift"),
        ("imaging_keyword_shift",    "imaging keyword shift"),
        ("calibration_drift",        "calibration drift"),
        ("subgroup_performance_drift", "subgroup performance drift"),
    ):
        section = payload.get(section_key)
        if not isinstance(section, dict):
            continue
        status = (section.get("status") or "").lower()
        if status and status not in {"stable", "no_drift", "passed", "available"}:
            out.append({
                "name": f"drift::{section_key}",
                "category": "data_quality",
                "example": section.get("message") or f"{label} flagged as {status}",
                "risk": f"Model performance may degrade under observed {label}.",
                "detection": "drift_report (latest)",
                "mitigation": "Investigate drift source, retrain or relabel as needed.",
                "benchmark_coverage": ["drift_report"],
                "remaining_gap": "Specific feature-level cause from drift artifact.",
                "severity": "medium",
            })
    return out


def _overall_status(by_severity: dict[str, int], entries: list[dict[str, Any]]) -> str:
    if any(
        e.get("severity") == "high" and e.get("remaining_gap")
        for e in entries
    ):
        return "needs_attention"
    if by_severity.get("high", 0) == 0:
        return "strong"
    return "acceptable"


__all__ = [
    "ENGINEERING_RISKS",
    "build_failure_mode_registry",
    "load_failure_mode_registry",
]
