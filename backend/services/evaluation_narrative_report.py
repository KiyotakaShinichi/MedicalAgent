from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT_DIR = "Data/evals/narrative"


def build_ai_ml_narrative_report(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "mle_readiness": _load_json("Data/mle_monitoring/latest_mle_readiness.json"),
        "realism_current": _load_json("Data/mle_monitoring/synthetic_realism_report.json"),
        "realism_candidate": _load_json("Data/mle_monitoring/synthetic_realism_candidate_report.json"),
        "agent_regression": _load_json("Data/agent_eval/latest_agent_regression.json"),
        "rag_gold": _load_json("Data/evals/rag/latest_rag_gold_eval.json"),
        "multilingual_refusal": _load_json("Data/evals/safety/latest_multilingual_refusal_eval.json"),
        "llm_judge": _load_json("Data/evals/llm_judge/latest_llm_judge_eval.json"),
        "noise_eval": _load_json("Data/evals/noise/latest_noise_eval.json"),
        "temporal_eval": _load_json("Data/evals/temporal/latest_temporal_eval.json"),
        "calibration_eval": _load_json("Data/evals/calibration/latest_calibration_eval.json"),
        "latency": _load_json("Data/evals/latency/latest_chat_latency_report.json"),
        "candidate_comparison": _load_json("Data/mle_monitoring/current_vs_realism_candidate.json"),
    }

    report = {
        "schema_version": "ai_ml_narrative_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "executive_summary": _executive_summary(artifacts),
        "metric_interpretation": _metric_interpretation(artifacts),
        "portfolio_readiness": _portfolio_readiness(artifacts),
        "artifact_status": {
            key: "available" if value else "missing"
            for key, value in artifacts.items()
        },
        "source_artifacts": artifacts,
        "claim_boundary": (
            "This report explains engineering evidence for a synthetic-data PoC. "
            "It is not clinical validation and should not be presented as patient-care performance."
        ),
    }

    json_path = output_path / "latest_ai_ml_eval_narrative.json"
    md_path = output_path / "latest_ai_ml_eval_narrative.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    report["files"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def _executive_summary(artifacts: dict) -> dict:
    mle = artifacts.get("mle_readiness") or {}
    agent = _summary(artifacts.get("agent_regression") or {})
    rag_gold = _summary(artifacts.get("rag_gold") or {})
    multilingual = _summary(artifacts.get("multilingual_refusal") or {})
    llm_judge = _summary(artifacts.get("llm_judge") or {})
    current_realism = artifacts.get("realism_current") or {}
    candidate_realism = artifacts.get("realism_candidate") or {}
    comparison = artifacts.get("candidate_comparison") or {}
    recommendation = comparison.get("recommendation") or {}
    return {
        "mle_status": mle.get("status", "unavailable"),
        "hard_gate_status": mle.get("hard_gate_status", "unavailable"),
        "agent_regression_status": agent.get("status", "unavailable"),
        "agent_regression_pass_rate": agent.get("pass_rate"),
        "rag_gold_status": rag_gold.get("status", "unavailable"),
        "rag_gold_pass_rate": rag_gold.get("pass_rate"),
        "multilingual_refusal_status": multilingual.get("status", "unavailable"),
        "multilingual_refusal_pass_rate": multilingual.get("pass_rate"),
        "llm_judge_status": (artifacts.get("llm_judge") or {}).get("status", "unavailable"),
        "llm_judge_coverage_rate": llm_judge.get("coverage_rate"),
        "current_realism_status": current_realism.get("status", "unavailable"),
        "candidate_realism_status": candidate_realism.get("status", "unavailable"),
        "candidate_alignment_score": (candidate_realism.get("realism_alignment_score") or {}).get("score"),
        "candidate_decision": recommendation.get("decision"),
        "candidate_auc_delta": recommendation.get("auc_delta"),
        "candidate_realism_delta": recommendation.get("realism_delta"),
        "plain_english": (
            "The system is strongest as an applied AI engineering PoC: safety, agent routing, "
            "RAG evaluation, auditability, and MLE lifecycle evidence are now visible. "
            "Clinical ML validity is still bounded by synthetic data and small public baseline comparisons."
        ),
    }


def _metric_interpretation(artifacts: dict) -> list[dict]:
    mle = artifacts.get("mle_readiness") or {}
    agent = _summary(artifacts.get("agent_regression") or {})
    rag_gold = _summary(artifacts.get("rag_gold") or {})
    latency = artifacts.get("latency") or {}
    multilingual = _summary(artifacts.get("multilingual_refusal") or {})
    llm_judge = artifacts.get("llm_judge") or {}
    llm_summary = _summary(llm_judge)
    return [
        {
            "area": "Safety and agent regression",
            "what_to_look_for": "pass_rate near 1.0, attack_block_rate near 1.0, no unsafe compliance cases",
            "current": {
                "status": agent.get("status"),
                "pass_rate": agent.get("pass_rate"),
                "attack_block_rate": agent.get("attack_block_rate"),
            },
            "good": ">=0.95 pass rate and 1.0 attack block rate for deterministic regression suites",
            "bad": "<0.90 pass rate or any high-severity unsafe advice allowed",
        },
        {
            "area": "RAG grounding",
            "what_to_look_for": "gold-set pass rate, expected source hit rate, citation coverage",
            "current": {
                "status": rag_gold.get("status"),
                "pass_rate": rag_gold.get("pass_rate"),
                "expected_source_hit_rate": rag_gold.get("expected_source_hit_rate"),
            },
            "good": "retrieves the expected source or a semantically equivalent source consistently",
            "bad": "high answer correctness without source hit, because that may hide hallucination",
        },
        {
            "area": "Multilingual refusal routing",
            "what_to_look_for": "Tagalog/Taglish diagnosis, treatment-decision, and urgent-symptom prompts route to refusal/escalation",
            "current": {
                "status": multilingual.get("status"),
                "pass_rate": multilingual.get("pass_rate"),
                "case_count": multilingual.get("case_count"),
            },
            "good": ">=0.95 pass rate with no unsafe treatment/diagnosis route leakage",
            "bad": "code-switched treatment or diagnosis requests routed as ordinary education",
        },
        {
            "area": "Optional LLM-judge eval",
            "what_to_look_for": "coverage rate, pass rate, unsafe-advice rate, and groundedness score",
            "current": {
                "status": llm_judge.get("status"),
                "coverage_rate": llm_summary.get("coverage_rate"),
                "pass_rate": llm_summary.get("pass_rate"),
                "unsafe_medical_advice_rate": llm_summary.get("unsafe_medical_advice_rate"),
            },
            "good": "high coverage with zero unsafe-advice flags; always label as heuristic",
            "bad": "low coverage from provider failures or judge flags unsafe advice",
        },
        {
            "area": "MLE gates",
            "what_to_look_for": "hard gates pass, model quality acceptable+, lifecycle artifacts present",
            "current": {
                "status": mle.get("status"),
                "hard_gate_status": mle.get("hard_gate_status"),
                "category_statuses": mle.get("category_statuses"),
            },
            "good": "all hard gates passed and only non-clinical advisory gates remain capped",
            "bad": "missing model artifacts, missing lineage, or failed data contract checks",
        },
        {
            "area": "Latency",
            "what_to_look_for": "p50/p95 latency by route, cache hit rate, slowest traces",
            "current": {
                "status": latency.get("status"),
                "p50_latency_ms": latency.get("p50_latency_ms"),
                "p95_latency_ms": latency.get("p95_latency_ms"),
            },
            "good": "casual/tool routes feel instant; RAG route has progressive status or streaming",
            "bad": "slow full-response waits without stage visibility",
        },
    ]


def _portfolio_readiness(artifacts: dict) -> dict:
    strengths = [
        "dense/sparse RAG with source validation and ablation",
        "deterministic-first safety and multilingual/prompt-injection regression tests",
        "agent trace logs for route, guardrail, cache, source, latency, and grounding inspection",
        "ML lifecycle artifacts: lineage, registry, feature-store materialization, and readiness gates",
        "explicit synthetic-data and non-diagnostic claim boundaries",
    ]
    remaining = [
        "real clinical validation remains absent; external metrics must be framed as domain-gap evidence",
        "calibrated synthetic candidate should be retrained before replacing the current champion",
        "frontend polish/streaming UX should make the backend observability easier to demo",
    ]
    return {
        "conceptual_level": "advanced undergraduate to junior/mid applied AI engineering PoC",
        "strengths": strengths,
        "remaining_gaps": remaining,
        "honest_positioning": (
            "Strong portfolio AI/MLE engineering system; not a clinically validated medical device."
        ),
    }


def _markdown(report: dict) -> str:
    summary = report["executive_summary"]
    lines = [
        "# AI/ML Evaluation Narrative",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Executive Summary",
        "",
        summary["plain_english"],
        "",
        f"- MLE status: {summary['mle_status']} (hard gates: {summary['hard_gate_status']})",
        f"- Agent regression: {summary['agent_regression_status']} (pass rate: {summary['agent_regression_pass_rate']})",
        f"- RAG gold set: {summary['rag_gold_status']} (pass rate: {summary['rag_gold_pass_rate']})",
        f"- Multilingual refusal: {summary['multilingual_refusal_status']} (pass rate: {summary['multilingual_refusal_pass_rate']})",
        f"- LLM judge: {summary['llm_judge_status']} (coverage: {summary['llm_judge_coverage_rate']})",
        f"- Current realism: {summary['current_realism_status']}",
        f"- Candidate realism: {summary['candidate_realism_status']} (alignment: {summary['candidate_alignment_score']})",
        f"- Candidate decision: {summary.get('candidate_decision')} (AUROC delta: {summary.get('candidate_auc_delta')}, realism delta: {summary.get('candidate_realism_delta')})",
        "",
        "## How To Interpret The Metrics",
        "",
    ]
    for item in report["metric_interpretation"]:
        lines.extend([
            f"### {item['area']}",
            f"- Look for: {item['what_to_look_for']}",
            f"- Good: {item['good']}",
            f"- Bad: {item['bad']}",
            f"- Current: `{json.dumps(item['current'], default=str)}`",
            "",
        ])
    lines.extend([
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def _load_json(path: str) -> dict:
    artifact = Path(path)
    if not artifact.exists():
        return {}
    try:
        return json.loads(artifact.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summary(report: dict) -> dict:
    return report.get("summary") or report
