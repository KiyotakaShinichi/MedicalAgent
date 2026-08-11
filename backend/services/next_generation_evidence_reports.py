"""Generate honest Markdown reports from executed NLCare evidence artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
ARTIFACTS = {
    "integrity": "Data/evals/governance/latest_next_generation_eval_integrity.json",
    "adversarial": "Data/evals/safety/latest_adversarial_generalization_v_next.json",
    "section": "Data/evals/rag/latest_section_aware_retrieval_eval.json",
    "baseline": "Data/evals/rag/latest_rag_baseline_comparison.json",
    "failure": "Data/evals/rag/latest_rag_failure_attribution_v_next.json",
    "tenant": "Data/evals/security/latest_tenant_isolation_security_eval.json",
    "poison": "Data/evals/security/latest_corpus_poisoning_eval.json",
    "load": "Data/evals/ops/latest_synthetic_load_matrix.json",
    "local_load": "Data/evals/ops/latest_load_test_report.json",
    "latency": "Data/evals/models/latest_agent_latency_probe.json",
    "automation": "Data/evals/ops/latest_automation_fault_injection.json",
    "rag_resilience": "Data/evals/rag/latest_rag_degradation_resilience.json",
    "data_reliability": "Data/evals/ops/latest_data_platform_reliability_eval.json",
    "review": "Data/evals/governance/latest_human_review_feedback_ingestion.json",
    "review_readiness": "Data/evals/governance/latest_external_review_execution_readiness.json",
    "saas": "Data/evals/ops/latest_saas_foundation_readiness.json",
    "eval_run": "Data/evals/governance/latest_nlcare_eval_run.json",
    "ai_trinity": "Data/evals/governance/latest_ai_trinity_tradeoff.json",
    "citation_selector": "Data/evals/rag/latest_claim_conditioned_citation_selector_holdout.json",
    "provider_probe": "Data/evals/ops/latest_provider_api_path_capture.json",
}
BOUNDARY = (
    "NLCare remains synthetic-only, non-diagnostic, not clinically validated, and not production healthcare ready. "
    "Internal tests are engineering evidence, not evidence of patient benefit or medical effectiveness."
)


def write_next_generation_reports(*, report_dir: str | Path = REPORT_DIR) -> list[Path]:
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    data = {name: _read(ROOT / path) for name, path in ARTIFACTS.items()}
    reports = {
        "next_generation_gap_analysis.md": _gap_analysis(data),
        "adversarial_generalization_v_next.md": _adversarial(data),
        "retrieval_ablation_study.md": _retrieval(data),
        "runtime_performance_v_next.md": _performance(data),
        "tenant_isolation_security.md": _tenant(data),
        "reliability_and_chaos.md": _reliability(data),
        "nlcare_next_generation_final_report.md": _final(data),
    }
    paths: list[Path] = []
    for name, content in reports.items():
        path = target / name
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _header(title: str) -> list[str]:
    return [
        f"# {title}",
        "",
        f"Generated from repository artifacts at `{datetime.now(timezone.utc).isoformat()}`.",
        "",
        f"> {BOUNDARY}",
        "",
    ]


def _gap_analysis(data: dict[str, dict[str, Any]]) -> str:
    integrity = data["integrity"]
    adversarial = data["adversarial"]
    section = data["section"]
    lines = _header("NLCare Next-Generation Gap Analysis")
    lines += [
        "## Current state",
        "",
        "NLCare has a mature portfolio-grade safety and evaluation surface: deterministic and semantic boundaries, source-governed RAG, bounded tool workflows, tenant-aware SaaS scaffolding, synthetic temporal ML, observability, and release gates.",
        "",
        "## Existing strengths",
        "",
        f"- Frozen dataset integrity failures: `{integrity.get('integrity_failure_count', 'NOT_RUN')}`.",
        f"- Internal post-change adversarial mutation pass rate: `{_get(adversarial, 'mutation_matrix.pass_rate')}`; this bank is tuning-used.",
        f"- Tenant security: `{data['tenant'].get('status', 'NOT_RUN')}`.",
        f"- Corpus poisoning: `{data['poison'].get('status', 'NOT_RUN')}`.",
        "- The codebase preserves negative RAG findings instead of promoting complexity by default.",
        "",
        "## Confirmed weaknesses",
        "",
        f"- Existing v7 internal holdout pass rate remains `{_get(adversarial, 'frozen_v7_read_only_attribution.source_pass_rate')}` and was already inspected.",
        f"- Independently authored external evaluation: `{adversarial.get('external_generalization_status', 'BLOCKED_EXTERNAL')}`.",
        f"- Section-aware live promotion: `{_get(section, 'decision.promoted_to_live_retrieval')}`.",
        "- Dense serving is available locally, but the restricted Docker profile remains sparse until quality and latency evidence justify promotion.",
        "- No clinician, nurse, or genetic counselor review has been completed.",
        "",
        "## Technical debt and risk",
        "",
        "- Multiple historical eval artifacts overlap in cases, so consolidated attribution reports both raw rows and unique case-stage counts.",
        "- Process-local load measurements isolate planner concurrency and cannot substitute for network/provider/DB load.",
        "- The synthetic ML layer demonstrates engineering discipline but cannot establish clinical realism or transfer.",
        "- A large release-gate inventory can dilute attention; critical blockers must remain distinct from informational scaffolds.",
        "",
        "## External blockers",
        "",
        "- Independently authored no-read RAG and adversarial cases.",
        "- Oncology clinician or nurse wording/safety review.",
        "- Genetics-qualified VUS review.",
        "- Managed-cloud credentials, independent security review, and non-synthetic traffic evidence.",
        "- Real-data or patient-facing work would require institutional governance and appropriate ethics/IRB pathways.",
        "",
        "## Implementation order",
        "",
        "1. Preserve frozen hashes and complete external no-read authoring.",
        "2. Adjudicate patient-facing versus clinician-facing source expectations.",
        "3. Optimize only failure buckets with enough mass and verify on untouched data.",
        "4. Run managed synthetic staging, backup/restore, and distributed load drills.",
        "5. Convert accepted human findings into versioned regression tests.",
    ]
    return "\n".join(lines)


def _adversarial(data: dict[str, dict[str, Any]]) -> str:
    artifact = data["adversarial"]
    mutation = artifact.get("mutation_matrix") or {}
    frozen = artifact.get("frozen_v7_read_only_attribution") or {}
    lines = _header("Adversarial Generalization V-Next")
    lines += [
        "## Method",
        "",
        "The v7 result is read-only attribution over an already-inspected internal holdout. A separate post-change mutation matrix evaluates generalized wrappers and safe controls and is explicitly marked tuning-used.",
        "",
        "## Before and after evidence",
        "",
        "| Evidence | Pass rate | Unsafe leakage/block signal | Interpretation |",
        "|---|---:|---:|---|",
        f"| Frozen v7 stored baseline | {frozen.get('source_pass_rate', 'NOT_RUN')} | leakage {frozen.get('source_unsafe_leakage_rate', 'NOT_RUN')} | Internal and author-contaminated; not rerun |",
        f"| V-next mutation matrix | {mutation.get('pass_rate', 'NOT_RUN')} | unsafe block {mutation.get('unsafe_block_rate', 'NOT_RUN')} | Tuning-used regression only |",
        f"| Safe negative controls | {mutation.get('safe_negative_pass_rate', 'NOT_RUN')} | over-refusal {mutation.get('over_refusal_rate', 'NOT_RUN')} | Internal controls |",
        "",
        "## Architecture change",
        "",
        "An independent operation-authorization guard now sits before tool planning. Privacy, prompt-injection, and cross-patient operations can be denied even if another classifier misses, while the existing semantic/security route remains an independent blocking layer.",
        "",
        "## Remaining failures and limits",
        "",
        f"- Frozen v7 attributed failure count: `{frozen.get('source_failure_count', 'NOT_RUN')}`.",
        f"- External generalization status: `{artifact.get('external_generalization_status', 'BLOCKED_EXTERNAL')}`.",
        "- A perfect tuning-used mutation score must not replace an untouched independent result.",
        "- No claim of solved safety is authorized.",
    ]
    return "\n".join(lines)


def _retrieval(data: dict[str, dict[str, Any]]) -> str:
    section = data["section"]
    baseline = data["baseline"]
    failure = data["failure"]
    decision = section.get("decision") or {}
    known = section.get("known_miss_evaluation") or {}
    lines = _header("NLCare Retrieval Ablation Study")
    lines += [
        "## Experiment",
        "",
        "The repository compares BM25, dense FAISS, hybrid RRF, query rewriting, parent-child expansion, source-tier filtering, and section-aware variants. The section experiment uses corrected structural metadata but remains internal and tuning-used.",
        "",
        "## Baseline decision",
        "",
        f"- Complex stack improvement over BM25 proven: `{baseline.get('improvement_proven_vs_bm25', _get(baseline, 'summary.improvement_proven_vs_bm25'))}`.",
        "- Prior evidence showed governance value from source-tier filtering, but did not prove raw retrieval superiority over BM25.",
        "- The citation pruner remains not promoted because it regressed citation precision.",
        "",
        "## Section-aware result",
        "",
        f"- Known section misses evaluated: `{section.get('known_section_miss_count', 'NOT_RUN')}`.",
        f"- Recovered misses: `{known.get('recovered_misses', 'NOT_RUN')}`.",
        f"- Remaining misses: `{known.get('remaining_misses', 'NOT_RUN')}`.",
        f"- Regression cases: `{known.get('regression_cases', 'NOT_RUN')}`.",
        f"- Section hit delta: `{decision.get('section_hit_rate_delta', 'NOT_RUN')}`.",
        f"- Paper Recall@10 delta: `{decision.get('paper_recall_at_10_delta', 'NOT_RUN')}`.",
        f"- Paper precision@5 delta: `{decision.get('expected_paper_precision_at_5_delta', 'NOT_RUN')}`.",
        f"- Promoted to live retrieval: `{decision.get('promoted_to_live_retrieval', False)}`.",
        "",
        "## Failure attribution",
        "",
        _stage_table(failure.get("aggregate_by_stage") or {}),
        "",
        "## Dense serving decision",
        "",
        "A fingerprint-matched local dense index is implemented and benchmarkable. Restricted synthetic staging keeps sparse fallback until held-out evidence justifies dense/hybrid quality and latency. This is an evidence-based non-promotion, not a missing feature.",
    ]
    return "\n".join(lines)


def _performance(data: dict[str, dict[str, Any]]) -> str:
    load = data["load"]
    latency = data["latency"]
    mixed = _read(ROOT / "Data/evals/agentic_tool_use/latest_mixed_query_scale_eval.json")
    lines = _header("NLCare Runtime Performance V-Next")
    lines += [
        "## Measurement scopes",
        "",
        "- Planner load matrix: process-local route and authorization concurrency only.",
        "- Agent latency probe: local sparse RAG with in-memory SQLite.",
        "- Mixed query stress: internally generated research, garbage, and dangerous prompts.",
        "",
        "## Cold and warm evidence",
        "",
        f"- Planner prewarm: `{_get(load, 'prewarm.duration_ms')}` ms across `{_get(load, 'prewarm.query_count')}` route families.",
        f"- Agent warmup: `{_get(latency, 'warmup.total_ms')}` ms.",
        f"- Mixed-query prewarm: `{_get(mixed, 'summary.prewarm_latency_ms')}` ms.",
        "",
        "## Concurrency matrix",
        "",
        "| Concurrency | Requests | Throughput rps | Error rate | p50 ms | p95 ms | p99 ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in load.get("profiles") or []:
        l = row.get("latency_ms") or {}
        lines.append(f"| {row.get('concurrency')} | {row.get('request_count')} | {row.get('throughput_rps')} | {row.get('error_rate')} | {l.get('p50')} | {l.get('p95')} | {l.get('p99')} |")
    lines += [
        "",
        "## Decision",
        "",
        "No production SLO is claimed. Provider-token accounting remains incomplete when provider usage metadata is absent, and staged serving is not declared production-ready. The cold/warm split is mandatory for future comparisons.",
    ]
    return "\n".join(lines)


def _tenant(data: dict[str, dict[str, Any]]) -> str:
    tenant = data["tenant"]
    lines = _header("NLCare Tenant-Isolation Security")
    lines += [
        "## Controlled matrix",
        "",
        "Three disposable synthetic tenants exercise foreign IDs, role confusion, cache keys, vector namespaces, project relationships, worker scope, and authorization bypass attempts.",
        "",
        f"- Status: `{tenant.get('status', 'NOT_RUN')}`.",
        f"- Cases: `{tenant.get('total_n', tenant.get('case_count', 'NOT_RUN'))}`.",
        f"- Passed: `{tenant.get('pass_count', tenant.get('passed', 'NOT_RUN'))}`.",
        f"- Leakage rate: `{tenant.get('leakage_rate', _get(tenant, 'metrics.leakage_rate'))}`.",
        "- This is controlled local security regression evidence, not a penetration test or certification.",
        "",
        "## Remaining deployment work",
        "",
        "- Managed PostgreSQL row-level policy verification.",
        "- Real OIDC provider integration and session revocation drills.",
        "- Gateway/WAF configuration, secret rotation, independent penetration testing, and incident ownership.",
    ]
    return "\n".join(lines)


def _reliability(data: dict[str, dict[str, Any]]) -> str:
    automation = data["automation"]
    rag = data["rag_resilience"]
    load = data["load"]
    lines = _header("NLCare Reliability and Chaos Evidence")
    lines += [
        "## Executed drills",
        "",
        f"- Automation fault injection: `{automation.get('status', 'NOT_RUN')}`, `{automation.get('passed_count', 'NOT_RUN')}/{automation.get('scenario_count', 'NOT_RUN')}` passed.",
        f"- RAG degradation resilience: `{rag.get('status', 'NOT_RUN')}`.",
        f"- Planner concurrency matrix: `{load.get('status', 'NOT_RUN')}`.",
        f"- Forbidden tool exposure: `{_get(load, 'invariants.forbidden_tool_exposure_count')}`.",
        f"- Exceptions under planner load: `{_get(load, 'invariants.exception_count')}`.",
        "",
        "## Covered failure conditions",
        "",
        "- Duplicate enqueue and delivery, lease contention, stale-lease recovery, bounded retries, dead letter and audited requeue.",
        "- Stable event IDs after crash-like replay, signature rotation, tamper rejection, and stale-event rejection.",
        "- Dense-index degradation and local fallback through existing RAG resilience drills.",
        "",
        "## Not yet proven",
        "",
        "- Managed Redis/PostgreSQL outage behavior, point-in-time restore, multi-host worker termination, real provider timeouts, and network partition recovery remain BLOCKED_EXTERNAL or require managed staging.",
        "- No external notification is treated as clinician acknowledgement or emergency coverage.",
    ]
    return "\n".join(lines)


def _final(data: dict[str, dict[str, Any]]) -> str:
    section = data["section"]
    adversarial = data["adversarial"]
    review = data["review"]
    saas = data["saas"]
    trinity = data["ai_trinity"]
    selector = data["citation_selector"]
    provider_probe = data["provider_probe"]
    lines = _header("NLCare Next-Generation Engineering Evidence Report")
    sections = [
        ("1. Executive summary", "NLCare is a strong portfolio-grade healthcare AI engineering prototype with unusually explicit safety, retrieval, MLE, security, and governance evidence. Its strongest new contribution is a reproducible evidence program that preserves negative results. It remains far from a clinical product."),
        ("2. Starting state", "The repository already included source-governed dense/sparse RAG, bounded tools, claim checks, synthetic temporal ML, n8n automation scaffolding, role dashboards, traces, and a large release gate. Weak evidence included internally authored holdouts, section retrieval misses, sparse staged serving, and no completed external review."),
        ("3. Changes implemented", "Added frozen-dataset integrity, independent route authorization, retrieved-context integrity, corrected section parsing, section-aware ablation, adversarial attribution/mutations, corpus-poisoning tests, tenant-isolation attacks, failure attribution, load profiling, review feedback validation, a unified nlcare_eval runner, and a non-compensatory Accuracy-Latency-Unit-Cost gate."),
        ("4. Architecture changes", "The request path now has an additional operation-level authorization layer. The retrieval path sanitizes instruction-like context before assembly. Ingestion persists section headings and canonical section provenance. SaaS workers validate organization/project relationships before execution."),
        ("5. Evaluation methodology", "All internal, frozen, tuning-used, and blocked-external assets are identified in a registry with hashes and usage restrictions. Reports include code/index/dataset provenance where the unified runner is used."),
        ("6. Frozen/generalization datasets", f"Integrity status: `{data['integrity'].get('status', 'NOT_RUN')}` with `{data['integrity'].get('integrity_failure_count', 'NOT_RUN')}` failures. Independent external work remains `{data['integrity'].get('external_review_status', 'BLOCKED_EXTERNAL')}`."),
        ("7. Adversarial results", f"Stored v7 pass rate is `{_get(adversarial, 'frozen_v7_read_only_attribution.source_pass_rate')}`. The new tuning-used mutation matrix pass rate is `{_get(adversarial, 'mutation_matrix.pass_rate')}`. This does not prove external generalization."),
        ("8. Retrieval ablation results", f"The previous full source-governed stack did not prove raw Recall@10 superiority over BM25. Section-aware promotion is `{_get(section, 'decision.promoted_to_live_retrieval')}`."),
        ("9. Section-aware retrieval results", f"Recovered `{_get(section, 'known_miss_evaluation.recovered_misses')}` of `{section.get('known_section_miss_count', 'NOT_RUN')}` known internal misses, with `{_get(section, 'known_miss_evaluation.regression_cases')}` regressions."),
        ("10. Dense-vs-BM25-vs-hybrid decision", "Dense FAISS is implemented, indexed, and locally benchmarkable. Sparse remains the restricted-staging default because complexity has not earned a held-out quality/latency promotion."),
        ("11. Failure attribution", f"Largest unique case-stage bucket: `{_get(data['failure'], 'engineering_decision.largest_observed_bucket')}`."),
        ("12. Corpus-poisoning results", f"Status: `{data['poison'].get('status', 'NOT_RUN')}`. Generation-context poison rate: `{data['poison'].get('generation_context_poison_rate', _get(data['poison'], 'metrics.generation_context_poison_rate'))}`."),
        ("13. Tenant-isolation results", f"Status: `{data['tenant'].get('status', 'NOT_RUN')}`. No result is a penetration-test claim."),
        ("14. Runtime performance and AI Trinity", f"Planner load status: `{data['load'].get('status', 'NOT_RUN')}`. AI Trinity decision: `{trinity.get('decision', 'NOT_RUN')}` with accuracy `{_get(trinity, 'summary.accuracy_status')}`, latency `{_get(trinity, 'summary.latency_status')}`, and unit cost `{_get(trinity, 'summary.unit_cost_status')}`. Normal-API provider probe: `{provider_probe.get('status', 'NOT_RUN')}` with coverage `{provider_probe.get('provider_usage_coverage_rate', 'NOT_RUN')}`. Missing provider telemetry is not treated as zero cost. These are internal measurements, not a production SLO."),
        ("15. Chaos/reliability results", f"Automation fault status: `{data['automation'].get('status', 'NOT_RUN')}`; RAG degradation status: `{data['rag_resilience'].get('status', 'NOT_RUN')}`."),
        ("16. Human-review status", f"Feedback ingestion status: `{review.get('status', 'BLOCKED_EXTERNAL')}`; accepted rows: `{review.get('accepted_feedback_row_count', 0)}`. No clinician sign-off is implied."),
        ("17. Deployment status", f"Synthetic SaaS foundation: `{saas.get('status', 'NOT_RUN')}`. Managed deployment completed: `{saas.get('managed_cloud_deployment_completed', False)}`."),
        ("18. Test/release evidence", "The final verification section must be read with the latest ship and release-gate artifacts. A green engineering gate does not establish clinical readiness."),
        ("19. Negative results", f"Full-stack raw retrieval superiority remains unproven, the prior citation pruner regressed precision, and the frozen claim-conditioned selector changed citation precision from `{selector.get('baseline_top3_citation_precision', 'NOT_RUN')}` to `{selector.get('selector_citation_precision', 'NOT_RUN')}` with promotion `{selector.get('promotion_decision', 'NOT_RUN')}`. V7 generalization was weak, provider token coverage is incomplete, and external review is absent."),
        ("20. Remaining weaknesses", "Independent evaluation, clinician/genetics review, real-data validity, managed outage/restore drills, provider cost telemetry, and human-factors testing remain the main credibility gaps."),
        ("21. External blockers", "External authors/reviewers, managed cloud credentials, real traffic, restricted datasets, institutional governance, and any future IRB pathway are outside this repository pass."),
        ("22. Promotion decisions", "Promote the dataset-integrity gate, operation authorization, context-integrity sanitizer, tenant relation checks, and controlled security regressions. Promote section-aware retrieval only if the artifact's predeclared conditions pass. Do not promote clinical authority, dense serving by appearance, the negative citation pruner, or the negative claim-conditioned selector."),
        ("23. Next highest-value task", "Complete one independently authored no-read RAG/adversarial evaluation and one qualified reviewer packet. Inside the repo, next run the same fixed suites in managed synthetic staging with traceable provider usage and backup/restore drills."),
    ]
    for title, body in sections:
        lines.extend([f"## {title}", "", body, ""])
    return "\n".join(lines)


def _stage_table(rows: dict[str, Any]) -> str:
    if not rows:
        return "No failure attribution artifact was available."
    output = ["| Stage | Unique case count | Share |", "|---|---:|---:|"]
    for stage, value in rows.items():
        output.append(f"| {stage} | {value.get('count')} | {value.get('share_of_stage_assignments')} |")
    return "\n".join(output)


def _get(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return "NOT_RUN"
        value = value.get(part)
        if value is None:
            return "NOT_RUN"
    return value


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


__all__ = ["write_next_generation_reports"]
