"""Sections 16-17: latency and runtime quality, and the release
discipline and test strategy.

Extracted verbatim from ``build_story`` in
``scripts/generate_project_guide_pdf.py``, which had grown to 1032 lines in a
single function. Flowable content and ordering are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reportlab.lib.units import mm
from scripts.project_guide.components import P, bullets, callout, data_table, flow_diagram, metric_row, page_break, section
from scripts.project_guide.evidence import _dig, _fmt, _pct

if TYPE_CHECKING:
    from scripts.project_guide.evidence import Evidence

def build(story: list[Any], ev: Evidence) -> None:
    """Append this module's sections to `story`, in order."""
    latency = ev.latency
    sentinel = ev.sentinel

    section(story, "16", "Latency and runtime quality", "Current local measurements are diagnostic and sparse; production readiness remains false.")
    routes = latency.get("routes", latency.get("route_results", []))
    route_rows: list[list[Any]] = []
    if isinstance(routes, dict):
        routes = [{"route": key, **(value if isinstance(value, dict) else {})} for key, value in routes.items()]
    for item in list(routes)[:8]:
        route_rows.append(
            [
                item.get("route", item.get("route_id", "route")),
                _fmt(item.get("sample_count"), 0),
                _fmt(item.get("current_p50_ms"), 1),
                _fmt(item.get("current_p95_ms"), 1),
                str(item.get("bottleneck_stage", "not reported")),
                _fmt(item.get("production_ready", False)),
            ]
        )
    if route_rows:
        story.append(data_table(["Route", "N", "p50 ms", "p95 ms", "Bottleneck", "Prod ready"], route_rows, [45 * mm, 13 * mm, 22 * mm, 22 * mm, 45 * mm, 23 * mm]))
    story.append(P("Runtime sentinel", "Heading2Custom"))
    story.append(
        metric_row(
            [
                ("Observed p50", f"{_fmt(_dig(sentinel, 'metrics', 'latency_ms', 'p50'), 1)} ms"),
                ("Observed p95", f"{_fmt(_dig(sentinel, 'metrics', 'latency_ms', 'p95'), 1)} ms"),
                ("Observed p99", f"{_fmt(_dig(sentinel, 'metrics', 'latency_ms', 'p99'), 1)} ms"),
                ("Cache hit rate", _pct(_dig(sentinel, 'metrics', 'cache_hit_rate'))),
            ],
            tone="neutral",
        )
    )
    story.append(
        callout(
            "Interpret carefully",
            "Route samples are small, some paths are unsampled, and local hardware/network/model state strongly affects timing. A green engineering gate means configured checks passed; it does not establish a production SLO, hospital availability, or real-world load behavior.",
            tone="amber",
        )
    )
    story.append(P("Performance priorities", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Profile retrieval, embedding, LLM queue time, generation, claim validation, and persistence independently.",
                "Batch or precompute embeddings and cache only policy-safe education with fingerprint invalidation.",
                "Set route-specific budgets; deterministic refusal and structured tools should not pay normal-RAG cost.",
                "Use warm/cold measurements, at least hundreds of samples per route, and concurrency/load distributions before making performance claims.",
                "Expose timeouts, degraded modes, cancellation, and user-visible progress without hiding incomplete responses.",
            ]
        )
    )
    page_break(story)
    section(story, "17", "Release discipline and test strategy", "The ship gate verifies engineering integration while release policy distinguishes hard blockers from warnings and information.")
    story.append(
        flow_diagram(
            [
                ("Backend integration", "Breast-monitoring, agent, RAG, model, boundary behavior"),
                ("Frontend unit", "Components, labels, interactions, state"),
                ("Browser smoke", "Patient, clinician, admin workflows on isolated data"),
                ("Static quality", "TypeScript lint and production build"),
                ("Artifact gate", "Freshness, statuses, metrics, claim-boundary locks"),
                ("Ship result", "Pass/fail as engineering evidence only"),
            ],
            columns=3,
        )
    )
    story.append(P("Release tiers", "Heading2Custom"))
    story.append(
        data_table(
            ["Tier", "Examples", "Policy"],
            [
                ["Hard blocker", "Unsafe leakage on critical routes, medical-boundary regression, data leakage, integration failure, clinical overclaim", "Release fails"],
                ["Warning", "Weak held-out safety, over-refusal increase, retrieval lift unproven, high unsupported context, latency over budget", "Visible and reviewed; does not masquerade as success"],
                ["Supporting", "Schema readiness, synthetic quality proxies, dataset maps", "Context for reviewers, not proof"],
                ["Informational", "Prepared packets, experiments, scaffolds, negative-result galleries", "Cannot turn the gate green by itself"],
            ],
            [30 * mm, 87 * mm, 53 * mm],
        )
    )
    story.append(P("Core commands", "Heading2Custom"))
    story.append(P("python scripts/ship.py", "Formula"))
    story.append(P("python -m pytest tests/test_breast_monitoring.py -q", "Formula"))
    story.append(P("python scripts/run_large_scale_agent_prompt_eval.py", "Formula"))
    story.append(P("python scripts/run_release_gate.py", "Formula"))
    story.append(P("cd frontend-react; npm run test; npm run test:e2e; npm run lint; npm run build", "Formula"))
    story.append(P("Testing gaps that remain", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Independent external-author prompts and holdouts are not complete.",
                "No clinician usability or overtrust study exists.",
                "Long-duration soak, failure injection, database failover, and production concurrency tests remain incomplete.",
                "Browser accessibility testing should expand beyond smoke paths to keyboard, screen-reader semantics, and narrow viewport audits.",
                "Security posture requires deployment-specific secret scanning, dependency remediation, threat modeling, and infrastructure review.",
            ]
        )
    )
    page_break(story)
