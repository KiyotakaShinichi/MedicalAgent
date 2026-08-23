"""Sections 13-15: medical structure and claim boundaries, software
architecture, and caching/observability/automation.

Extracted verbatim from ``build_story`` in
``scripts/generate_project_guide_pdf.py``, which had grown to 1032 lines in a
single function. Flowable content and ordering are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reportlab.lib.units import mm
from scripts.project_guide.components import P, bullets, callout, data_table, flow_diagram, page_break, section

if TYPE_CHECKING:
    from scripts.project_guide.evidence import Evidence

def build(story: list[Any], ev: Evidence) -> None:
    """Append this module's sections to `story`, in order."""
    section(story, "13", "Medical structure and claim boundaries", "The medical layer organizes evidence and review routes while preserving clinician authority.")
    story.append(P("Evidence classes", "Heading2Custom"))
    story.append(
        data_table(
            ["Evidence class", "Examples", "Permitted system use", "Blocked leap"],
            [
                ["Laboratory", "WBC, hemoglobin, platelets, ANC", "Record, trend, disclose reference context, route review", "Diagnose anemia, neutropenia, infection, or prescribe action"],
                ["Symptoms", "Fever, nausea, pain, dyspnea, bleeding", "Capture severity/timing, urgent routing", "Diagnose cause or minimize danger"],
                ["Imaging", "MRI, CT, ultrasound, mammogram text", "Organize clinician-authored wording and temporal context", "Independently declare response, recurrence, or progression"],
                ["Biomarkers", "ER, PR, HER2, Ki-67", "Context and record organization", "Select or change treatment"],
                ["Genetics", "BRCA1/2, PALB2, VUS", "Record and genetic-counselor review readiness", "Infer mutation, quantify inherited risk, treat VUS as positive"],
                ["Tumor markers", "CA 15-3, CA 27.29, CEA", "Trend context and clinician-review questions", "Prove recurrence or treatment failure"],
                ["Treatment", "Cycles, medication, interruptions", "Timeline context", "Recommend start, stop, switch, delay, or dose"],
            ],
            [30 * mm, 39 * mm, 55 * mm, 46 * mm],
        )
    )
    story.append(P("Allowed response functions", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Source-backed general education and portal help.",
                "Patient-entered record organization and missing-data explanation.",
                "Non-diagnostic longitudinal summary with provenance.",
                "Questions to ask a clinician, genetic counselor, or pharmacist.",
                "Urgent or crisis escalation without diagnostic interpretation.",
                "Warm, bounded emotional support before education or review routing.",
            ]
        )
    )
    story.append(P("Always blocked", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Diagnosis confirmation, prognosis, survival estimate, false reassurance, or clinical severity grading.",
                "Treatment recommendation, medication or supplement safety conclusion, or dosage change.",
                "Genetic-risk prediction, VUS-positive interpretation, or tumor-marker recurrence conclusion.",
                "Cross-patient access, private data exfiltration, or prompt-based policy override.",
                "Any wording that implies clinician approval, clinical validation, patient benefit, hospital readiness, or production healthcare readiness.",
            ]
        )
    )
    story.append(
        callout(
            "Review status",
            "Reviewer packets exist for external authors, clinicians or nurses, genetic counselors, and senior MLE reviewers, but no external or clinical review is complete. Prepared packets are process scaffolding, not sign-off.",
            tone="red",
        )
    )
    page_break(story)
    section(story, "14", "Software architecture", "The implementation is modular enough to inspect, test, and replace components without hiding policy inside one agent file.")
    story.append(
        flow_diagram(
            [
                ("React / TypeScript", "Role-isolated patient, clinician, and admin portals"),
                ("FastAPI", "Auth, patient, clinician, admin, model, and eval routers"),
                ("Services", "Intent, safety, retrieval, validation, prediction, tracing"),
                ("Persistence", "Application DB, evaluation JSON/JSONL/CSV, model and KB artifacts"),
                ("Local inference", "Ollama-compatible LLM endpoint where configured"),
                ("Vector retrieval", "Local FAISS by default; optional Pinecone adapter boundary"),
                ("Automation", "Optional n8n workflows for non-clinical ops and review queues"),
                ("Quality gates", "Pytest, Vitest, Playwright, lint, build, release artifacts"),
            ]
        )
    )
    story.append(P("Frontend engineering", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Reusable cards, badges, error panes, modals/drawers, quick actions, trace panels, and API hooks.",
                "Patient tools are grouped behind the chat composer attachment control rather than occupying the conversation surface.",
                "Fixed-size dashboard structures and responsive grids protect against label and card reflow.",
                "Clinical constants, units, caveats, and labels are centralized to reduce wording drift.",
                "Every key synthetic metric now includes 'Meaning' and 'How calculated' help text.",
            ]
        )
    )
    story.append(P("Backend engineering", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Responsibility-named services separate intent classification, retrieval, source policy, claim checks, finalization, caching, and traces.",
                "Pydantic/API schemas and role checks constrain request shapes and ownership.",
                "Prediction traces are optional and transactionally isolated; disabling trace persistence does not leave uncommitted rows.",
                "Evaluation runners emit machine-readable artifacts with schema versions, timestamps, claim boundaries, and contamination metadata.",
                "Cross-platform scripts avoid requiring GNU Make on Windows.",
            ]
        )
    )
    story.append(P("Primary technical debt", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "The artifact count is high and must stay tiered so informational scaffolds cannot dilute hard blockers.",
                "Some test/demo synchronization paths emit SQLAlchemy identity-map warnings and should be cleaned before persistent deployment.",
                "A local single-process setup is suitable for demonstration, not evidence of resilient distributed operation.",
                "Secrets, database migrations, backups, SLOs, and disaster recovery need deployment-specific validation before public hosting.",
            ]
        )
    )
    page_break(story)
    section(story, "15", "Caching, observability, and automation", "Performance controls are safety-aware and automation is kept out of clinical decision authority.")
    story.append(P("Safety-aware cache", "Heading2Custom"))
    story.append(
        flow_diagram(
            [
                ("Gate request", "Exclude urgent, patient-specific, privacy, genetics-risk, or treatment-decision content"),
                ("Build key", "Exact or semantic key plus policy context"),
                ("Bind KB", "Knowledge-base fingerprint prevents stale evidence reuse"),
                ("Validate hit", "TTL, citations, source policy, answerability"),
                ("Return / miss", "Reuse only safe education; otherwise run full pipeline"),
            ],
            columns=5,
        )
    )
    story.append(P("Trace contract", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Request and correlation IDs connect frontend actions, backend routes, retrieval, model inference, and evaluation records.",
                "Traces store decisions and evidence metadata, not hidden reasoning or private chain-of-thought.",
                "Latency is decomposed by routing, retrieval, generation, validation, cache, and tool execution where available.",
                "Admin dashboards expose failures and negative results rather than only green aggregate scores.",
            ]
        )
    )
    story.append(P("n8n integration boundary", "Heading2Custom"))
    story.append(
        data_table(
            ["Good automation", "Trigger", "Required controls"],
            [
                ["Evaluation refresh", "Scheduled or commit-tagged run", "Read-only inputs, artifact lineage, failure notification"],
                ["Review queue notification", "New synthetic/demo review item", "No medical interpretation; role-scoped links"],
                ["Artifact freshness monitor", "Stale critical eval", "Idempotency, deduplication, audit log"],
                ["Backup/export", "Scheduled demo-data snapshot", "Encryption, retention policy, no real patient data"],
                ["Deployment smoke", "Post-deploy hook", "Health checks, rollback signal, no autonomous promotion"],
            ],
            [43 * mm, 49 * mm, 78 * mm],
        )
    )
    story.append(P("Pinecone integration boundary", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Treat Pinecone as an optional vector-index implementation behind the existing retrieval interface, not as a quality claim.",
                "Namespaces must separate environment, corpus version, audience policy, and source tier.",
                "Every vector record must preserve canonical source ID, chunk ID, parent ID, tier, allowed use, staleness, and KB fingerprint.",
                "Run the same frozen baseline comparison against FAISS and Pinecone before promotion; record recall, grounding, policy correctness, latency, and cost.",
                "Keep local FAISS as a deterministic fallback and do not route patient-facing content to a remote service without an approved data policy.",
            ]
        )
    )
    story.append(
        callout(
            "Industry-ready means measurable",
            "Adding n8n or Pinecone does not make a healthcare product production-ready. The credible gain is a replaceable interface, idempotent workflows, provenance, environment isolation, failure handling, cost/latency measurement, and a reversible promotion decision.",
            tone="teal",
        )
    )
    page_break(story)
