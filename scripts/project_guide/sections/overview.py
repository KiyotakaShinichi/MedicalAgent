"""Cover page and sections 01-03: what the system is, the product
surfaces, and the bounded-agent architecture.

Extracted verbatim from ``build_story`` in
``scripts/generate_project_guide_pdf.py``, which had grown to 1032 lines in a
single function. Flowable content and ordering are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datetime import datetime, timezone
from reportlab.lib.units import mm
from reportlab.platypus import Spacer
from scripts.project_guide.components import P, bullets, callout, data_table, flow_diagram, metric_row, page_break, section
from scripts.project_guide.evidence import _fmt
from scripts.project_guide.theme import AccentRule, PINK, ROOT

if TYPE_CHECKING:
    from scripts.project_guide.evidence import Evidence

def build(story: list[Any], ev: Evidence) -> None:
    """Append this module's sections to `story`, in order."""
    rag = ev.rag
    prompt_eval = ev.prompt_eval

    story.extend(
        [
            Spacer(1, 16 * mm),
            P("REVIEWER-FACING ENGINEERING GUIDE", "CoverKicker"),
            P("NLCare / MedicalAgent", "CoverTitle"),
            P(
                "Safety-governed breast cancer monitoring engineering prototype with source-governed RAG, bounded agent workflows, synthetic temporal ML, trace diagnostics, and release-gated governance.",
                "CoverSub",
            ),
            AccentRule(170 * mm, PINK),
            Spacer(1, 8 * mm),
            callout(
                "Clinical boundary",
                "This is a synthetic-only, non-diagnostic engineering prototype. It has no real patient data, no clinician-reviewed labels, no IRB or ethics approval, no clinician sign-off, and no clinical validation. It must not be used for diagnosis, treatment, prognosis, dosage, genetic-risk interpretation, tumor-marker conclusions, or real patient care.",
                tone="red",
            ),
            Spacer(1, 7 * mm),
            metric_row(
                [
                    ("Large internal prompt bank", _fmt(prompt_eval.get("prompt_bank_n"))),
                    ("RAG goldset cases", _fmt(rag.get("total_n"))),
                    ("Release artifacts", "149"),
                    ("Clinical validation", "FALSE"),
                ]
            ),
            Spacer(1, 9 * mm),
            P("Purpose", "Heading2Custom"),
            P(
                "This document explains what the system does, how the numbers are produced, what was actually measured, where the architecture helps, where it does not, and how to run and review the project without turning engineering evidence into medical authority. It is both a project guide and an audit trail for technical reviewers.",
            ),
            P("Generated", "Heading2Custom"),
            P(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
            P("Canonical project root", "Heading2Custom"),
            P(str(ROOT).replace("&", "&amp;"), "Formula"),
        ]
    )
    page_break(story)
    section(story, "01", "Executive map", "The project is a monitored workflow, not an autonomous medical decision-maker.")
    story.append(
        flow_diagram(
            [
                ("Patient portal", "Chat, symptom/lab/imaging/medication capture, timeline, record summaries"),
                ("Safety scope", "Urgency, privacy, medical-authority, injection, cross-patient boundary"),
                ("Bounded agent", "Answer, retrieve, clarify, save a reviewed structure, refuse, or escalate"),
                ("Source governance", "Allowed-use, source tier, staleness, audience suitability"),
                ("RAG", "BM25 + dense candidates, RRF, rewrite, expansion, citations"),
                ("Output checks", "Claim support, contradiction traps, post-generation safety"),
                ("Synthetic ML", "Classification, regression, review hints, abstention envelopes"),
                ("Audit surfaces", "Traces, eval artifacts, dashboards, release gates"),
            ]
        )
    )
    story.append(P("What is genuinely strong", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "The system separates deterministic safety checks, retrieval, generation, validation, tool execution, and final verification into inspectable stages.",
                "Patient-facing retrieval enforces audience and source-tier policies even when that reduces raw recall.",
                "Structured updates require an explicit data-bearing intent; vague conversation is not supposed to become a database write.",
                "Every synthetic model head can abstain independently when modalities or evidence are insufficient.",
                "Negative results remain visible: retrieval lift over BM25 is not proven, the experimental context pruner was not promoted, and synthetic scores are not presented as clinical evidence.",
            ]
        )
    )
    story.append(P("What remains weak", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "No external-author RAG holdout, clinician review, genetic counselor review, real cohort validation, or production traffic evidence is complete.",
                "Claim validation still relies on deterministic and heuristic fallbacks when a stronger entailment model is unavailable.",
                "The internal RAG goldset contains patient-facing versus clinician-facing label tension around source filtering.",
                "Adversarial safety performance varies sharply by holdout design; the frozen v4 baseline remains a visible generalization weakness.",
                "Latency samples are sparse and do not establish production readiness.",
            ]
        )
    )
    page_break(story)
    section(story, "02", "Patient and reviewer workflows", "The product surface is organized around record capture, explanation, escalation, and review.")
    story.append(P("Patient workflow", "Heading2Custom"))
    story.append(
        flow_diagram(
            [
                ("1. Ask or record", "Natural-language chat or explicit form/tool selection"),
                ("2. Route", "Conversation, RAG education, structured update, refusal, or escalation"),
                ("3. Confirm", "Clarify missing fields and verify intended write"),
                ("4. Persist", "Store only validated structured records with trace metadata"),
                ("5. Explain", "Show source-backed context and what each displayed number means"),
                ("6. Review", "Surface records and non-diagnostic signals for care-team discussion"),
            ],
            columns=3,
        )
    )
    story.append(P("Automatic logging behavior", "Heading2Custom"))
    story.append(
        callout(
            "No silent writes",
            "A phrase such as 'I have an upset stomach' may be interpreted as a symptom-reporting intent, but the safe workflow should extract a candidate record and ask for missing details or confirmation before persistence. Casual context, distress language, hypotheticals, negation, or a statement about another person must not create a patient record. Explicit form submission remains the clearest path.",
            tone="amber",
        )
    )
    story.append(P("Role separation", "Heading2Custom"))
    story.append(
        data_table(
            ["Role", "Primary view", "Allowed use", "Boundary"],
            [
                ["Patient", "Timeline, support chat, record forms", "Record organization and source-backed education", "No diagnosis, treatment choice, dosage, prognosis, or genetic interpretation"],
                ["Clinician reviewer", "Patient records, review queue, model evidence", "Review synthetic signals and patient-entered context", "No implied approval; outputs remain unvalidated"],
                ["Admin / MLE reviewer", "Eval artifacts, traces, release gates", "Engineering quality, failures, latency, model governance", "Cannot convert internal metrics into clinical evidence"],
            ],
            [25 * mm, 42 * mm, 55 * mm, 48 * mm],
        )
    )
    story.append(P("User-facing explanation contract", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Every synthetic score is labelled as synthetic and non-clinical.",
                "Each dashboard metric exposes a short 'Meaning' and 'How calculated' explanation.",
                "Probability means model class probability on synthetic data, not a personal chance of benefit or outcome.",
                "Confidence reflects evidence availability and uncertainty rules, not medical certainty.",
                "Reference bands are defaults for engineering display and are not personalized medical interpretation.",
            ]
        )
    )
    page_break(story)
    section(story, "03", "AI and bounded-agent architecture", "A stateful router chooses from a restricted action vocabulary and verifies the final action.")
    story.append(
        flow_diagram(
            [
                ("Normalize", "Unicode NFKC, control removal, typo and noisy punctuation handling"),
                ("Scope", "Security, medical boundary, urgency, distress, unsafe semantic intent"),
                ("Resolve state", "Recent safety boundary, multi-turn context, TTL, explicit reset"),
                ("Choose action", "Answer, retrieve, clarify, stage update, refuse, escalate"),
                ("Execute", "Read-only retrieval or schema-validated write"),
                ("Verify", "Route/action consistency, forbidden-authority check, unsafe-write check"),
                ("Finalize", "Patient-safe response, citations when appropriate"),
                ("Trace", "Decision metadata, IDs, timing, guardrail status; no private chain-of-thought"),
            ]
        )
    )
    story.append(P("Multi-turn safety state", "Heading2Custom"))
    story.append(
        callout(
            "Boundary carryover",
            "When a user follows an urgent or unsafe request with a short phrase such as 'go where?', 'what dose?', or 'send that too', the router preserves the prior boundary for a limited number of turns. The state has an explicit turn budget, decays, and is cleared when the conversation safely changes topic.",
            tone="teal",
        )
    )
    story.append(P("Tool-use invariants", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "A write requires a recognized tool intent, schema-valid fields, patient/session ownership, and verifier approval.",
                "Vague upload language does not fabricate imaging findings; report text, impression, or structured metadata is required.",
                "Retrospective notes can be organized, but directive treatment-change or dosage requests are blocked from tool execution.",
                "Cross-patient requests, hidden-instruction attempts, and privacy-sensitive data requests never inherit permission from chat context.",
                "Failure defaults to clarification, refusal, or review routing, not a guessed write.",
            ]
        )
    )
    story.append(P("Agent actions are bounded", "Heading2Custom"))
    story.append(
        data_table(
            ["Action", "When used", "Safety property"],
            [
                ["answer", "Low-risk conversation or portal help", "No medical authority"],
                ["retrieve", "Source-backed education is answerable", "Governed evidence and citation checks"],
                ["clarify", "Intent or required structured fields are incomplete", "Prevents guessed records"],
                ["stage_update", "Explicit symptom/lab/imaging/medication/note intent", "Schema and ownership validation before commit"],
                ["refuse", "Privacy, exfiltration, diagnosis/treatment/prognosis boundary", "No tool call and no unsafe citation"],
                ["escalate", "Urgent symptoms, crisis, or clinician review required", "Warm handoff without diagnosis"],
            ],
            [28 * mm, 72 * mm, 70 * mm],
        )
    )
    page_break(story)
