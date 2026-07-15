"""Generate the NLCare engineering guide as a reviewer-facing PDF.

The document is intentionally evidence-led. It reads current local evaluation
artifacts at generation time and keeps engineering evidence separate from
clinical evidence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "NLCare_MedicalAgent_Engineering_Guide.pdf"

INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#D8DEE8")
PAPER = colors.HexColor("#F7F8FB")
WHITE = colors.white
TEAL = colors.HexColor("#087F8C")
TEAL_SOFT = colors.HexColor("#E8F6F7")
PINK = colors.HexColor("#D92D75")
PINK_SOFT = colors.HexColor("#FCEBF3")
AMBER = colors.HexColor("#B54708")
AMBER_SOFT = colors.HexColor("#FFF4E5")
RED = colors.HexColor("#B42318")
RED_SOFT = colors.HexColor("#FEECEB")
GREEN = colors.HexColor("#067647")
GREEN_SOFT = colors.HexColor("#EAF8F1")
BLUE = colors.HexColor("#175CD3")
BLUE_SOFT = colors.HexColor("#EAF2FF")


def _load(relative: str, default: Any | None = None) -> Any:
    path = ROOT / relative
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def _dig(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "not reported"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "not reported"
    return f"{100 * value:.{digits}f}%"


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "CoverKicker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=TEAL,
        spaceAfter=5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=32,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=18,
        textColor=MUTED,
        spaceAfter=7 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=INK,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=INK,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        textColor=INK,
        spaceAfter=2.5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.7,
        leading=10.5,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        "BulletCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.3,
        leftIndent=4 * mm,
        firstLineIndent=-2.5 * mm,
        bulletIndent=0,
        textColor=INK,
        spaceAfter=1.6 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.4,
        leading=9,
        textColor=WHITE,
    )
)
styles.add(
    ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        "MetricValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=17,
        textColor=INK,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        "MetricLabel",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        textColor=MUTED,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        "Formula",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=8.1,
        leading=11.5,
        textColor=INK,
        leftIndent=3 * mm,
        rightIndent=3 * mm,
        spaceBefore=1.5 * mm,
        spaceAfter=2 * mm,
    )
)


class AccentRule(Flowable):
    def __init__(self, width: float, color: colors.Color = PINK):
        super().__init__()
        self.width = width
        self.height = 2 * mm
        self.color = color

    def draw(self) -> None:
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, 1.2 * mm, 0.6 * mm, fill=1, stroke=0)


def P(text: str, style: str = "BodyCustom") -> Paragraph:
    return Paragraph(text, styles[style])


def bullets(items: Iterable[str]) -> list[Paragraph]:
    return [P(f"- {item}", "BulletCustom") for item in items]


def callout(title: str, text: str, *, tone: str = "blue") -> Table:
    palette = {
        "blue": (BLUE, BLUE_SOFT),
        "green": (GREEN, GREEN_SOFT),
        "amber": (AMBER, AMBER_SOFT),
        "red": (RED, RED_SOFT),
        "pink": (PINK, PINK_SOFT),
        "teal": (TEAL, TEAL_SOFT),
    }
    foreground, background = palette[tone]
    table = Table(
        [[P(title, "Heading2Custom"), P(text)]],
        colWidths=[42 * mm, 128 * mm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("TEXTCOLOR", (0, 0), (0, 0), foreground),
                ("BOX", (0, 0), (-1, -1), 0.7, foreground),
                ("LINEBEFORE", (0, 0), (0, -1), 3, foreground),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def metric_row(metrics: list[tuple[str, str]], *, tone: str = "teal") -> Table:
    background = TEAL_SOFT if tone == "teal" else PAPER
    cells = [[P(value, "MetricValue") for _, value in metrics], [P(label, "MetricLabel") for label, _ in metrics]]
    table = Table(cells, colWidths=[170 * mm / len(metrics)] * len(metrics), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def data_table(headers: list[str], rows: list[list[Any]], widths: list[float] | None = None) -> LongTable:
    prepared = [[P(str(cell), "TableHeader") for cell in headers]]
    prepared.extend([[P(str(cell), "TableCell") for cell in row] for row in rows])
    table = LongTable(prepared, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for index in range(1, len(prepared)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), PAPER))
    table.setStyle(TableStyle(commands))
    return table


def flow_diagram(steps: list[tuple[str, str]], *, columns: int = 4) -> Table:
    rows: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    for title, detail in steps:
        current.append(P(f"<b>{title}</b><br/><font color='#667085'>{detail}</font>", "TableCell"))
        if len(current) == columns:
            rows.append(current)
            current = []
    if current:
        current.extend([P("")] * (columns - len(current)))
        rows.append(current)
    table = Table(rows, colWidths=[170 * mm / columns] * columns, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PAPER),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def section(story: list[Any], number: str, title: str, subtitle: str = "") -> None:
    story.append(P(f"{number} / NLCare engineering guide", "CoverKicker"))
    story.append(P(title, "SectionTitle"))
    story.append(AccentRule(170 * mm, TEAL if number in {"02", "03", "04", "05", "06"} else PINK))
    story.append(Spacer(1, 2 * mm))
    if subtitle:
        story.append(P(subtitle, "CoverSub"))


def page_break(story: list[Any]) -> None:
    story.append(PageBreak())


def _page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(INK)
    canvas.rect(0, height - 8 * mm, width, 8 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 7.2)
    canvas.setFillColor(WHITE)
    canvas.drawString(20 * mm, height - 5.2 * mm, "NLCARE / MEDICALAGENT")
    canvas.setFont("Helvetica", 6.8)
    canvas.drawRightString(width - 20 * mm, height - 5.2 * mm, "Engineering prototype | Synthetic-only | Not clinically validated")
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 13 * mm, width - 20 * mm, 13 * mm)
    canvas.setFont("Helvetica", 6.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 8.5 * mm, "Reviewer guide generated from local project artifacts")
    canvas.drawRightString(width - 20 * mm, 8.5 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _configuration(artifact: dict[str, Any], config_id: str) -> dict[str, Any]:
    configurations = artifact.get("configurations", {})
    if isinstance(configurations, dict):
        item = configurations.get(config_id, {})
        if isinstance(item, dict):
            summary = item.get("summary")
            return summary if isinstance(summary, dict) else item
        return {}
    for item in configurations:
        if isinstance(item, dict) and (
            item.get("configuration_id") == config_id or item.get("id") == config_id
        ):
            summary = item.get("summary")
            return summary if isinstance(summary, dict) else item
    return {}


def build_story() -> list[Any]:
    rag = _load("Data/evals/rag/latest_rag_baseline_comparison.json")
    prompt_eval = _load("Data/evals/agentic_tool_use/latest_large_scale_agent_prompt_eval.json")
    safety = _load("Data/evals/safety/latest_adversarial_safety_regression.json")
    safety_v4 = _load("Data/evals/safety/latest_adversarial_holdout_v4_baseline.json")
    temporal = _load("Data/evals/models/latest_patient_temporal_cv.json")
    paired = _load("Data/evals/models/latest_paired_model_comparison.json")
    per_head = _load("Data/evals/models/latest_per_head_calibration.json")
    conformal = _load("Data/evals/models/latest_response_conformal_calibration.json")
    latency = _load("Data/evals/ops/latest_route_latency_budget.json")
    sentinel = _load("Data/evals/ops/latest_runtime_quality_sentinel.json")
    release = _load("Data/evals/governance/latest_release_gate_explanation.json")

    bm25 = _configuration(rag, "bm25_only")
    full = _configuration(rag, "hybrid_rrf_query_rewrite_parent_child_source_tier")
    rag_summary = rag.get("summary", {})
    best = _configuration(
        rag,
        str(rag_summary.get("best_configuration", rag.get("best_configuration_id", "hybrid_rrf_query_rewrite"))),
    )
    if not best:
        best = _configuration(rag, "hybrid_rrf_query_rewrite")

    story: list[Any] = []

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

    section(story, "04", "Source-governed RAG", "The retrieval stack is deliberately evaluated against simpler baselines rather than assumed to be better.")
    story.append(
        flow_diagram(
            [
                ("Intent policy", "Select evidence mode and allowed audience/source rules"),
                ("Query rewrite", "Normalize and expand retrieval wording; retain original query"),
                ("Sparse candidates", "BM25 lexical match"),
                ("Dense candidates", "Local embedding similarity and FAISS"),
                ("RRF fusion", "Combine ranks without assuming score comparability"),
                ("Context expansion", "Parent-child windows for local context"),
                ("Source filter", "Tier, allowed use, staleness, patient suitability"),
                ("Answerability", "Cited answer, limited answer, insufficient/conflicting evidence, review, refusal"),
                ("Generate", "Constrained response from retained evidence"),
                ("Validate", "Claim-source support, contradictions, output boundary"),
            ],
            columns=5,
        )
    )
    story.append(P("Core retrieval formulas", "Heading2Custom"))
    story.append(P("BM25(q,d) = sum_t IDF(t) * [tf(t,d)*(k1+1)] / [tf(t,d) + k1*(1-b+b*|d|/avgdl)]", "Formula"))
    story.append(P("cosine(q,d) = (q dot d) / (||q|| * ||d||)", "Formula"))
    story.append(P("RRF(d) = sum_r 1 / (k + rank_r(d))", "Formula"))
    story.append(P("MRR = (1/N) * sum_i 1 / rank_i(first relevant)", "Formula"))
    story.append(P("DCG@k = sum_i=1..k relevance_i / log2(i+1); NDCG@k = DCG@k / ideal_DCG@k", "Formula"))
    story.append(P("Why the layers exist", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "BM25 protects exact terms, drug names, genes, and portal labels; dense search captures paraphrase similarity.",
                "RRF combines rank positions because sparse and dense scores are not directly calibrated.",
                "Rewrite and parent-child expansion are hypotheses, not free improvements; each is measured through ablation.",
                "Source-tier filtering is a governance control. It may correctly reduce patient-facing recall when expected labels point to disallowed evidence.",
                "Answerability and claim validation prevent low-quality retrieval from silently becoming a confident response.",
            ]
        )
    )
    page_break(story)

    section(story, "05", "RAG evidence and negative results", "Current internal frozen-goldset evidence supports governance value, not raw retrieval superiority.")
    comparison_rows = []
    for label, item in [("BM25 only", bm25), ("Best observed internal config", best), ("Full source-governed stack", full)]:
        metrics = item.get("metrics", item)
        comparison_rows.append(
            [
                label,
                _fmt(metrics.get("recall_at_5"), 4),
                _fmt(metrics.get("recall_at_10"), 4),
                _fmt(metrics.get("mrr"), 4),
                _fmt(metrics.get("ndcg_at_10"), 4),
                _fmt(metrics.get("citation_precision"), 4),
                _fmt(metrics.get("source_tier_correctness"), 4),
                _fmt(metrics.get("latency_p95_ms"), 1),
            ]
        )
    story.append(
        data_table(
            ["Configuration", "R@5", "R@10", "MRR", "NDCG@10", "Citation precision", "Tier correctness", "p95 ms"],
            comparison_rows,
            [41 * mm, 15 * mm, 15 * mm, 15 * mm, 19 * mm, 23 * mm, 22 * mm, 20 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "Honest verdict",
            f"On {rag.get('total_n', 'the current')} internal goldset cases, the full source-governed stack has Recall@10 {_fmt(full.get('recall_at_10'), 4)} versus BM25 {_fmt(bm25.get('recall_at_10'), 4)}. The recorded complex-stack delta is {_fmt(rag_summary.get('complex_stack_improvement_over_bm25'), 4)}, and improvement_proven_vs_bm25 is {_fmt(rag_summary.get('improvement_proven_vs_bm25'))}. The complex stack is retained for governance and audience filtering, not advertised as a proven raw-recall improvement.",
            tone="amber",
        )
    )
    story.append(P("Metric interpretation", "Heading2Custom"))
    story.append(
        data_table(
            ["Metric", "What it asks", "What it does not prove"],
            [
                ["Recall@10", "Did an expected source appear in the top ten?", "That the final answer is medically correct"],
                ["MRR", "How early was the first expected source?", "That later context is clean"],
                ["NDCG@10", "How well were graded relevant sources ranked?", "Clinical usefulness"],
                ["Citation precision", "What share of cited chunks matched expected evidence?", "Complete semantic entailment"],
                ["Claim-support rate", "What share of evaluated claims had support?", "Absence of all hallucinations"],
                ["Unsupported-context rate", "How much retained context was not expected?", "Real-world safety"],
                ["Source-tier correctness", "Were audience/source policies followed?", "Retrieval relevance by itself"],
            ],
            [34 * mm, 67 * mm, 69 * mm],
        )
    )
    story.append(P("Known RAG limitations", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Full-stack citation precision is about 0.52 in the current internal comparison, leaving substantial room for better source alignment.",
                "The stage-oracle diagnostic attributes the largest recall loss to source filtering and goldset audience mismatch, which must be adjudicated without weakening governance.",
                "The experimental context pruner increased Recall@5 but reduced citation precision and was not promoted.",
                "The cross-encoder reranker remains experimental because improvement has not been proven.",
                "The no-read external-author RAG holdout is prepared but incomplete; the current metrics are not independent validation.",
            ]
        )
    )
    page_break(story)

    section(story, "06", "Agent and safety evaluation", "Large prompt volume is useful for regression coverage, but does not replace independent red-teaming.")
    total_prompts = prompt_eval.get("prompt_bank_n")
    story.append(
        metric_row(
            [
                ("Synthetic prompt variants", _fmt(total_prompts)),
                ("Classifier pass", _pct(_dig(prompt_eval, "classifier_sweep", "pass_rate"))),
                ("Sampled route accuracy", _pct(_dig(prompt_eval, "bounded_agent_end_to_end_sample", "route_accuracy"))),
                ("Multi-turn pass", _pct(_dig(prompt_eval, "multi_turn_bounded_agent", "conversation_pass_rate"))),
            ]
        )
    )
    story.append(P("What the 5,000-prompt regression covers", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Unsafe and safe-negative prompt families with punctuation noise, wrappers, typos, Taglish/code-switching, emotional phrasing, and near-boundary wording.",
                "A bounded end-to-end sample that checks route selection, verifier results, forbidden medical authority, and unsafe database writes.",
                "Structured symptom, medication, imaging, lab, and treatment-note cases that distinguish logging from conversation.",
                "Seventy two-turn conversations that test urgent, privacy, treatment, cross-patient, structured-update, and topic-change state behavior.",
            ]
        )
    )
    story.append(
        callout(
            "Contamination warning",
            "This bank was used during hardening and is labelled as a tuning-used internal regression. A perfect post-hardening score means known regression families are covered; it does not prove held-out generalization or clinical safety.",
            tone="red",
        )
    )
    story.append(P("Broader adversarial evidence", "Heading2Custom"))
    story.append(
        data_table(
            ["Suite", "N", "Result", "Interpretation"],
            [
                ["Original adversarial regression", _fmt(safety.get("total_n")), _pct(_dig(safety, "metrics", "refusal_correctness")), "Needs attention; privacy and treatment variants remain material"],
                ["Frozen holdout v4 baseline", _fmt(safety_v4.get("total_n")), _pct(safety_v4.get("pass_rate")), "Current visible generalization weakness; do not call safety solved"],
                ["Large tuning-used prompt regression", _fmt(total_prompts), _pct(_dig(prompt_eval, "classifier_sweep", "pass_rate")), "Regression coverage only; not a holdout"],
            ],
            [43 * mm, 18 * mm, 26 * mm, 83 * mm],
        )
    )
    story.append(P("Safety metrics", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Unsafe leakage rate: unsafe cases that reached an unsafe answer or action.",
                "Refusal correctness: unsafe boundary cases refused or redirected as expected.",
                "Escalation correctness: urgent/crisis cases routed to immediate help or clinical review without diagnosis.",
                "Over-refusal: safe educational cases incorrectly blocked; measured separately so stronger guards do not erase legitimate education.",
                "Unsafe write count: structured records created by requests that should never write; the expected value is zero.",
            ]
        )
    )
    page_break(story)

    section(story, "07", "Claim grounding and uncertainty", "Retrieval confidence, claim support, and medical authority are distinct decisions.")
    story.append(
        flow_diagram(
            [
                ("Extract claims", "Split generated text into checkable factual or medical propositions"),
                ("Align evidence", "Map claims to cited snippets and source metadata"),
                ("Check support", "Lexical/semantic support plus deterministic contradiction traps"),
                ("Grade answerability", "Sufficient, limited, insufficient, conflicting, review, or refuse"),
                ("Apply boundary", "Block diagnosis, treatment, dosage, prognosis, genetic/tumor-marker conclusion"),
                ("Finalize", "Keep supported education, soften limited evidence, or replace with safe routing"),
            ],
            columns=3,
        )
    )
    story.append(P("Why a single confidence number is misleading", "Heading2Custom"))
    story.append(
        data_table(
            ["Signal", "Meaning", "Failure mode"],
            [
                ["Retrieval confidence", "Quality and agreement of retrieved evidence", "High similarity can still retrieve the wrong source"],
                ["Source-tier confidence", "Whether evidence meets source and audience policy", "High tier does not guarantee query relevance"],
                ["Citation-support confidence", "Estimated claim-to-snippet support", "Heuristic support can miss negation, temporality, and entity mismatch"],
                ["Evidence conflict flag", "Sources or extracted statements disagree", "Absence of detected conflict is not proof of agreement"],
                ["Answerability state", "Policy decision for response mode", "It is a routing decision, not medical certainty"],
            ],
            [35 * mm, 62 * mm, 73 * mm],
        )
    )
    story.append(P("High-risk contradiction patterns", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Negation inversions: 'no need to contact a clinician' versus evidence that recommends review.",
                "Temporal inversions: prior findings described as current, or current symptoms described as resolved.",
                "Entity mismatches: another patient's record, a different drug, gene, marker, or imaging modality.",
                "Authority escalation: educational evidence converted into diagnosis, treatment, dosage, or prognosis.",
                "Conditional collapse: a statement that applies only under defined circumstances presented as universal.",
                "VUS and tumor-marker overreach: uncertainty or trend context converted into positive mutation or recurrence claims.",
            ]
        )
    )
    story.append(
        callout(
            "Validator limit",
            "The claim validator is an engineering safety layer, not a clinical-grade medical entailment system. When semantic dependencies are unavailable, deterministic and heuristic fallbacks can produce both false confidence and unnecessary refusal. The final authority boundary therefore remains independent of citation support.",
            tone="amber",
        )
    )
    page_break(story)

    section(story, "08", "Synthetic ML and MLE lifecycle", "The ML layer demonstrates lifecycle discipline; it does not demonstrate clinical validity.")
    story.append(
        flow_diagram(
            [
                ("Synthetic journeys", "Versioned longitudinal generator, seeds, missingness, noise scenarios"),
                ("Contracts", "Schema, ranges, identity, date ordering, target definitions"),
                ("Features", "Patient-cycle rows, modality indicators, temporal summaries"),
                ("Splits", "Patient-grouped temporal folds; no patient overlap"),
                ("Train", "Linear, tree, kernel/MLP, temporal and small Transformer candidates"),
                ("Evaluate", "Discrimination, error, calibration, paired tests, subgroup and noise checks"),
                ("Envelope", "Evidence sufficiency, uncertainty, confidence, independent abstention"),
                ("Registry", "Lineage hash, model version, champion/candidate status"),
                ("Promotion gate", "Monitor/review/context only; treatment use is blocked"),
                ("Prediction trace", "Input evidence, missing modalities, output, version, timestamp"),
            ],
            columns=5,
        )
    )
    story.append(P("Model heads and safe boundaries", "Heading2Custom"))
    story.append(
        data_table(
            ["Head", "Engineering target", "Output", "Permitted interpretation"],
            [
                ["Response-pattern classification", "Synthetic binary response-pattern target", "Class probability and label", "Synthetic grouping for review; not personal benefit probability"],
                ["Response-score regression", "Synthetic continuous response score", "Point estimate and uncertainty band", "Simulator-target summary; not tumor response or prognosis"],
                ["Toxicity review signal", "Simulator-built support-review priority", "Review hint / abstention", "Shortcut-prone review signal; not toxicity diagnosis or grade"],
                ["Evidence sufficiency", "Availability and quality of required modalities", "Sufficient / insufficient plus missing data", "Whether the model should speak, not whether the patient is safe"],
            ],
            [42 * mm, 52 * mm, 34 * mm, 42 * mm],
        )
    )
    story.append(P("Predictor classes are not interchangeable", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Imaging/report trends and clinician-reviewed imaging summaries carry the strongest synthetic response-monitoring context.",
                "CBC/labs, symptoms, treatment timing, interruptions, and interventions contribute monitoring and review context.",
                "ER/PR/HER2/Ki-67 are contextual biomarkers, not direct treatment-response proof in this prototype.",
                "Genetic records and family history are for genetic-counseling readiness and context, not mutation inference from scans.",
                "Tumor-marker trends are review context only and cannot prove recurrence or treatment failure.",
                "Treatment history is structured context, never a regimen recommendation target.",
            ]
        )
    )
    page_break(story)

    section(story, "09", "Classification and regression mathematics", "This section explains how the displayed synthetic model numbers are produced.")
    story.append(P("Classification", "Heading2Custom"))
    story.append(P("logit(p) = beta_0 + sum_j beta_j*x_j; p = 1 / (1 + exp(-logit(p)))", "Formula"))
    story.append(P("binary cross-entropy = -(1/N) * sum_i [y_i*log(p_i) + (1-y_i)*log(1-p_i)]", "Formula"))
    story.append(
        P(
            "Linear logistic regression provides a transparent baseline. Tree ensembles and neural candidates learn nonlinear interactions, but are promoted only when paired tests, calibration, robustness, and traceability support the change. A value such as 0.82 means the model assigned 82% probability to its synthetic class under the fitted data distribution. It does not mean an 82% personal chance of treatment success.",
        )
    )
    story.append(P("Regression", "Heading2Custom"))
    story.append(P("prediction = f(x); residual_i = y_i - prediction_i", "Formula"))
    story.append(P("MAE = (1/N) * sum_i |y_i - prediction_i|", "Formula"))
    story.append(P("R^2 = 1 - sum_i residual_i^2 / sum_i (y_i - mean(y))^2", "Formula"))
    story.append(
        P(
            "The response-score regressor estimates a synthetic target created by the simulator. MAE reports average absolute error in that target's units. R-squared reports explained synthetic variance. Neither metric establishes a clinically meaningful treatment-response score.",
        )
    )
    story.append(P("Hybrid display number", "Heading2Custom"))
    story.append(P("hybrid_signal = 0.65 * calibrated_class_probability + 0.35 * normalized_regression_score", "Formula"))
    story.append(
        callout(
            "Display meaning",
            "The hybrid signal is a UI synthesis of two synthetic model heads when both have sufficient evidence. It is not a validated health score. If one head abstains, the UI must disclose missing evidence instead of silently substituting certainty.",
            tone="pink",
        )
    )
    story.append(P("Monitoring context index", "Heading2Custom"))
    story.append(P("index = clamp(base_pattern - min(urgent_flags*12,35) - min(watch_flags*5,20) - min(symptom_severity*1.5,12) - synthetic_lab_penalty, 0, 100)", "Formula"))
    story.append(
        P(
            "The index is an engineering presentation score that begins with the synthetic response-pattern signal and applies capped deductions for review flags, symptom severity, and synthetic-lab provenance. It is explicitly not a personalized health score, clinical severity grade, prognosis, or triage rule.",
        )
    )
    page_break(story)

    section(story, "10", "Calibration, uncertainty, and abstention", "A model can rank well and still be overconfident; each property is measured separately.")
    heads = per_head.get("heads", {})
    classification = heads.get("response_classification", {})
    regression = heads.get("response_regression", {})
    toxicity = heads.get("toxicity", {})
    story.append(
        metric_row(
            [
                ("Synthetic classifier AUROC", _fmt(classification.get("auroc"), 4)),
                ("Synthetic classifier Brier", _fmt(classification.get("brier"), 4)),
                ("Synthetic classifier ECE", _fmt(classification.get("ece"), 4)),
                ("Synthetic regression MAE", _fmt(regression.get("mae"), 4)),
            ]
        )
    )
    story.append(P("Calibration metrics", "Heading2Custom"))
    story.append(P("Brier = (1/N) * sum_i (p_i - y_i)^2", "Formula"))
    story.append(P("ECE = sum_b (n_b/N) * |accuracy_b - confidence_b|", "Formula"))
    story.extend(
        bullets(
            [
                "AUROC measures ranking across thresholds, not probability accuracy and not clinical usefulness.",
                "Brier score measures squared probability error; lower is better, but the target is still synthetic.",
                "ECE summarizes bin-level confidence mismatch and depends on binning choices.",
                "Reliability intervals are included because a single calibration number can hide small-bin uncertainty.",
            ]
        )
    )
    story.append(P("Split-conformal interval", "Heading2Custom"))
    story.append(P("qhat = quantile_{ceil((n+1)*(1-alpha))/n}(|y_cal - prediction_cal|)", "Formula"))
    story.append(P("interval(x) = [prediction(x) - qhat, prediction(x) + qhat]", "Formula"))
    temporal_cv = temporal.get("patient_level_temporal_cv", {})
    story.append(
        data_table(
            ["Conformal item", "Current synthetic artifact"],
            [
                ["Nominal coverage", _fmt(conformal.get("nominal_coverage"), 3)],
                ["Raw coverage", _fmt(conformal.get("raw_coverage"), 3)],
                ["Adjusted coverage", _fmt(conformal.get("adjusted_coverage"), 3)],
                ["Adjusted median band width", _fmt(conformal.get("adjusted_median_band_width"), 3)],
                ["Boundary", conformal.get("claim_boundary", "Synthetic-only interval calibration")],
            ],
            [55 * mm, 115 * mm],
        )
    )
    story.append(P("Abstention", "Heading2Custom"))
    story.append(P("predict only if evidence_sufficient AND confidence >= threshold AND not OOD; otherwise abstain", "Formula"))
    story.append(
        P(
            "Raising a confidence threshold usually reduces coverage while increasing retained-case accuracy. The project tracks that tradeoff explicitly; it does not hide difficult cases by reporting only the retained subset.",
        )
    )
    story.append(
        callout(
            "Toxicity warning",
            f"The toxicity head can show extremely high synthetic discrimination ({_fmt(toxicity.get('auroc'), 4)} in the current per-head artifact), but shortcut audits identify simulator-derived near-label-proxy risk. It remains review-hint-only and must not be presented as a clinical toxicity predictor.",
            tone="red",
        )
    )
    page_break(story)

    section(story, "11", "Statistical testing and temporal design", "The statistics test engineering comparisons inside the synthetic experiment; they do not repair target validity.")
    story.append(P("Patient-level temporal cross-validation", "Heading2Custom"))
    story.append(
        flow_diagram(
            [
                ("Group", "All cycles from one synthetic patient remain together"),
                ("Order", "Train precedes validation in time"),
                ("Walk forward", "Repeat across sequential patient-time folds"),
                ("Audit", "Check patient overlap and temporal violations"),
            ]
        )
    )
    story.append(
        data_table(
            ["Temporal-CV field", "Current artifact", "Interpretation"],
            [
                ["Configured folds", _fmt(temporal.get("n_folds")), f"{_fmt(temporal_cv.get('n_folds_with_auc'))} folds reported AUROC"],
                ["Mean AUROC", _fmt(temporal_cv.get("roc_auc_mean"), 6), "Saturated synthetic result; not clinical discrimination"],
                ["Brier", _fmt(temporal_cv.get("brier_mean"), 6), "Synthetic probability error"],
                ["Patient overlap", _fmt(temporal_cv.get("patient_overlap_pairs")), "Expected zero"],
                ["Temporal violations", _fmt(temporal_cv.get("temporal_violations")), "Must remain visible and investigated"],
            ],
            [40 * mm, 35 * mm, 95 * mm],
        )
    )
    story.append(P("Exact paired classification comparison", "Heading2Custom"))
    story.append(P("McNemar uses discordant pairs b and c: chi_square = (|b-c|-1)^2 / (b+c)", "Formula"))
    story.append(
        P(
            "Because candidate and champion predict the same held-out rows, McNemar's test asks whether one model wins on significantly more discordant examples. It is stronger than comparing two rounded accuracy values but still inherits the synthetic dataset's limitations.",
        )
    )
    class_rows: list[list[Any]] = []
    for item in paired.get("classification_comparisons", paired.get("classification", []))[:4]:
        class_rows.append(
            [
                item.get("candidate", item.get("candidate_model", "candidate")),
                _fmt(item.get("champion_accuracy"), 4),
                _fmt(item.get("candidate_accuracy"), 4),
                _fmt(item.get("accuracy_delta_champion_minus_candidate"), 4),
                _fmt(item.get("p_value"), 6),
            ]
        )
    if class_rows:
        story.append(data_table(["Candidate", "Champion acc.", "Candidate acc.", "Delta", "Exact p"], class_rows, [62 * mm, 27 * mm, 27 * mm, 24 * mm, 30 * mm]))
    story.append(P("Paired bootstrap for regression", "Heading2Custom"))
    story.append(P("For each resample: delta_MAE* = MAE(champion*) - MAE(candidate*)", "Formula"))
    story.append(
        P(
            "Rows are resampled as matched pairs, preserving that both models saw the same examples. The empirical percentile interval estimates uncertainty in the MAE difference. If the interval crosses zero, the comparison does not clearly establish a winner.",
        )
    )
    reg_rows: list[list[Any]] = []
    for item in paired.get("regression_comparisons", paired.get("regression", []))[:4]:
        ci = item.get("bootstrap_ci", item.get("mae_delta_ci", []))
        if isinstance(ci, dict):
            ci_text = f"[{_fmt(ci.get('lower'), 3)}, {_fmt(ci.get('upper'), 3)}]"
        elif isinstance(ci, list) and len(ci) >= 2:
            ci_text = f"[{_fmt(ci[0], 3)}, {_fmt(ci[1], 3)}]"
        elif item.get("ci_low") is not None and item.get("ci_high") is not None:
            ci_text = f"[{_fmt(item.get('ci_low'), 3)}, {_fmt(item.get('ci_high'), 3)}]"
        else:
            ci_text = "not reported"
        reg_rows.append(
            [
                item.get("candidate", item.get("candidate_model", "candidate")),
                _fmt(item.get("champion_mae"), 3),
                _fmt(item.get("candidate_mae"), 3),
                _fmt(item.get("mae_delta_champion_minus_candidate"), 3),
                ci_text,
            ]
        )
    if reg_rows:
        story.append(data_table(["Candidate", "Champion MAE", "Candidate MAE", "Delta", "Bootstrap CI"], reg_rows, [62 * mm, 27 * mm, 27 * mm, 24 * mm, 30 * mm]))
    page_break(story)

    section(story, "12", "Missingness, robustness, and OOD", "Missing data is modeled, disclosed, and allowed to force abstention; it is not replaced by false certainty.")
    story.append(P("Missingness representation", "Heading2Custom"))
    story.append(P("x'_j = impute(x_j); m_j = 1[x_j is missing]; model_input = [x', m]", "Formula"))
    story.extend(
        bullets(
            [
                "Imputed values and missingness indicators are separate features so the model can learn synthetic missingness patterns.",
                "Modality-dropout training simulates absent imaging, CBC, symptoms, demographics, or intervention context.",
                "Each model head has minimum-evidence rules; a classification head can speak while regression abstains, or vice versa.",
                "The response includes modalities present, modalities missing, sufficiency, uncertainty, and an abstention reason.",
                "This demonstrates graceful degradation in the simulator; it does not prove safe missing-data handling for real patients.",
            ]
        )
    )
    story.append(P("OOD and quality gates", "Heading2Custom"))
    story.append(
        data_table(
            ["Gate", "Question", "Safe behavior"],
            [
                ["Schema", "Are fields, units, dates, and categories valid?", "Reject or request correction"],
                ["Range", "Is a value outside configured engineering expectations?", "Flag quality issue; do not infer a diagnosis"],
                ["Distribution", "Is the feature vector far from synthetic training support?", "Mark OOD and abstain/review"],
                ["Modality", "Are required evidence classes absent?", "Independent head abstention"],
                ["Shortcut", "Does a feature behave like a target proxy?", "Block promotion and document the weakness"],
                ["Subgroup floor", "Does performance collapse in a synthetic subgroup?", "Hold promotion and report the slice"],
            ],
            [32 * mm, 70 * mm, 68 * mm],
        )
    )
    story.append(
        callout(
            "Critical distinction",
            "A missingness-aware model can return a number when inputs are incomplete, but it should do so only when that head's evidence rules allow it. Confidence is not a license to invent absent clinical context. The safest valid output may be 'insufficient evidence' with a list of missing modalities.",
            tone="amber",
        )
    )
    page_break(story)

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

    section(story, "18", "Runbook, ports, and repository map", "Use the local URLs for engineering review only. Test-only and optional services are deliberately separated.")
    story.append(P("Default local services", "Heading2Custom"))
    story.append(
        data_table(
            ["Service", "Default URL / port", "Purpose", "Status meaning"],
            [
                ["React frontend", "http://127.0.0.1:5173", "Patient, clinician, and admin UI", "Active only when Vite is running"],
                ["FastAPI backend", "http://127.0.0.1:8017", "API and application services", "Engineering API, not a clinical deployment"],
                ["FastAPI docs", "http://127.0.0.1:8017/docs", "OpenAPI explorer", "Developer surface"],
                ["Ollama", "http://127.0.0.1:11434", "Optional local LLM runtime", "Model availability depends on local installation"],
                ["n8n", "http://127.0.0.1:5678", "Optional automation control plane", "Not started by the core app"],
                ["Pinecone", "No local port", "Optional managed vector backend", "Cloud service; disabled unless configured"],
                ["Playwright frontend", "http://127.0.0.1:5273", "Disposable E2E test server", "Test-only"],
                ["Playwright backend", "http://127.0.0.1:8117", "Disposable isolated-data API", "Test-only"],
            ],
            [34 * mm, 43 * mm, 51 * mm, 42 * mm],
        )
    )
    story.append(P("Repository landmarks", "Heading2Custom"))
    story.append(
        data_table(
            ["Path", "Purpose"],
            [
                ["backend/services/", "Agent, RAG, safety, prediction, trace, and evaluation services"],
                ["backend/routers/", "Role/API surfaces"],
                ["frontend-react/src/", "React/TypeScript product UI"],
                ["tests/", "Backend integration, unit, regression, and governance tests"],
                ["frontend-react/tests/", "Vitest and Playwright suites"],
                ["Data/evals/", "Machine-readable engineering evidence and failure artifacts"],
                ["config/release_gate_thresholds.yaml", "Release tier and threshold policy"],
                ["docs/", "Architecture, eval, review, boundary, and runbook documentation"],
                ["scripts/ship.py", "Cross-platform end-to-end engineering gate"],
            ],
            [67 * mm, 103 * mm],
        )
    )
    story.append(P("Startup checklist", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Confirm environment variables and demo-only database configuration.",
                "Start the FastAPI backend on port 8017 and verify /health or /docs.",
                "Start the Vite frontend on port 5173 and verify login plus patient/clinician/admin smoke paths.",
                "Verify Ollama only if the selected agent configuration requires local generation.",
                "Do not expose n8n, databases, or developer docs publicly without authentication and deployment review.",
            ]
        )
    )
    page_break(story)

    section(story, "19", "Current verdict and next engineering moves", "The best next work improves independent credibility and system simplicity, not feature count.")
    story.append(P("Current constraint-aware verdict", "Heading2Custom"))
    story.append(
        data_table(
            ["Dimension", "Current engineering reading", "Highest-ROI next move"],
            [
                ["AI / RAG", "Layered and inspectable; retrieval superiority and claim precision remain unproven", "Complete no-read holdout and promote entailment-grade claim validation only if it wins"],
                ["Agent safety", "Strong known-regression coverage; held-out variation remains weak", "External/mutation-authored multi-turn red team with safe-negative controls"],
                ["ML / MLE", "Strong lifecycle scaffolding; saturated synthetic evidence", "Noisier generator v2, temporal-violation cleanup, target redesign, and stronger uncertainty slices"],
                ["Medical", "Boundaries are explicit; zero expert review", "One structured oncology nurse/clinician review and one VUS-focused genetic counselor review"],
                ["SWE", "Good modularity and release discipline; local deployment evidence only", "Containerized staging, migration/backup drills, dependency/security remediation, soak/load testing"],
                ["Product", "Usable reviewer surfaces; explanation burden remains high", "Task-based usability and overtrust tests, then simplify labels and dashboards from observed friction"],
            ],
            [32 * mm, 70 * mm, 68 * mm],
        )
    )
    story.append(P("Recommended sequence", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "1. Freeze the current engineering baseline and archive the exact ship artifacts.",
                "2. Complete external no-read RAG and adversarial authoring before more retrieval tuning.",
                "3. Resolve the patient-facing versus clinician-facing goldset adjudication without weakening source policy.",
                "4. Fix the synthetic temporal violations and redesign shortcut-prone targets before adding larger models.",
                "5. Run task-based usability/overtrust sessions using synthetic/demo records only.",
                "6. Add containerized staging, database migrations, backup/restore, observability, rate limiting, and failure injection.",
                "7. Integrate optional n8n/Pinecone only behind measured, reversible adapters.",
                "8. Seek external review; further internal artifact growth has diminishing credibility returns.",
            ]
        )
    )
    story.append(
        callout(
            "What can be claimed",
            "NLCare demonstrates safety-governed RAG, bounded agent workflows, adversarial and metamorphic evaluation, source/claim traceability, synthetic temporal MLE governance, and release discipline in an engineering prototype.",
            tone="green",
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "What still cannot be claimed",
            "Clinical validation, clinician approval, real-world patient safety or benefit, FHIR interoperability, hospital readiness, production healthcare readiness, diagnostic/treatment/prognostic authority, real-patient generalization, or proven raw-retrieval superiority over BM25.",
            tone="red",
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(P("Engineering north star", "Heading2Custom"))
    story.append(
        P(
            "The project's strongest form is not the one with the most agents, metrics, or artifacts. It is the one in which every action is bounded, every number is explainable, every negative result stays visible, every promotion is reversible, and every clinical claim remains below the evidence actually available.",
        )
    )

    return story


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="NLCare / MedicalAgent Engineering Guide",
        author="NLCare engineering prototype",
        subject="SWE, AI engineering, RAG, synthetic MLE, medical safety, and statistics",
    )
    document.build(build_story(), onFirstPage=_page, onLaterPages=_page)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
