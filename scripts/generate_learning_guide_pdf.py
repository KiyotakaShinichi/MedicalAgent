"""Render the NLCare Engineering Academy Markdown guide as a polished PDF."""
from __future__ import annotations

import html
import hashlib
import re
import textwrap
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "NLCARE_PROJECT_LEARNING_GUIDE.md"
OUTPUT = ROOT / "output" / "pdf" / "NLCare_Engineering_Academy.pdf"

INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#D8DEE8")
PAPER = colors.HexColor("#F7F8FB")
TEAL = colors.HexColor("#087F8C")
TEAL_SOFT = colors.HexColor("#E8F6F7")
PINK = colors.HexColor("#D92D75")
PINK_SOFT = colors.HexColor("#FCEBF3")
AMBER = colors.HexColor("#B54708")
AMBER_SOFT = colors.HexColor("#FFF4E5")
RED = colors.HexColor("#B42318")
WHITE = colors.white

PAGE_W, PAGE_H = A4
LEFT = 20 * mm
RIGHT = 18 * mm
TOP = 19 * mm
BOTTOM = 18 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=TEAL,
            spaceAfter=5 * mm,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=18,
            textColor=MUTED,
            spaceAfter=7 * mm,
        ),
        "part": ParagraphStyle(
            "Part",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            textColor=TEAL,
            spaceBefore=2 * mm,
            spaceAfter=6 * mm,
            keepWithNext=True,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=INK,
            spaceBefore=5 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=INK,
            spaceBefore=4 * mm,
            spaceAfter=1.8 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.5,
            textColor=TEAL,
            spaceBefore=3 * mm,
            spaceAfter=1.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.9,
            leading=13,
            textColor=INK,
            spaceAfter=2.4 * mm,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=12.3,
            leftIndent=5 * mm,
            firstLineIndent=-3.5 * mm,
            textColor=INK,
            spaceAfter=1.4 * mm,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=13,
            textColor=INK,
            leftIndent=5 * mm,
            rightIndent=4 * mm,
            spaceAfter=1.5 * mm,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.3,
            leading=10.2,
            textColor=INK,
            leftIndent=3 * mm,
            rightIndent=3 * mm,
            borderPadding=3 * mm,
            borderWidth=0.5,
            borderColor=LINE,
            borderRadius=2,
            backColor=PAPER,
            spaceBefore=1 * mm,
            spaceAfter=3 * mm,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9,
            textColor=WHITE,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=9.3,
            textColor=INK,
        ),
        "toc_title": ParagraphStyle(
            "TocTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=INK,
            spaceAfter=5 * mm,
        ),
        "toc0": ParagraphStyle(
            "TOC0",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=TEAL,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=2 * mm,
        ),
        "toc1": ParagraphStyle(
            "TOC1",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            textColor=INK,
            leftIndent=6 * mm,
            firstLineIndent=0,
        ),
    }


STYLES = _styles()


def _inline(text: str) -> str:
    """Convert the small inline-Markdown subset used by the guide."""
    escaped = html.escape(text, quote=False)
    code_values: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_values.append(match.group(1))
        return f"@@CODE{len(code_values) - 1}@@"

    escaped = re.sub(r"`([^`]+)`", stash_code, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\[([^]]+)]\(([^)]+)\)", r'<link href="\2" color="#087F8C">\1</link>', escaped)
    for index, value in enumerate(code_values):
        escaped = escaped.replace(
            f"@@CODE{index}@@",
            f'<font name="Courier" color="#B42318">{value}</font>',
        )
    return escaped


def _table_widths(rows: list[list[str]]) -> list[float]:
    n_cols = max(len(row) for row in rows)
    weights: list[float] = []
    for col in range(n_cols):
        max_len = max((len(row[col]) if col < len(row) else 0) for row in rows)
        weights.append(min(3.2, max(0.8, max_len / 28)))
    total = sum(weights)
    return [CONTENT_W * weight / total for weight in weights]


def _make_table(rows: list[list[str]]) -> LongTable:
    n_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (n_cols - len(row)) for row in rows]
    formatted: list[list[Paragraph]] = []
    for row_index, row in enumerate(normalized):
        style = STYLES["table_header"] if row_index == 0 else STYLES["table_cell"]
        formatted.append([Paragraph(_inline(cell), style) for cell in row])
    table = LongTable(formatted, colWidths=_table_widths(normalized), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAPER]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.6 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * mm),
            ]
        )
    )
    return table


def _wrap_code(lines: list[str], width: int = 96) -> str:
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
            continue
        indent = re.match(r"\s*", line).group(0)
        wrapped.extend(
            textwrap.wrap(
                line,
                width=width,
                subsequent_indent=indent + "  ",
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped)


class LearningGuideDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="NLCare Engineering Academy",
            author="NLCare / MedicalAgent",
            subject="A complete repository-grounded learning guide",
        )
        frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, id="normal")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=_draw_page))

    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"Part", "H1"}:
            return
        text = flowable.getPlainText()
        level = 0 if style_name == "Part" or text == "How to use this guide" else 1
        key = "heading-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def _draw_page(canvas: object, doc: BaseDocTemplate) -> None:
    canvas.saveState()
    page_num = canvas.getPageNumber()
    if page_num > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.4)
        canvas.line(LEFT, PAGE_H - 12 * mm, PAGE_W - RIGHT, PAGE_H - 12 * mm)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(TEAL)
        canvas.drawString(LEFT, PAGE_H - 9.5 * mm, "NLCARE ENGINEERING ACADEMY")
        canvas.setFont("Helvetica", 6.7)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 9.5 * mm, "Synthetic-only, nonclinical engineering guide")
        canvas.line(LEFT, 12 * mm, PAGE_W - RIGHT, 12 * mm)
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(LEFT, 8.8 * mm, "Not medical training or clinical validation")
        canvas.drawRightString(PAGE_W - RIGHT, 8.8 * mm, f"Page {page_num}")
    canvas.restoreState()


def _cover() -> list[Flowable]:
    boundary = Table(
        [[Paragraph(
            "<b>Clinical boundary</b><br/>Engineering prototype only. Synthetic-only data and model signals. "
            "No clinical validation, patient-care authority, clinician approval, or production healthcare readiness.",
            STYLES["quote"],
        )]],
        colWidths=[CONTENT_W],
    )
    boundary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PINK_SOFT),
                ("BOX", (0, 0), (-1, -1), 0.7, PINK),
                ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
            ]
        )
    )
    track = Table(
        [
            [Paragraph("45", STYLES["cover_title"]), Paragraph("12", STYLES["cover_title"]), Paragraph("40", STYLES["cover_title"])],
            [Paragraph("chapters", STYLES["quote"]), Paragraph("hands-on labs", STYLES["quote"]), Paragraph("mastery questions", STYLES["quote"])],
        ],
        colWidths=[CONTENT_W / 3] * 3,
    )
    track.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("BACKGROUND", (0, 0), (-1, -1), TEAL_SOFT),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    return [
        Spacer(1, 20 * mm),
        Paragraph("NLCARE / MEDICALAGENT", STYLES["cover_kicker"]),
        Paragraph("NLCare Engineering Academy", STYLES["cover_title"]),
        Paragraph(
            "A complete learning guide to the AI, RAG, agentic, ML, statistics, XAI, software, data, automation, infrastructure, security, deployment, and medical-governance concepts in this repository.",
            STYLES["cover_subtitle"],
        ),
        boundary,
        Spacer(1, 11 * mm),
        track,
        Spacer(1, 13 * mm),
        Paragraph(
            "Study the mechanism. Reproduce the evidence. Inspect the failures. Defend the claim boundary.",
            STYLES["cover_subtitle"],
        ),
        PageBreak(),
    ]


def _toc() -> list[Flowable]:
    toc = TableOfContents()
    toc.levelStyles = [STYLES["toc0"], STYLES["toc1"]]
    return [
        Paragraph("Contents", STYLES["toc_title"]),
        Paragraph(
            "Parts appear in teal; chapters are indented. The PDF is generated from the editable Markdown source.",
            STYLES["body"],
        ),
        toc,
        PageBreak(),
    ]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*", line))


def _table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_markdown(lines: list[str]) -> list[Flowable]:
    story: list[Flowable] = []
    index = 0
    in_code = False
    code_lines: list[str] = []

    while index < len(lines):
        raw = lines[index].rstrip("\n")
        stripped = raw.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(XPreformatted(_wrap_code(code_lines), STYLES["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(raw)
            index += 1
            continue
        if not stripped:
            index += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 1.5 * mm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=LINE, spaceBefore=1 * mm, spaceAfter=2 * mm))
            index += 1
            continue

        if stripped.startswith("# "):
            if stripped.startswith("# Part "):
                if story and not isinstance(story[-1], PageBreak):
                    story.append(PageBreak())
                story.append(Paragraph(_inline(stripped[2:]), STYLES["part"]))
            else:
                story.append(Paragraph(_inline(stripped[2:]), STYLES["part"]))
            index += 1
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(_inline(stripped[3:]), STYLES["h1"]))
            index += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(_inline(stripped[4:]), STYLES["h2"]))
            index += 1
            continue
        if stripped.startswith("#### "):
            story.append(Paragraph(_inline(stripped[5:]), STYLES["h3"]))
            index += 1
            continue

        if stripped.startswith("> "):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("> "):
                quote_lines.append(lines[index].strip()[2:])
                index += 1
            quote = Table(
                [[Paragraph(_inline(" ".join(quote_lines)), STYLES["quote"])]],
                colWidths=[CONTENT_W],
            )
            quote.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), AMBER_SOFT),
                ("LINEBEFORE", (0, 0), (0, -1), 2.2, AMBER),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
            ]))
            story.append(quote)
            story.append(Spacer(1, 2 * mm))
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and _is_separator(lines[index + 1]):
            rows = [_table_row(raw)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(_table_row(lines[index]))
                index += 1
            story.append(_make_table(rows))
            story.append(Spacer(1, 3 * mm))
            continue

        if re.match(r"^-\s+", stripped):
            item = re.sub(r"^-\s+", "", stripped)
            story.append(Paragraph(f"- {_inline(item)}", STYLES["bullet"]))
            index += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            story.append(Paragraph(_inline(stripped), STYLES["bullet"]))
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            nxt = lines[index].strip()
            if not nxt or nxt.startswith(("#", "- ", "> ", "```", "|")) or nxt == "---" or re.match(r"^\d+\.\s+", nxt):
                break
            paragraph_lines.append(nxt)
            index += 1
        story.append(Paragraph(_inline(" ".join(paragraph_lines)), STYLES["body"]))

    if in_code and code_lines:
        story.append(XPreformatted(_wrap_code(code_lines), STYLES["code"]))
    return story


def build_pdf(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = source.read_text(encoding="utf-8").splitlines()
    story: list[Flowable] = []
    story.extend(_cover())
    story.extend(_toc())
    # Skip cover metadata through the first horizontal rule. It is represented on the PDF cover.
    first_rule = next((i for i, line in enumerate(lines) if line.strip() == "---"), -1)
    story.extend(_parse_markdown(lines[first_rule + 1 :]))
    doc = LearningGuideDocTemplate(str(output))
    doc.multiBuild(story)
    return output


if __name__ == "__main__":
    result = build_pdf()
    print(result)
