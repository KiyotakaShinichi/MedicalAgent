"""Reusable flowable builders shared by every section of the guide.

These wrap reportlab primitives in the document's own vocabulary - a callout, a
metric row, a data table, a flow diagram - so section modules describe content
rather than table geometry. Extracted unchanged from
``scripts/generate_project_guide_pdf.py``.
"""

from __future__ import annotations

from typing import Any, Iterable

from reportlab.lib.units import mm
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from scripts.project_guide.theme import (
    AMBER,
    AMBER_SOFT,
    BLUE,
    BLUE_SOFT,
    GREEN,
    GREEN_SOFT,
    INK,
    LINE,
    PAPER,
    PINK,
    PINK_SOFT,
    RED,
    RED_SOFT,
    TEAL,
    TEAL_SOFT,
    AccentRule,
    styles,
)

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
