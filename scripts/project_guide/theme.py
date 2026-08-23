"""Page geometry, palette, paragraph styles, and the running page frame.

Everything here is presentation-only: no evidence is read and no document
content is decided. Split out of ``scripts/generate_project_guide_pdf.py`` so
the section modules can share one visual definition instead of reaching into
the entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Flowable


ROOT = Path(__file__).resolve().parents[2]
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
