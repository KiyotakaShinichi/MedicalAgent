"""Generate the NLCare engineering guide as a reviewer-facing PDF.

The document is intentionally evidence-led. It reads current local evaluation
artifacts at generation time and keeps engineering evidence separate from
clinical evidence.

The content itself lives in ``scripts/project_guide``: `theme` (palette, styles,
page frame), `components` (reusable flowables), `evidence` (artifact loading),
and `sections` (one module per group of document sections). This module keeps
the CLI and the document setup, which is all a caller needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate

# `python scripts/generate_project_guide_pdf.py` puts only `scripts/` on
# sys.path, so the repository root is added before importing the package.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.project_guide import build_story  # noqa: E402
from scripts.project_guide.theme import OUTPUT, _page  # noqa: E402

__all__ = ["OUTPUT", "build_story", "main"]


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
