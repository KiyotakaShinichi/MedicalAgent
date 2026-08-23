"""The reviewer-facing NLCare engineering guide, as composable parts.

``scripts/generate_project_guide_pdf.py`` was a single 1466-line module whose
``build_story`` alone ran to 1032 lines. It is split here into presentation
(`theme`), reusable flowables (`components`), artifact loading (`evidence`),
and one module per group of document sections (`sections`). The entrypoint
keeps the CLI and the document setup.
"""

from __future__ import annotations

from typing import Any

from scripts.project_guide.evidence import Evidence
from scripts.project_guide.sections import SECTION_BUILDERS


def build_story() -> list[Any]:
    """The full flowable stream for the guide, in document order."""
    evidence = Evidence.load()
    story: list[Any] = []
    for build_section in SECTION_BUILDERS:
        build_section(story, evidence)
    return story


__all__ = ["Evidence", "SECTION_BUILDERS", "build_story"]
