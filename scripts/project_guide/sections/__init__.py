"""The guide's sections, in document order.

`SECTION_BUILDERS` is the running order of the document. Each builder appends
its flowables to the shared story, so the tuple - not import order - is what
decides what a reader sees first.
"""

from __future__ import annotations

from typing import Any, Callable

from scripts.project_guide.sections import overview
from scripts.project_guide.sections import retrieval
from scripts.project_guide.sections import safety
from scripts.project_guide.sections import modeling
from scripts.project_guide.sections import statistics
from scripts.project_guide.sections import architecture
from scripts.project_guide.sections import operations
from scripts.project_guide.sections import runbook

if True:  # keep the builders' order explicit and reviewable
    SECTION_BUILDERS: tuple[Callable[[list[Any], Any], None], ...] = (
        overview.build,
        retrieval.build,
        safety.build,
        modeling.build,
        statistics.build,
        architecture.build,
        operations.build,
        runbook.build,
    )

__all__ = ["SECTION_BUILDERS"]
