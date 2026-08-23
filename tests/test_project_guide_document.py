"""The engineering guide PDF survived decomposition unchanged.

`scripts/generate_project_guide_pdf.py` was a 1466-line module whose
`build_story` alone ran to 1032 lines. The document now lives in
`scripts/project_guide/` - `theme`, `components`, `evidence`, and one module per
group of sections - and the entrypoint keeps only the CLI.

That split is only safe if the rendered document is the same one. This file
replays `tests/contracts/project_guide_baseline.json`, a snapshot of the
flowable stream taken from the pre-decomposition `build_story`, and requires an
exact match on every flowable's type, style, text, and table contents, in order.

Two things are deliberately normalised out, because they are not properties of
the refactor:

* the document's own generation timestamp, which differs between any two runs;
* nothing else - evidence values are *not* normalised. If an artifact stops
  loading, its cells become "not reported" and these tests fail, which is the
  intended behaviour: it is exactly the bug a path change in `theme.ROOT`
  introduced during this refactor.

The baseline is a historical record. A deliberate future edit to the guide
should update it in the same commit, so the diff shows the document changing on
purpose.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_PATH = ROOT / "tests" / "contracts" / "project_guide_baseline.json"

pytest.importorskip("reportlab", reason="reportlab backs the guide PDF")

from scripts.project_guide import build_story  # noqa: E402
from scripts.project_guide.evidence import Evidence  # noqa: E402
from scripts.project_guide.sections import SECTION_BUILDERS  # noqa: E402
from scripts.project_guide.theme import OUTPUT, ROOT as THEME_ROOT  # noqa: E402

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC")


def _describe(flowable) -> str:
    """One stable line per flowable: what a reader would actually see."""
    name = type(flowable).__name__
    if name == "Paragraph":
        style = getattr(getattr(flowable, "style", None), "name", "?")
        return f"Paragraph[{style}]:{getattr(flowable, 'text', '')}"
    if name == "Spacer":
        return f"Spacer:{getattr(flowable, 'width', '?')}x{getattr(flowable, 'height', '?')}"
    if name == "PageBreak":
        return "PageBreak"
    if name == "AccentRule":
        return f"AccentRule:{getattr(flowable, 'width', '?')}:{getattr(flowable, 'color', '?')}"
    if name in ("Table", "LongTable"):
        rows = []
        for row in getattr(flowable, "_cellvalues", []):
            cells = []
            for cell in row:
                if hasattr(cell, "text"):
                    cells.append(str(cell.text))
                elif isinstance(cell, list):
                    cells.append("|".join(str(getattr(x, "text", x)) for x in cell))
                else:
                    cells.append(str(cell))
            rows.append(cells)
        return f"{name}:{json.dumps(rows, sort_keys=True)}"
    return f"{name}:{repr(flowable)[:200]}"


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def story() -> list:
    return build_story()


@pytest.fixture(scope="module")
def lines(story: list) -> list[str]:
    return [_TIMESTAMP.sub("{generated}", _describe(f)) for f in story]


# ─── the equivalence proof ───────────────────────────────────────────────────


def test_flowable_count_is_unchanged(baseline: dict, story: list) -> None:
    assert len(story) == baseline["flowable_count"]


def test_every_flowable_is_identical(baseline: dict, lines: list[str]) -> None:
    """Reported per flowable, so a failure names the one that moved."""
    expected = baseline["lines"]
    for index, (before, after) in enumerate(zip(expected, lines)):
        assert after == before, (
            f"flowable {index} changed during decomposition:\n"
            f"  before: {before[:400]}\n  after:  {after[:400]}"
        )


def test_story_digest_matches(baseline: dict, lines: list[str]) -> None:
    """Whole-document check, catching anything the positional loop misses."""
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    assert digest == baseline["story_digest"]


def test_heading_order_is_unchanged(baseline: dict, story: list) -> None:
    headings = [
        getattr(f, "text", "")
        for f in story
        if type(f).__name__ == "Paragraph"
        and getattr(getattr(f, "style", None), "name", "")
        in ("SectionTitle", "Heading2Custom")
    ]
    assert headings == baseline["heading_order"]


def test_flowable_type_mix_is_unchanged(baseline: dict, story: list) -> None:
    counts: dict[str, int] = {}
    for f in story:
        counts[type(f).__name__] = counts.get(type(f).__name__, 0) + 1
    assert counts == baseline["type_counts"]


def test_rendered_page_count_is_unchanged(baseline: dict, tmp_path: Path) -> None:
    """Renders the real document; layout, not just content, must be stable."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate

    from scripts.project_guide.theme import _page

    document = SimpleDocTemplate(
        str(tmp_path / "guide.pdf"),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
    )
    document.build(build_story(), onFirstPage=_page, onLaterPages=_page)
    assert document.page == baseline["page_count"]


def test_output_file_name_is_unchanged(baseline: dict) -> None:
    """Consumers reference this artifact by name."""
    assert OUTPUT.name == baseline["output_file_name"]


# ─── properties the split layout must keep ───────────────────────────────────


def test_theme_root_points_at_the_repository_root() -> None:
    """The evidence paths resolve against this.

    Regression guard for a real defect in this refactor: `theme.py` sits two
    directories below the root, so carrying the original `parents[1]` over
    resolved every artifact path against `scripts/` and rendered the entire
    document as "not reported" - with no error raised anywhere.
    """
    assert THEME_ROOT == ROOT
    assert (THEME_ROOT / "Data" / "evals").is_dir()


def test_evidence_actually_loaded(story: list) -> None:
    """A silently-empty document would still pass a pure structural check."""
    evidence = Evidence.load()
    assert evidence.rag, "RAG comparison artifact did not load"
    assert evidence.prompt_eval, "prompt evaluation artifact did not load"

    rendered = "\n".join(_describe(f) for f in story)
    not_reported = rendered.count("not reported")
    assert not_reported < 40, (
        f"{not_reported} 'not reported' cells: evidence paths are probably broken"
    )


def test_sections_concatenate_to_the_story(story: list) -> None:
    """Guards the seam the split introduced: a module left out of the tuple.

    Every builder is defined, imported, and lint-clean on its own; only this
    check notices one missing from `SECTION_BUILDERS`.
    """
    evidence = Evidence.load()
    rebuilt: list = []
    for build_section in SECTION_BUILDERS:
        build_section(rebuilt, evidence)
    assert len(rebuilt) == len(story)
    assert [type(f).__name__ for f in rebuilt] == [type(f).__name__ for f in story]


def test_no_section_module_is_empty() -> None:
    evidence = Evidence.load()
    for build_section in SECTION_BUILDERS:
        collected: list = []
        build_section(collected, evidence)
        assert collected, f"{build_section.__module__} contributes no flowables"


def test_clinical_boundary_survives_the_split(lines: list[str]) -> None:
    """The non-clinical disclaimer is the document's load-bearing claim.

    It must not be lost to a section-grouping mistake, so it is asserted by
    content rather than left to the positional comparison.
    """
    rendered = "\n".join(lines)
    assert "Clinical boundary" in rendered
    assert "non-diagnostic engineering prototype" in rendered
    assert "must not be used for diagnosis" in rendered
    assert "Clinical validation" in rendered and "FALSE" in rendered
