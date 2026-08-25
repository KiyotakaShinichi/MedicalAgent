"""Only complete absence is benign; everything in between must fail closed.

The first version of the corpus inspector proved that the files a manifest
listed existed, and nothing more. That accepted a corpus holding one paper of
the reviewed twenty-one, duplicate PMCIDs, entries with no PMCID at all, and
zero-byte articles - each of which would have produced benchmark numbers for a
selection nobody chose, reported as though the real suite had run.

Two other defects are pinned here. The telemetry probe decided whether it
depended on the corpus by comparing `cases is QUERY_CASES`, so `list(QUERY_CASES)`
or a deepcopy - the canonical questions, asked of a knowledge base holding no
papers - slipped past the preflight. And downstream consumers still presented
recorded historical metrics in current-sounding language.

The expected selection is read from `scripts/download_research_papers.py`, which
holds the reviewed list as a literal. Nothing here hardcodes a count, and no
article text appears in any fixture: the placeholder files below are written by
the tests themselves.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.services.research_paper_corpus import (
    ABSENT,
    AVAILABLE,
    NOT_EVALUATED_STATUS,
    CorpusInspection,
    ResearchPaperCorpusInvalid,
    expected_pmcids,
    inspect_research_paper_corpus,
)
from backend.services.research_paper_kb_eval import run_research_paper_kb_eval
from backend.services.research_paper_query_telemetry import (
    QUERY_CASES,
    suite_requires_research_paper_corpus,
)


def _canonical_manifest(tmp_path: Path, *, pmcids=None, status="downloaded", body="paper text"):
    """A manifest describing the full reviewed selection, with real files."""
    directory = tmp_path / "research_papers"
    directory.mkdir(parents=True, exist_ok=True)
    ids = sorted(expected_pmcids()) if pmcids is None else list(pmcids)
    items = []
    for pmcid in ids:
        name = f"{pmcid}_placeholder.txt"
        (directory / name).write_text(body, encoding="utf-8")
        items.append({"pmcid": pmcid, "path": name, "status": status, "bytes": len(body)})
    manifest = directory / "research_papers_manifest.json"
    manifest.write_text(json.dumps({"items": items}), encoding="utf-8")
    return manifest


def _rewrite(manifest: Path, mutate) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")


# --- the source of truth ------------------------------------------------------


def test_the_expected_selection_comes_from_the_tracked_downloader() -> None:
    """Derived, not hardcoded, so the contract cannot drift from the selection."""
    from scripts.download_research_papers import PAPERS

    assert expected_pmcids() == frozenset(str(paper["pmcid"]) for paper in PAPERS)
    assert len(expected_pmcids()) == len(PAPERS), "the reviewed selection has duplicates"


# --- corpus classification ----------------------------------------------------


def test_the_complete_reviewed_selection_is_valid(tmp_path: Path) -> None:
    manifest = _canonical_manifest(tmp_path)
    inspection = inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)

    assert inspection.state == AVAILABLE
    assert inspection.item_count == len(expected_pmcids())


def test_a_single_paper_corpus_is_invalid(tmp_path: Path) -> None:
    """One of twenty-one is a broken corpus, not a small one.

    This is the case that motivated the hardening: it previously validated and
    would have reported paper-retrieval metrics measured against a corpus that
    was never the reviewed selection.
    """
    manifest = _canonical_manifest(tmp_path, pmcids=sorted(expected_pmcids())[:1])
    with pytest.raises(ResearchPaperCorpusInvalid, match="reviewed selection"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_an_extra_unrecognised_paper_is_invalid(tmp_path: Path) -> None:
    manifest = _canonical_manifest(tmp_path, pmcids=sorted(expected_pmcids()) + ["PMC0000000"])
    with pytest.raises(ResearchPaperCorpusInvalid, match="reviewed selection"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_an_entry_without_a_pmcid_is_invalid(tmp_path: Path) -> None:
    manifest = _canonical_manifest(tmp_path)
    _rewrite(manifest, lambda p: p["items"][0].update({"pmcid": ""}))

    with pytest.raises(ResearchPaperCorpusInvalid, match="no PMCID"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_a_duplicate_pmcid_is_invalid(tmp_path: Path) -> None:
    """The same paper counted twice would inflate coverage."""
    manifest = _canonical_manifest(tmp_path)
    _rewrite(manifest, lambda p: p["items"][1].update({"pmcid": p["items"][0]["pmcid"]}))

    with pytest.raises(ResearchPaperCorpusInvalid, match="more than once"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_a_zero_byte_article_is_invalid(tmp_path: Path) -> None:
    manifest = _canonical_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    (manifest.parent / payload["items"][0]["path"]).write_text("", encoding="utf-8")

    with pytest.raises(ResearchPaperCorpusInvalid, match="unusable"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_a_non_paper_file_is_invalid(tmp_path: Path) -> None:
    """Only the file types the downloader writes count as papers."""
    manifest = _canonical_manifest(tmp_path)
    (manifest.parent / "notes.md").write_text("not a paper", encoding="utf-8")
    _rewrite(manifest, lambda p: p["items"][0].update({"path": "notes.md"}))

    with pytest.raises(ResearchPaperCorpusInvalid, match="unusable"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_a_referenced_file_that_is_absent_is_invalid(tmp_path: Path) -> None:
    manifest = _canonical_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    (manifest.parent / payload["items"][0]["path"]).unlink()

    with pytest.raises(ResearchPaperCorpusInvalid, match="unusable"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


@pytest.mark.parametrize("status", ["failed", "skipped", "", "pending"])
def test_an_unacquired_status_is_invalid(tmp_path: Path, status: str) -> None:
    """`failed` and `skipped` describe an acquisition that never completed."""
    manifest = _canonical_manifest(tmp_path, status=status)
    with pytest.raises(ResearchPaperCorpusInvalid):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_a_malformed_manifest_is_invalid(tmp_path: Path) -> None:
    manifest = _canonical_manifest(tmp_path)
    manifest.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ResearchPaperCorpusInvalid, match="could not be read"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_an_empty_manifest_is_invalid(tmp_path: Path) -> None:
    manifest = _canonical_manifest(tmp_path)
    _rewrite(manifest, lambda p: p.update({"items": []}))

    with pytest.raises(ResearchPaperCorpusInvalid, match="lists no papers"):
        inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)


def test_complete_absence_remains_the_only_benign_case(tmp_path: Path) -> None:
    """The whole point of the hardening, stated once.

    Nothing partial is optional-absent. Only a checkout that never fetched the
    corpus gets the not-evaluated path.
    """
    inspection = inspect_research_paper_corpus(
        manifest_path=tmp_path / "research_papers" / "research_papers_manifest.json",
        root=tmp_path,
    )
    assert inspection.state == ABSENT


# --- telemetry: a semantic contract, not object identity ----------------------


def test_canonical_questions_require_the_corpus_however_they_were_built() -> None:
    """The defect: an identity check let a copy of the suite bypass preflight."""
    assert suite_requires_research_paper_corpus([dict(c) for c in QUERY_CASES]) is True
    assert suite_requires_research_paper_corpus(list(QUERY_CASES)) is True
    assert suite_requires_research_paper_corpus(copy.deepcopy(list(QUERY_CASES))) is True
    assert suite_requires_research_paper_corpus([dict(QUERY_CASES[0])]) is True, (
        "one canonical question still asks about the papers"
    )


def test_a_custom_suite_does_not_require_the_corpus() -> None:
    assert suite_requires_research_paper_corpus([{"id": "fixture", "query": "q"}]) is False
    assert suite_requires_research_paper_corpus([]) is False


def test_the_requirement_can_be_stated_outright(monkeypatch, tmp_path: Path) -> None:
    """The explicit contract that replaces inferring from object identity."""
    from backend.services import research_paper_query_telemetry as telemetry

    monkeypatch.setattr(
        telemetry,
        "inspect_research_paper_corpus",
        lambda **_: CorpusInspection(state=ABSENT, reason="absent", manifest_path="none"),
    )

    report = telemetry.run_research_paper_query_telemetry(
        output_path=tmp_path / "telemetry.json",
        failures_path=tmp_path / "failures.json",
        cases=[{
            "id": "fixture",
            "category": "fixture",
            "style": "formal",
            "query": "fixture query",
            "allowed_intents": ["education"],
        }],
        requires_research_paper_corpus=True,
    )
    assert report["status"] == NOT_EVALUATED_STATUS


def test_a_canonical_copy_cannot_bypass_the_preflight(monkeypatch, tmp_path: Path) -> None:
    """End to end, through the runner, with the corpus reported absent."""
    from backend.services import research_paper_query_telemetry as telemetry

    monkeypatch.setattr(
        telemetry,
        "inspect_research_paper_corpus",
        lambda **_: CorpusInspection(state=ABSENT, reason="absent", manifest_path="none"),
    )

    report = telemetry.run_research_paper_query_telemetry(
        output_path=tmp_path / "telemetry.json",
        failures_path=tmp_path / "failures.json",
        cases=copy.deepcopy(list(QUERY_CASES)),
    )
    assert report["evaluated"] is False
    assert report["status"] == NOT_EVALUATED_STATUS


# --- current-run identity and downstream currency -----------------------------


def test_the_non_results_of_one_call_share_a_run_id(tmp_path: Path) -> None:
    reports = run_research_paper_kb_eval(
        manifest_path=tmp_path / "research_papers" / "research_papers_manifest.json",
        eval_path=tmp_path / "eval.json",
        audit_path=tmp_path / "audit.json",
        failures_path=tmp_path / "failures.json",
    )
    ids = {report["evaluation_run_id"] for report in reports.values()}
    assert len(ids) == 1, f"artifacts from one call carry different ids: {ids}"


def test_recorded_negative_results_are_never_presented_as_current() -> None:
    """The 9-paper / 32-case numbers describe a corpus that no longer exists.

    They are kept verbatim on purpose - a negative result must not be quietly
    revised - but they are labelled so nobody reads them as this run's.
    """
    from backend.services.governance_artifacts.negative_results import (
        build_negative_results_gallery,
    )

    gallery = build_negative_results_gallery()
    assert gallery["items"], "the gallery is empty"
    for item in gallery["items"]:
        assert item["evidence_currency"] == "historical"
        assert item["metric_value_is_current_run"] is False


def test_evidence_maturity_quotes_no_case_count_when_nothing_ran() -> None:
    from backend.services.evidence_maturity_matrix import _research_paper_sentence

    absent = _research_paper_sentence({"status": NOT_EVALUATED_STATUS, "evaluated": False})
    assert "not evaluated in this run" in absent
    assert "-case" not in absent, "a case count reads as a measurement that did not happen"

    present = _research_paper_sentence({"status": "needs_attention", "case_count": 44})
    assert "44-case" in present
