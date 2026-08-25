"""A complete raw corpus is not the same as a complete ingested one.

Every reviewed paper can sit on disk while the chunk artifact holds only twenty
of the twenty-one. Before this gate the evaluation ran happily over the twenty:
`manifest_to_chunk_coverage` reported 0.9524, the status came back
`needs_attention`, and the step exited zero. A benchmark measured over a corpus
missing a paper is not a slightly worse benchmark - it is a measurement of a
different corpus, published under the name of the reviewed one.

Coverage was already computed. It was only ever a metric, and a metric cannot
stop a run. It is now a precondition.

Identity is what gets checked, not volume: any number of chunks may carry one
PMCID, because that is just how a paper was split. Non-paper knowledge-base
content is not this contract's business and is never rejected.

No article text appears in these fixtures - the chunks below are authored here
and carry PMCIDs only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.research_paper_corpus import (
    ResearchPaperCorpusInvalid,
    expected_pmcids,
    paper_chunks_with_verified_provenance,
)


def _is_paper_chunk(row: dict) -> bool:
    """Stands in for the evaluator's own owned-source predicate."""
    return bool(row.get("research_paper_owned"))


def _chunk_file(tmp_path: Path, pmcids, *, extra=(), curated=1, name="chunks.json") -> Path:
    rows: list[dict] = [
        {"pmcid": pmcid, "research_paper_owned": True, "section": "body"} for pmcid in pmcids
    ]
    rows += [{"pmcid": pmcid, "research_paper_owned": True} for pmcid in extra]
    rows += [
        {"source": "curated_medical_kb", "text": "unrelated curated guidance"}
        for _ in range(curated)
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"chunks": rows}), encoding="utf-8")
    return path


def _reviewed() -> list[str]:
    return sorted(expected_pmcids())


# --- complete provenance passes ----------------------------------------------


def test_complete_provenance_is_accepted(tmp_path: Path) -> None:
    path = _chunk_file(tmp_path, _reviewed())
    chunks = paper_chunks_with_verified_provenance(path, _is_paper_chunk)
    assert len(chunks) == len(_reviewed()) + 1


def test_many_chunks_per_paper_are_fine(tmp_path: Path) -> None:
    """Papers are split into several chunks; volume is not the contract."""
    reviewed = _reviewed()
    path = _chunk_file(tmp_path, reviewed * 4)
    chunks = paper_chunks_with_verified_provenance(path, _is_paper_chunk)
    assert len(chunks) == len(reviewed) * 4 + 1


def test_unrelated_curated_chunks_are_never_rejected(tmp_path: Path) -> None:
    """A knowledge base is mostly not research papers."""
    path = _chunk_file(tmp_path, _reviewed(), curated=250)
    chunks = paper_chunks_with_verified_provenance(path, _is_paper_chunk)
    assert sum(1 for row in chunks if not _is_paper_chunk(row)) == 250


# --- incomplete provenance fails closed --------------------------------------


def test_one_missing_paper_fails_closed(tmp_path: Path) -> None:
    """The exact case that reported coverage 0.9524 and exited zero."""
    path = _chunk_file(tmp_path, _reviewed()[:-1])
    with pytest.raises(ResearchPaperCorpusInvalid, match="incomplete research-paper provenance"):
        paper_chunks_with_verified_provenance(path, _is_paper_chunk)


def test_several_missing_papers_fail_closed(tmp_path: Path) -> None:
    path = _chunk_file(tmp_path, _reviewed()[:15])
    with pytest.raises(ResearchPaperCorpusInvalid) as excinfo:
        paper_chunks_with_verified_provenance(path, _is_paper_chunk)
    assert "of 21 reviewed papers have no ingested chunks" in str(excinfo.value)


def test_zero_research_paper_chunks_fails_closed(tmp_path: Path) -> None:
    """The raw corpus is there; nothing was ingested from it."""
    path = _chunk_file(tmp_path, [], curated=40)
    with pytest.raises(ResearchPaperCorpusInvalid, match="21 of 21 reviewed papers"):
        paper_chunks_with_verified_provenance(path, _is_paper_chunk)


def test_a_missing_chunk_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ResearchPaperCorpusInvalid, match="paper provenance cannot be established"):
        paper_chunks_with_verified_provenance(tmp_path / "absent.json", _is_paper_chunk)


def test_an_unreviewed_paper_fails_closed(tmp_path: Path) -> None:
    """Provenance breaks in the other direction too.

    Chunks claiming a paper nobody selected mean the benchmark would measure
    retrieval over content outside the reviewed corpus.
    """
    path = _chunk_file(tmp_path, _reviewed(), extra=["PMC0000000"])
    with pytest.raises(ResearchPaperCorpusInvalid, match="unreviewed papers"):
        paper_chunks_with_verified_provenance(path, _is_paper_chunk)


def test_incomplete_provenance_is_never_optional_absence(tmp_path: Path) -> None:
    """It must not be mistaken for the benign not-evaluated path.

    Absence is a clone that never fetched the corpus. This is a corpus that was
    fetched and then half-ingested, which is a failure and has to exit non-zero.
    """
    path = _chunk_file(tmp_path, _reviewed()[:-1])
    with pytest.raises(ResearchPaperCorpusInvalid) as excinfo:
        paper_chunks_with_verified_provenance(path, _is_paper_chunk)

    message = str(excinfo.value).lower()
    assert "not_evaluated" not in message
    assert "optional" not in message


# --- the evaluator honours it ------------------------------------------------


def test_the_evaluator_refuses_to_score_a_half_ingested_corpus(tmp_path: Path, monkeypatch) -> None:
    """End to end: the gate sits in front of the real evaluation.

    Uses the repository's own owned-source predicate rather than the stand-in,
    so this fails if the evaluator ever stops routing through the check.
    """
    from backend.services import research_paper_kb_eval as evaluator

    reviewed = _reviewed()
    rows = [
        {
            "pmcid": pmcid,
            "source": "research_paper",
            "source_type": "research_paper",
            "section": "body",
        }
        for pmcid in reviewed[:-1]
    ]
    chunk_path = tmp_path / "chunks.json"
    chunk_path.write_text(json.dumps({"chunks": rows}), encoding="utf-8")

    monkeypatch.setattr(evaluator, "_is_owned_research_source", lambda row: "pmcid" in row)

    with pytest.raises(ResearchPaperCorpusInvalid):
        evaluator.paper_chunks_with_verified_provenance(
            chunk_path, evaluator._is_owned_research_source
        )


def test_total_corpus_absence_is_still_the_benign_path(tmp_path: Path) -> None:
    """The provenance gate must not swallow the optional-absence contract.

    A checkout that never fetched the papers still reports not-evaluated and
    exits zero; only a *present* corpus is held to provenance.
    """
    from backend.services.research_paper_corpus import NOT_EVALUATED_STATUS
    from backend.services.research_paper_kb_eval import run_research_paper_kb_eval

    reports = run_research_paper_kb_eval(
        manifest_path=tmp_path / "research_papers" / "research_papers_manifest.json",
        eval_path=tmp_path / "eval.json",
        audit_path=tmp_path / "audit.json",
        failures_path=tmp_path / "failures.json",
    )
    assert reports["evaluation"]["status"] == NOT_EVALUATED_STATUS
    assert reports["evaluation"]["evaluated"] is False


# --- telemetry corpus-dependency, alongside the provenance change -------------


def test_mixed_suite_can_declare_its_corpus_dependency(monkeypatch, tmp_path: Path) -> None:
    """A suite mixing canonical and custom cases is not auto-gated.

    Content-based detection deliberately does not guess for mixed suites, so the
    explicit flag is how such a caller states the dependency either way.
    """
    from backend.services import research_paper_query_telemetry as telemetry
    from backend.services.research_paper_corpus import (
        ABSENT,
        NOT_EVALUATED_STATUS,
        CorpusInspection,
    )

    mixed = [
        dict(telemetry.QUERY_CASES[0]),
        {
            "id": "custom-probe",
            "category": "fixture",
            "style": "formal",
            "query": "unrelated",
            "allowed_intents": ["education"],
        },
    ]
    assert telemetry.suite_requires_research_paper_corpus(mixed) is False

    monkeypatch.setattr(
        telemetry,
        "inspect_research_paper_corpus",
        lambda **_: CorpusInspection(state=ABSENT, reason="absent", manifest_path="none"),
    )
    report = telemetry.run_research_paper_query_telemetry(
        output_path=tmp_path / "t.json",
        failures_path=tmp_path / "f.json",
        cases=mixed,
        requires_research_paper_corpus=True,
    )
    assert report["status"] == NOT_EVALUATED_STATUS
