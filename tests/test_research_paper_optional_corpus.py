"""The optional research-paper corpus: absent, present, or broken.

The 21 PMC papers are licence-restricted and intentionally untracked, so most
checkouts do not have them. The release configuration already marks all five
research-paper artifacts `required: false`; these tests pin the runtime half of
that policy, which is where it was missing.

Three things are easy to get wrong here and each has a test:

* calling an absent corpus a PASS, which would claim an evaluation happened;
* leaving the previous run's artifacts in place, which is worse, because those
  files are tracked and hold real numbers that the release gate would read as
  this run's evidence;
* collapsing a half-built corpus into the same "unavailable" branch, which
  would turn a corrupted state into a green run.

No article text appears in any fixture. The manifests below are authored here
and reference files these tests create.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.research_paper_corpus import (
    ABSENT,
    AVAILABLE,
    NOT_EVALUATED_STATUS,
    CorpusInspection,
    ResearchPaperCorpusInvalid,
    inspect_research_paper_corpus,
    not_evaluated_artifact,
)
from backend.services.research_paper_kb_eval import run_research_paper_kb_eval

RELEASE_CONFIG = Path(__file__).resolve().parents[1] / "config/release_gate_thresholds.yaml"

RESEARCH_PAPER_ARTIFACTS = (
    "Data/evals/rag/latest_research_paper_kb_audit.json",
    "Data/evals/rag/latest_research_paper_retrieval_eval.json",
    "Data/evals/rag/latest_research_paper_retrieval_failures.json",
    "Data/evals/rag/latest_research_paper_query_telemetry.json",
    "Data/evals/rag/latest_research_paper_query_telemetry_failures.json",
)


def _manifest(tmp_path: Path, items: list[dict]) -> Path:
    directory = tmp_path / "research_papers"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "research_papers_manifest.json"
    path.write_text(
        json.dumps({"schema_version": "test_manifest_v1", "items": items}),
        encoding="utf-8",
    )
    return path


def _paper(directory: Path, name: str) -> str:
    """A stand-in file. Never article text - only a placeholder byte string."""
    target = directory / name
    target.write_text("test placeholder, not article text", encoding="utf-8")
    return name


# --- classification -----------------------------------------------------------


def test_absent_corpus_is_absent(tmp_path: Path) -> None:
    inspection = inspect_research_paper_corpus(
        manifest_path=tmp_path / "research_papers" / "research_papers_manifest.json",
        root=tmp_path,
    )
    assert inspection.state == ABSENT
    assert inspection.absent is True


def test_complete_corpus_is_available(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [])
    name = _paper(manifest.parent, "PMC0000001_placeholder.txt")
    manifest.write_text(
        json.dumps({"items": [{"pmcid": "PMC0000001", "path": name}]}),
        encoding="utf-8",
    )

    inspection = inspect_research_paper_corpus(manifest_path=manifest, root=manifest.parent)
    assert inspection.state == AVAILABLE
    assert inspection.item_count == 1


def test_manifest_referencing_a_missing_file_is_invalid(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [{"pmcid": "PMC1", "path": "not_here.txt"}])
    with pytest.raises(ResearchPaperCorpusInvalid, match="not present"):
        inspect_research_paper_corpus(manifest_path=manifest, root=tmp_path)


def test_unparseable_manifest_is_invalid(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [])
    manifest.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ResearchPaperCorpusInvalid, match="could not be read"):
        inspect_research_paper_corpus(manifest_path=manifest, root=tmp_path)


def test_empty_manifest_is_invalid(tmp_path: Path) -> None:
    """A manifest listing nothing is a build that produced nothing."""
    manifest = _manifest(tmp_path, [])
    with pytest.raises(ResearchPaperCorpusInvalid, match="lists no papers"):
        inspect_research_paper_corpus(manifest_path=manifest, root=tmp_path)


def test_papers_without_a_manifest_are_invalid(tmp_path: Path) -> None:
    """Half-acquired, not absent: evaluating would under-report the corpus."""
    directory = tmp_path / "research_papers"
    directory.mkdir(parents=True)
    _paper(directory, "PMC0000002_placeholder.txt")

    with pytest.raises(ResearchPaperCorpusInvalid, match="incomplete"):
        inspect_research_paper_corpus(
            manifest_path=directory / "research_papers_manifest.json",
            root=tmp_path,
        )


def test_a_broken_corpus_is_never_reported_as_absent(tmp_path: Path) -> None:
    """The distinction this whole module exists to keep.

    If any invalid state ever degraded into ABSENT, a corrupted corpus would
    produce a clean "not evaluated" run and nobody would notice.
    """
    manifest = _manifest(tmp_path, [{"pmcid": "PMC1", "path": "gone.txt"}])
    with pytest.raises(ResearchPaperCorpusInvalid):
        inspect_research_paper_corpus(manifest_path=manifest, root=tmp_path)


# --- the artifact a non-run produces -------------------------------------------


def _absent_inspection() -> CorpusInspection:
    return inspect_research_paper_corpus(
        manifest_path=Path("does") / "not" / "exist.json",
        root=Path("does"),
    )


def test_the_non_result_is_not_a_pass() -> None:
    payload = not_evaluated_artifact(
        schema_version="research_paper_retrieval_eval_v2",
        inspection=_absent_inspection(),
    )
    assert payload["status"] == NOT_EVALUATED_STATUS
    assert payload["evaluated"] is False
    assert payload["required"] is False
    assert payload["corpus_available"] is False
    assert payload["stale_evidence_used"] is False
    assert payload["reason"]

    for accepted in ("acceptable", "acceptable_internal_diagnostic", "strong", "passed"):
        assert payload["status"] != accepted


def test_the_non_result_carries_no_metrics() -> None:
    """Absent, not zero. A zero recall reads as a measured result."""
    payload = not_evaluated_artifact(
        schema_version="research_paper_retrieval_eval_v2",
        inspection=_absent_inspection(),
        paper_count=None,
        case_count=None,
    )
    assert payload["summary"] is None
    assert payload["paper_count"] is None
    assert payload["case_count"] is None

    for key, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            pytest.fail(f"{key}={value!r} is a fabricated metric on a non-run")


def test_the_non_result_keeps_its_provenance_negatives() -> None:
    """"We did not validate this clinically" stays true of a run that never ran."""
    payload = not_evaluated_artifact(
        schema_version="research_paper_kb_audit_v2",
        inspection=_absent_inspection(),
    )
    assert payload["clinical_validation"] is False
    assert payload["healthcare_production_ready"] is False
    assert payload["independent_holdout"] is False
    assert payload["was_used_for_tuning"] is False
    assert payload["live_patient_route_changed"] is False


# --- end to end, through the evaluator -----------------------------------------


def test_absent_corpus_writes_a_non_result_over_previous_numbers(tmp_path: Path) -> None:
    """The stale-evidence guard, which is the reason these files are rewritten.

    The real artifacts are tracked and hold the last successful run's metrics.
    Leaving them untouched would let the release gate read those as current.
    """
    stale = {
        "schema_version": "research_paper_retrieval_eval_v2",
        "status": "acceptable_internal_diagnostic",
        "generated_at": "2026-08-12T06:42:19.007412+00:00",
        "summary": {"full_stack_recall_at_10": 1.0},
    }
    eval_path = tmp_path / "eval.json"
    audit_path = tmp_path / "audit.json"
    failures_path = tmp_path / "failures.json"
    for path in (eval_path, audit_path, failures_path):
        path.write_text(json.dumps(stale), encoding="utf-8")

    reports = run_research_paper_kb_eval(
        manifest_path=tmp_path / "research_papers" / "research_papers_manifest.json",
        eval_path=eval_path,
        audit_path=audit_path,
        failures_path=failures_path,
    )

    assert reports["evaluation"]["evaluated"] is False
    for path in (eval_path, audit_path, failures_path):
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["status"] == NOT_EVALUATED_STATUS
        assert written["summary"] is None
        assert written["stale_evidence_used"] is False
        assert "full_stack_recall_at_10" not in json.dumps(written)


def test_absent_corpus_never_touches_the_network(tmp_path: Path, monkeypatch) -> None:
    """A missing corpus must not become a download."""
    import socket

    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("the absent-corpus path attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    reports = run_research_paper_kb_eval(
        manifest_path=tmp_path / "research_papers" / "research_papers_manifest.json",
        eval_path=tmp_path / "eval.json",
        audit_path=tmp_path / "audit.json",
        failures_path=tmp_path / "failures.json",
    )
    assert reports["evaluation"]["status"] == NOT_EVALUATED_STATUS


def test_a_broken_corpus_fails_the_step_rather_than_skipping_it(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [{"pmcid": "PMC1", "path": "vanished.txt"}])
    with pytest.raises(ResearchPaperCorpusInvalid):
        run_research_paper_kb_eval(
            manifest_path=manifest,
            eval_path=tmp_path / "eval.json",
            audit_path=tmp_path / "audit.json",
            failures_path=tmp_path / "failures.json",
        )


# --- the policy itself was not weakened ----------------------------------------


def test_research_paper_artifacts_are_still_optional_and_unchanged() -> None:
    """`required: false` is the existing policy; this work honours it, not edits it."""
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(RELEASE_CONFIG.read_text(encoding="utf-8"))

    found: dict[str, dict] = {}

    def walk(node):
        if isinstance(node, dict):
            path = node.get("path")
            if isinstance(path, str) and "research_paper" in path:
                found[path] = node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(config)

    for path in RESEARCH_PAPER_ARTIFACTS:
        assert path in found, f"{path} is no longer in the release configuration"
        assert found[path]["required"] is False, f"{path} was made required"


def test_the_non_result_status_is_not_silently_accepted() -> None:
    """It must not appear in any `accepted_status`.

    If it did, a run that never happened would read as an accepted result, and
    the honest artifact would become a way of passing the gate without evidence.
    """
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(RELEASE_CONFIG.read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            accepted = node.get("accepted_status")
            if isinstance(accepted, list):
                assert NOT_EVALUATED_STATUS not in accepted, (
                    f"{node.get('path')} accepts {NOT_EVALUATED_STATUS} as a result"
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(config)


def test_a_caller_supplying_its_own_cases_is_not_corpus_gated(monkeypatch, tmp_path: Path) -> None:
    """Only the default probe depends on the corpus.

    `QUERY_CASES` is the fixed suite that asks about the papers, so it is gated.
    A caller passing its own cases is exercising the runner itself and has no
    such dependency; short-circuiting it would answer a question it did not ask.
    This was a real regression: gating every call broke a runner test that stubs
    the pipeline outright, and it only showed up on a checkout without the
    corpus.
    """
    import backend.services.agent_rag as agent_rag
    from backend.services.research_paper_query_telemetry import (
        run_research_paper_query_telemetry,
    )

    monkeypatch.setattr(
        agent_rag,
        "run_patient_agent_pipeline",
        lambda **_: {"reply": "x", "citations": [], "sources": [], "intent": "education"},
    )
    monkeypatch.setattr(
        "backend.services.research_paper_query_telemetry.inspect_research_paper_corpus",
        lambda **_: CorpusInspection(state=ABSENT, reason="absent", manifest_path="none"),
    )

    report = run_research_paper_query_telemetry(
        output_path=tmp_path / "telemetry.json",
        failures_path=tmp_path / "failures.json",
        cases=[
            {
                "id": "fixture",
                "category": "fixture",
                "style": "formal",
                "query": "fixture query",
                "allowed_intents": ["education"],
            }
        ],
    )

    assert report.get("evaluated") is not False, (
        "an explicit-cases call was gated on the corpus it does not use"
    )
    assert report["query_count"] == 1


def test_the_default_probe_is_still_corpus_gated(monkeypatch, tmp_path: Path) -> None:
    """The other half of the same distinction."""
    from backend.services import research_paper_query_telemetry as telemetry

    monkeypatch.setattr(
        telemetry,
        "inspect_research_paper_corpus",
        lambda **_: CorpusInspection(state=ABSENT, reason="absent", manifest_path="none"),
    )

    report = telemetry.run_research_paper_query_telemetry(
        output_path=tmp_path / "telemetry.json",
        failures_path=tmp_path / "failures.json",
    )
    assert report["status"] == NOT_EVALUATED_STATUS
    assert report["evaluated"] is False
