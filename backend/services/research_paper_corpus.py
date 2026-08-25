"""Is the optional research-paper corpus here, missing, or broken?

The 21 PMC papers this repository evaluates against are downloaded by an
explicit maintainer step and deliberately never committed: their licences range
from CC BY to CC BY-NC-ND, and two carry no usable redistribution permission at
all. `KnowledgeBase/raw/research_papers/` is gitignored for that reason, so a
fresh clone cannot reconstruct the corpus and must not try.

The release configuration already says so - all five research-paper artifacts
are `required: false`. What was missing is the runtime half of that policy: the
evaluators ran unconditionally and crashed on the absent manifest, which is
neither an honest "not evaluated" nor a real failure.

Three states, kept apart on purpose:

* **absent** - nothing is there. The evaluation did not happen, and the run says
  so. This is the normal state of any clone that has not fetched the papers.
* **available** - manifest and every file it references are present, and the
  evaluators run exactly as before.
* **invalid** - a manifest that does not parse, references files that are not
  there, or lists nothing at all. This is *not* benign absence: something is
  half-built or corrupted, and reporting it as "optional corpus unavailable"
  would hide a real problem. It fails closed.

The distinction is the whole point. A blanket `except Exception -> not
evaluated` would turn a corrupted corpus into a green run.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT_DIR / "KnowledgeBase/raw/research_papers/research_papers_manifest.json"
CORPUS_DIR = MANIFEST_PATH.parent

AVAILABLE = "available"
ABSENT = "absent"
INVALID = "invalid"

#: Machine-readable state for a current run that did not evaluate the corpus.
#: Deliberately not one of the `accepted_status` values in the release gate: a
#: run that did not happen must not read as an accepted result. Because these
#: artifacts are `required: false`, it surfaces as a warning, not a failure.
NOT_EVALUATED_STATUS = "not_evaluated_optional_corpus"

ABSENT_REASON = (
    "optional restricted/local research-paper corpus unavailable; "
    "full text is licence-restricted and intentionally untracked"
)


#: Statuses the downloader records for a paper it actually obtained. Anything
#: else ("failed", "skipped") means the manifest is describing an acquisition
#: that did not complete.
ACQUIRED_STATUSES = frozenset({"downloaded", "exists"})

#: Suffixes the downloader writes. A manifest entry pointing at anything else is
#: not a paper.
PAPER_SUFFIXES = frozenset({".txt", ".xml"})


class ResearchPaperCorpusInvalid(RuntimeError):
    """The corpus is present but not usable, which is a failure, not an absence."""


def expected_pmcids() -> frozenset[str]:
    """The canonical selection, read from the tracked downloader.

    `scripts/download_research_papers.py` holds the reviewed list of papers as
    a literal, which makes it the single source of truth for what a complete
    corpus contains. Deriving from it means the expected set cannot drift from
    the selection the way a hardcoded count silently would.

    Imported lazily: this module is used on the absent-corpus path, where the
    downloader is never needed.
    """
    from scripts.download_research_papers import PAPERS

    return frozenset(str(paper["pmcid"]) for paper in PAPERS)


@dataclass(frozen=True)
class CorpusInspection:
    state: str
    reason: str
    manifest_path: str
    item_count: int = 0
    missing_files: tuple[str, ...] = field(default=())

    @property
    def available(self) -> bool:
        return self.state == AVAILABLE

    @property
    def absent(self) -> bool:
        return self.state == ABSENT


def _paper_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if p.is_file() and p.name != MANIFEST_PATH.name]


def inspect_research_paper_corpus(
    *,
    manifest_path: Path | str = MANIFEST_PATH,
    root: Path | str = ROOT_DIR,
) -> CorpusInspection:
    """Classify the corpus without reading a single line of article text."""
    manifest_path = Path(manifest_path)
    root = Path(root)
    rendered = manifest_path.as_posix()

    if not manifest_path.exists():
        stranded = _paper_files(manifest_path.parent)
        if stranded:
            # Papers without the manifest that describes them: half-acquired,
            # not absent. Evaluating would silently under-report the corpus.
            raise ResearchPaperCorpusInvalid(
                f"{len(stranded)} research-paper file(s) present but "
                f"{rendered} is missing; the corpus is incomplete"
            )
        return CorpusInspection(state=ABSENT, reason=ABSENT_REASON, manifest_path=rendered)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchPaperCorpusInvalid(f"{rendered} could not be read: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ResearchPaperCorpusInvalid(f"{rendered} is not a manifest object")

    items = manifest.get("items")
    if not isinstance(items, list):
        raise ResearchPaperCorpusInvalid(f"{rendered} has no 'items' list")
    if not items:
        # An empty manifest is a build that produced nothing, not a clone that
        # never fetched anything.
        raise ResearchPaperCorpusInvalid(f"{rendered} lists no papers")

    expected = expected_pmcids()
    seen: dict[str, str] = {}
    problems: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            raise ResearchPaperCorpusInvalid(f"{rendered} contains a non-object entry")

        pmcid = str(item.get("pmcid") or "").strip()
        if not pmcid:
            raise ResearchPaperCorpusInvalid(
                f"{rendered} contains an entry with no PMCID; provenance cannot be "
                "established for it"
            )
        if pmcid in seen:
            raise ResearchPaperCorpusInvalid(
                f"{rendered} lists {pmcid} more than once; the same paper counted "
                "twice would inflate coverage"
            )
        seen[pmcid] = pmcid

        status = str(item.get("status") or "").strip().lower()
        if status not in ACQUIRED_STATUSES:
            problems.append(f"{pmcid}: acquisition status {status!r}")
            continue

        relative = item.get("path") or item.get("file_name")
        if not relative:
            raise ResearchPaperCorpusInvalid(f"{rendered} entry {pmcid} names no file")

        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            candidate = manifest_path.parent / Path(relative).name
        if not candidate.exists():
            problems.append(f"{pmcid}: file missing ({relative})")
            continue
        if candidate.suffix.lower() not in PAPER_SUFFIXES:
            problems.append(f"{pmcid}: {candidate.suffix or 'no suffix'} is not a paper file")
            continue
        try:
            size = candidate.stat().st_size
        except OSError as exc:
            problems.append(f"{pmcid}: unreadable ({exc})")
            continue
        if size <= 0:
            problems.append(f"{pmcid}: file is empty")

    # Cardinality and membership together. A corpus holding one paper of the
    # reviewed twenty-one is not a small corpus, it is a broken one, and
    # evaluating it would report benchmark numbers for a selection nobody chose.
    present = frozenset(seen)
    if present != expected:
        missing_ids = sorted(expected - present)
        unexpected = sorted(present - expected)
        detail = []
        if missing_ids:
            detail.append(f"{len(missing_ids)} of {len(expected)} expected papers absent")
        if unexpected:
            detail.append(f"{len(unexpected)} unrecognised: {unexpected[:3]}")
        raise ResearchPaperCorpusInvalid(
            f"{rendered} does not match the reviewed selection - " + "; ".join(detail)
        )

    if problems:
        raise ResearchPaperCorpusInvalid(
            f"{rendered} describes {len(problems)} unusable paper(s): "
            f"{problems[:5]}{'...' if len(problems) > 5 else ''}"
        )

    return CorpusInspection(
        state=AVAILABLE,
        reason="research-paper corpus present",
        manifest_path=rendered,
        item_count=len(items),
    )


def not_evaluated_artifact(
    *,
    schema_version: str,
    inspection: CorpusInspection,
    **extra: Any,
) -> dict[str, Any]:
    """A current-run artifact that reports no evaluation and no metrics.

    Every numeric field is left out rather than zeroed. A zero recall reads as
    a measured result and would be indistinguishable from a genuinely bad run;
    absent metrics cannot be mistaken for one.

    `stale_evidence_used: false` is explicit because these artifacts overwrite
    tracked files that hold previous results. Writing this payload is what stops
    a consumer reading last month's numbers as this run's.
    """
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": NOT_EVALUATED_STATUS,
        "evaluated": False,
        "required": False,
        "corpus_available": False,
        "stale_evidence_used": False,
        "reason": inspection.reason,
        "manifest_path": inspection.manifest_path,
        # Provenance claims that stay true of a run that did not happen: each
        # asserts something was *not* done.
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "independent_holdout": False,
        "independent_literature_review": False,
        "was_used_for_tuning": False,
        "live_patient_route_changed": False,
        "response_content_retained": False,
        # No metrics were produced. Not zero - none.
        "summary": None,
    }
    payload.update(extra)
    return payload


def evaluation_run_id() -> str:
    """A short id shared by the artifacts one call writes."""
    return "paper-noeval-" + uuid.uuid4().hex[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_evidence_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_evidence_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def paper_chunks_with_verified_provenance(
    chunk_path: Path | str,
    is_paper_chunk,
    *,
    expected: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Load the knowledge-base chunks, refusing to continue on partial provenance.

    A complete raw corpus is not the same as a complete *ingested* one. Every
    reviewed paper can be on disk while the chunk artifact holds only twenty of
    the twenty-one, and the evaluation would then run happily over the twenty:
    coverage reported 0.9524, status came back needs_attention, and the gate
    exited zero. A benchmark measured over a corpus missing a paper is not a
    slightly worse benchmark, it is a measurement of a different corpus.

    Paper-level identity is what matters here, not volume. Any number of chunks
    may carry one PMCID - that is just how a paper was split - so this compares
    the *set* of ingested paper PMCIDs against the reviewed selection and never
    counts chunks.

    Unexpected paper PMCIDs fail too: content from a paper nobody reviewed is a
    provenance problem in the other direction. Curated and other non-paper KB
    chunks are untouched by this - `is_paper_chunk` decides what counts, and
    everything else is simply not this contract's business.
    """
    chunk_path = Path(chunk_path)
    if not chunk_path.exists():
        raise ResearchPaperCorpusInvalid(
            f"the raw corpus is present but {chunk_path.as_posix()} is not; "
            "paper provenance cannot be established"
        )

    chunks = load_evidence_json(chunk_path).get("chunks") or []
    reviewed = {pmcid.upper() for pmcid in (expected if expected is not None else expected_pmcids())}
    ingested = {
        str(row.get("pmcid") or "").upper()
        for row in chunks
        if is_paper_chunk(row) and row.get("pmcid")
    }

    missing = sorted(reviewed - ingested)
    unexpected = sorted(ingested - reviewed)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(
                f"{len(missing)} of {len(reviewed)} reviewed papers have no ingested "
                f"chunks: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            detail.append(f"chunks claim unreviewed papers: {unexpected[:5]}")
        raise ResearchPaperCorpusInvalid(
            "incomplete research-paper provenance - " + "; ".join(detail)
        )
    return chunks


def not_evaluated_reports_if_absent(
    manifest_path: Path | str,
    audit_path: Path,
    eval_path: Path,
    failures_path: Path,
) -> dict[str, Any] | None:
    """Write the three current-run non-results, or return None to evaluate.

    Writing is the point rather than a side effect. These artifacts are tracked
    and hold the previous run's numbers, so leaving them in place would let the
    release gate read month-old retrieval metrics as this run's evidence.

    Returns None when the corpus is present, so the caller proceeds unchanged.
    A corpus that is present but broken raises from the inspection instead.
    """
    inspection = inspect_research_paper_corpus(manifest_path=manifest_path)
    if not inspection.absent:
        return None

    # One id across the three artifacts this call writes, so a reader can tell
    # they describe the same non-run rather than three coincidental ones. It is
    # deliberately local: there is no repository-wide ship-run identity to
    # propagate, and inventing one to span separate Ship steps would be a much
    # larger change than this defect needs.
    run_id = evaluation_run_id()
    audit = not_evaluated_artifact(
        schema_version="research_paper_kb_audit_v2",
        inspection=inspection,
        evaluation_run_id=run_id,
        paper_count=None,
    )
    evaluation = not_evaluated_artifact(
        schema_version="research_paper_retrieval_eval_v2",
        inspection=inspection,
        evaluation_run_id=run_id,
        paper_count=None,
        case_count=None,
        configurations=None,
    )
    failures = not_evaluated_artifact(
        schema_version="research_paper_retrieval_failures_v1",
        inspection=inspection,
        evaluation_run_id=run_id,
        failure_count=None,
        failures=None,
    )
    write_evidence_json(audit_path, audit)
    write_evidence_json(eval_path, evaluation)
    write_evidence_json(failures_path, failures)
    return {"audit": audit, "evaluation": evaluation, "failures": failures}
