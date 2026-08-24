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


class ResearchPaperCorpusInvalid(RuntimeError):
    """The corpus is present but not usable, which is a failure, not an absence."""


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

    missing: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            raise ResearchPaperCorpusInvalid(f"{rendered} contains a non-object entry")
        relative = item.get("path") or item.get("file_name")
        if not relative:
            raise ResearchPaperCorpusInvalid(
                f"{rendered} entry {item.get('pmcid') or '<unknown>'} names no file"
            )
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            candidate = manifest_path.parent / Path(relative).name
        if not candidate.exists():
            missing.append(str(relative))

    if missing:
        raise ResearchPaperCorpusInvalid(
            f"{rendered} references {len(missing)} file(s) that are not present: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
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

    audit = not_evaluated_artifact(
        schema_version="research_paper_kb_audit_v2",
        inspection=inspection,
        paper_count=None,
    )
    evaluation = not_evaluated_artifact(
        schema_version="research_paper_retrieval_eval_v2",
        inspection=inspection,
        paper_count=None,
        case_count=None,
        configurations=None,
    )
    failures = not_evaluated_artifact(
        schema_version="research_paper_retrieval_failures_v1",
        inspection=inspection,
        failure_count=None,
        failures=None,
    )
    write_evidence_json(audit_path, audit)
    write_evidence_json(eval_path, evaluation)
    write_evidence_json(failures_path, failures)
    return {"audit": audit, "evaluation": evaluation, "failures": failures}
