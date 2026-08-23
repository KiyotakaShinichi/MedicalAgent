"""KB source and chunk identifiers must not depend on the host OS.

The defect
----------
`_source_metadata` seeded the source id with `str(path)`, and `_chunk_id` did
the same. `str(Path(...))` renders the separator of whatever OS is running, so
one source file produced two different ids:

    Windows  KnowledgeBase\\raw\\...\\minimum_evidence...md -> 28cfcee61ce1e4a4
    Linux    KnowledgeBase/raw/.../minimum_evidence...md    -> 191dafae170c06c0

That id is the key the KB source-governance map is looked up by. On Linux every
ingested chunk therefore resolved to no governance entry, reached the
pre-generation tier filter with no tier and no `allowed_use`, and was correctly
dropped — taking `retrieval_context`, citations, and the agent regression pass
rate with it. Only the hardcoded seed snippets, whose ids are static, survived.

What these tests lock in
------------------------
* the same logical source yields the same ids however the path is spelled;
* the ids still match the values twenty-one committed evidence artifacts
  already encode, so the fix does not silently re-issue identifiers;
* a source with no governance entry is still ungoverned — the fix restores
  metadata *discovery*, it must never turn missing metadata into accepted
  metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.kb_ingestion import (  # noqa: E402
    _canonical_source_key,
    _chunk_id,
    _source_metadata,
)

# A tracked curated source. Its Windows-derived id is the one the committed
# governance artifact and the live tier filter already key on.
_RELATIVE = (
    "KnowledgeBase/raw/curated_medical_kb/00_safety_boundaries/"
    "minimum_evidence_medical_claim_boundaries.md"
)
_ESTABLISHED_SOURCE_ID = "28cfcee61ce1e4a4"

# The same logical path, spelled every way ingestion can encounter it.
_SPELLINGS = [
    pytest.param(PurePosixPath(_RELATIVE), id="posix-separators"),
    pytest.param(PureWindowsPath(_RELATIVE), id="windows-separators"),
    pytest.param(Path(_RELATIVE), id="native-path"),
    pytest.param(_RELATIVE, id="plain-string"),
]


@pytest.mark.parametrize("path", _SPELLINGS)
def test_canonical_key_is_identical_for_every_spelling(path) -> None:
    assert _canonical_source_key(path) == _canonical_source_key(PureWindowsPath(_RELATIVE))


@pytest.mark.parametrize("path", _SPELLINGS)
def test_source_id_is_identical_for_every_spelling(path) -> None:
    """This is the id the governance map is looked up by."""
    metadata = _source_metadata(Path(str(path)), "body text", {})
    assert metadata["source_id"] == _ESTABLISHED_SOURCE_ID


@pytest.mark.parametrize("path", _SPELLINGS)
def test_chunk_id_is_identical_for_every_spelling(path) -> None:
    reference = _chunk_id(PureWindowsPath(_RELATIVE), 0, "chunk body")
    assert _chunk_id(path, 0, "chunk body") == reference


def test_source_id_still_matches_committed_evidence() -> None:
    """Guards against silently re-issuing every identifier.

    Twenty-one tracked artifacts, including a frozen claim-selector holdout
    bank, key on these ids. A change here invalidates all of them, so it must
    be a deliberate decision rather than a side effect.
    """
    metadata = _source_metadata(Path(_RELATIVE), "body text", {})
    assert metadata["source_id"] == _ESTABLISHED_SOURCE_ID


def test_distinct_sources_keep_distinct_ids() -> None:
    """Normalisation must not collapse different files onto one id."""
    a = _source_metadata(Path("KnowledgeBase/raw/curated_medical_kb/a.md"), "x", {})
    b = _source_metadata(Path("KnowledgeBase/raw/curated_medical_kb/b.md"), "x", {})
    assert a["source_id"] != b["source_id"]

    nested = _source_metadata(Path("KnowledgeBase/raw/other/a.md"), "x", {})
    assert nested["source_id"] != a["source_id"], "directory must still distinguish sources"


# ─── fail-closed: discovery restored, policy unchanged ───────────────────────


def test_unmanifested_source_receives_no_governance_grant() -> None:
    """The fix restores lookup; it must not invent permissions.

    A source with no manifest entry must come back with an empty `allowed_use`
    so the tier filter still rejects it. If this ever returns a populated list,
    missing metadata has become accepted metadata.
    """
    metadata = _source_metadata(Path("KnowledgeBase/raw/unknown/mystery.md"), "text", {})
    assert metadata["allowed_use"] == []
    assert metadata["evidence_role"] is None
    assert metadata["patient_facing_suitability"] is None
    assert metadata["not_allowed_for"] == []


def test_manifest_entry_is_still_honoured_when_present() -> None:
    """Discovery works in the positive direction too, or the fix is pointless."""
    manifest = {
        "mystery.md": {
            "allowed_use": ["education"],
            "evidence_role": "supporting",
            "patient_facing_suitability": "suitable",
        }
    }
    metadata = _source_metadata(Path("KnowledgeBase/raw/unknown/mystery.md"), "text", manifest)
    assert metadata["allowed_use"] == ["education"]
    assert metadata["evidence_role"] == "supporting"
