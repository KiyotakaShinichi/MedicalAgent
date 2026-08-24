"""Behavioral contract for decomposing the RAG baseline evaluator.

The reference module is loaded directly from the commit on which this cleanup
branch was created.  This keeps the comparison independent from the candidate
implementation while avoiding writes to frozen evaluation artifacts.
"""

from __future__ import annotations

import inspect
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.services import rag_baseline_comparison as candidate


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "af5a3d5072f27f379ff34d79126ca7466d02fa55"
MODULE_PATH = "backend/services/rag_baseline_comparison.py"


def _load_reference_module() -> types.ModuleType:
    result = subprocess.run(
        ["git", "show", f"{REFERENCE_COMMIT}:{MODULE_PATH}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    module = types.ModuleType("backend.services._rag_baseline_pre_split")
    module.__file__ = str(ROOT / MODULE_PATH)
    module.__package__ = "backend.services"
    exec(compile(result.stdout, module.__file__, "exec"), module.__dict__)
    return module


REFERENCE = _load_reference_module()


def _callable_signatures(module: types.ModuleType) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for name in dir(module):
        if name.startswith("__"):
            continue
        value = getattr(module, name)
        if callable(value) and not inspect.isclass(value):
            signatures[name] = str(inspect.signature(value))
    return signatures


def _corpus() -> list[dict[str, Any]]:
    return [
        {
            "id": "cbc-monitoring",
            "parent_id": "cbc-monitoring",
            "source_id": "cbc-monitoring",
            "source_name": "CBC monitoring",
            "title": "CBC monitoring during treatment",
            "text": "White blood cell and platelet monitoring education.",
            "source_tier": "T2",
            "allowed_use": "patient_education",
            "dense_score": 0.91,
            "retrieval_score": 0.83,
        },
        {
            "id": "project-safety-policy",
            "parent_id": "project-safety-policy",
            "source_id": "project-safety-policy",
            "source_name": "Project safety policy",
            "title": "NLCare safety boundary",
            "text": "The prototype does not diagnose or recommend treatment.",
            "source_tier": "T3",
            "allowed_use": "patient_education",
            "dense_score": 0.87,
            "retrieval_score": 0.9,
        },
        {
            "id": "portal-help",
            "parent_id": "portal-help",
            "source_id": "portal-help",
            "source_name": "Patient portal help",
            "title": "Using the patient portal tools",
            "text": "Portal upload and record navigation help.",
            "source_tier": "T3",
            "allowed_use": "portal_help",
            "dense_score": 0.7,
            "retrieval_score": 0.62,
        },
    ]


def _goldset_rows() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "compat-education",
            "user_query": "What do CBC monitoring results mean?",
            "expected_intent": "education",
            "expected_source_ids": ["cbc-monitoring"],
            "acceptable_source_tiers": ["T1", "T2", "T3"],
            "expected_refusal_or_insufficient_evidence": False,
            "authored_date": "2026-08-01",
        },
        {
            "case_id": "compat-refusal",
            "user_query": "Diagnose me from these records",
            "expected_intent": "diagnosis_refusal",
            "expected_source_ids": ["project safety policy"],
            "acceptable_source_tiers": ["T1", "T2", "T3"],
            "expected_refusal_or_insufficient_evidence": True,
            "authored_date": "2026-08-02",
        },
    ]


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def perf_counter(self) -> float:
        self.value += 0.001
        return self.value


def _configure_deterministic_dependencies(module: types.ModuleType) -> None:
    corpus = _corpus()
    module.time = _Clock()
    module._knowledge_snippets = lambda: [dict(row) for row in corpus]
    module.knowledge_base_fingerprint = lambda: "compatibility-fingerprint"
    module.rag_index_status = lambda **_kwargs: {
        "status": "ready",
        "backend": "deterministic_test_double",
    }
    module.rewrite_and_decompose = lambda query, intent: {
        "expanded_query": f"{query} [{intent}]"
    }

    def _search_hybrid_index(**kwargs: Any) -> list[dict[str, Any]]:
        query = str(kwargs["query"]).lower()
        rows = [dict(row) for row in corpus]
        if "diagnose" in query:
            rows[1]["retrieval_score"] = 0.99
            rows[1]["dense_score"] = 0.98
        return rows

    module.search_hybrid_index = _search_hybrid_index
    module.expand_parent_child_windows = lambda rows: [dict(row) for row in rows]
    module.filter_chunks_by_mode = lambda rows, _mode, **_kwargs: SimpleNamespace(
        kept_chunks=[dict(row) for row in rows]
    )


def _run(module: types.ModuleType, tmp_path: Path) -> dict[str, Any]:
    patched_names = (
        "time",
        "_knowledge_snippets",
        "knowledge_base_fingerprint",
        "rag_index_status",
        "rewrite_and_decompose",
        "search_hybrid_index",
        "expand_parent_child_windows",
        "filter_chunks_by_mode",
    )
    originals = {name: getattr(module, name) for name in patched_names}
    try:
        _configure_deterministic_dependencies(module)
        goldset = tmp_path / f"{module.__name__.split('.')[-1]}-goldset.jsonl"
        goldset.write_text(
            "\n".join(__import__("json").dumps(row) for row in _goldset_rows()) + "\n",
            encoding="utf-8",
        )
        return module.run_rag_baseline_comparison(
            goldset_path=goldset,
            comparison_output_path=tmp_path / f"{module.__name__}-comparison.json",
            failures_output_path=tmp_path / f"{module.__name__}-failures.json",
        )
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def _without_nondeterministic_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_nondeterministic_fields(item)
            for key, item in value.items()
            if key not in {"generated_at", "goldset_path"}
        }
    if isinstance(value, list):
        return [_without_nondeterministic_fields(item) for item in value]
    return value


def test_pre_split_compatibility_surface_and_signatures_are_preserved() -> None:
    required_names = {name for name in dir(REFERENCE) if not name.startswith("__")}
    assert not required_names.difference(dir(candidate))
    assert candidate.__all__ == REFERENCE.__all__
    assert _callable_signatures(candidate) == _callable_signatures(REFERENCE)
    assert candidate.CONFIGURATIONS == REFERENCE.CONFIGURATIONS
    assert candidate.LOGICAL_SOURCE_ALIASES == REFERENCE.LOGICAL_SOURCE_ALIASES
    assert candidate.REFUSAL_INTENTS == REFERENCE.REFUSAL_INTENTS


def test_pre_split_deterministic_runner_fingerprint_is_preserved(tmp_path: Path) -> None:
    expected = _without_nondeterministic_fields(_run(REFERENCE, tmp_path))
    actual = _without_nondeterministic_fields(_run(candidate, tmp_path))

    assert list(actual) == list(expected)
    assert list(actual["summary"]) == list(expected["summary"])
    assert [row["configuration"] for row in actual["rows"]] == [
        row["configuration"] for row in expected["rows"]
    ]
    assert actual == expected


def test_pre_split_helper_metrics_and_errors_are_preserved(tmp_path: Path) -> None:
    corpus = _corpus()
    query = "CBC white blood cell monitoring"
    assert candidate._bm25_only_retrieval(query, corpus, limit=2) == (
        REFERENCE._bm25_only_retrieval(query, corpus, limit=2)
    )

    case = _goldset_rows()[0]
    ranked = corpus[:2]
    assert candidate._score_case("compat", case, ranked, 4.25) == (
        REFERENCE._score_case("compat", case, ranked, 4.25)
    )

    for suffix, expected_exception in (
        ("missing.jsonl", FileNotFoundError),
        ("empty.jsonl", ValueError),
    ):
        path = tmp_path / suffix
        if suffix.startswith("empty"):
            path.write_text("", encoding="utf-8")
        for module in (REFERENCE, candidate):
            with pytest.raises(expected_exception) as caught:
                module._load_goldset(path)
            assert str(caught.value).endswith(str(path))
