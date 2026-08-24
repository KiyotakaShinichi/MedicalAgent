"""The agent_rag facade is a strict superset of the module it replaced.

`agent_rag` was carved down over several rounds; branch execution now lives in
`agent_pipeline_runner` and response shaping in `agent_result_shaping`. Four
earlier refactors in this repository broke consumers that depended on the old
module's *imported* attributes, its source path, or a monkeypatch seam - none of
which an AST comparison of defined symbols would notice.

So this compares the real thing: the pre-split module is loaded from git and its
runtime `dir()` and callable signatures are diffed against the current facade.

The seam that matters most is the last section. Tests patch
`agent_rag._run_patient_agent_pipeline_impl` and expect
`run_patient_agent_pipeline` to call the patched object. That only works while
the name is bound in *this* module's namespace and called unqualified - the same
indirection the module already documents for `route_intent_with_local_llm`.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import agent_rag  # noqa: E402

ORIGINAL_PATH = "backend/services/agent_rag.py"

EXTRACTED_MODULES = (
    "backend.services.agent_pipeline_runner",
    "backend.services.agent_result_shaping",
)


@pytest.fixture(scope="module")
def pre_split_module(tmp_path_factory):
    """The module as it was before this split, loaded from git."""
    source = subprocess.run(
        ["git", "show", f"HEAD:{ORIGINAL_PATH}"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    path = tmp_path_factory.mktemp("pre-split") / "agent_rag_original.py"
    path.write_text(source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("_agent_rag_pre_split", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_agent_rag_pre_split"] = module
    spec.loader.exec_module(module)
    return module


# ─── the compatibility surface ───────────────────────────────────────────────


def test_every_pre_split_attribute_survives(pre_split_module) -> None:
    """Including names the module only ever imported.

    This is the check the earlier AST-based version could not make, and the one
    that would have caught `_calibration_metrics` disappearing from the admin
    facade in the previous sprint.
    """
    before = {name for name in dir(pre_split_module) if not name.startswith("__")}
    after = {name for name in dir(agent_rag) if not name.startswith("__")}

    missing = sorted(before - after)
    assert not missing, f"the split dropped module attributes: {missing}"


def test_callable_signatures_are_unchanged(pre_split_module) -> None:
    drift = []
    for name in dir(pre_split_module):
        if name.startswith("__"):
            continue
        original = getattr(pre_split_module, name)
        current = getattr(agent_rag, name, None)
        if not callable(original) or not callable(current):
            continue
        try:
            before, after = inspect.signature(original), inspect.signature(current)
        except (TypeError, ValueError):
            continue
        if str(before) != str(after):
            drift.append(f"{name}: {before} -> {after}")
    assert not drift, f"signatures changed: {drift}"


def test_the_public_entrypoint_is_still_here() -> None:
    """Callers import this by name from this module; it must not move."""
    assert callable(agent_rag.run_patient_agent_pipeline)
    assert agent_rag.run_patient_agent_pipeline.__module__ == "backend.services.agent_rag"


# ─── the monkeypatch seam ────────────────────────────────────────────────────


def test_patching_the_impl_on_this_module_is_authoritative(monkeypatch) -> None:
    """The seam four refactors could have broken, asserted directly.

    `run_patient_agent_pipeline` must resolve the implementation from this
    module's namespace at call time. If it called
    `agent_pipeline_runner._run_patient_agent_pipeline_impl` instead, this patch
    would be silently ignored and the tests that rely on it would exercise the
    real pipeline while appearing to stub it.
    """
    sentinel = {"reply": "patched", "citations": []}
    calls = []

    def fake_impl(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(agent_rag, "_run_patient_agent_pipeline_impl", fake_impl)

    result = agent_rag.run_patient_agent_pipeline(
        db=None,
        patient_id="P001",
        query="what does CBC mean?",
        patient_context={},
        fallback_response="fallback",
    )

    assert result is sentinel, "the patched implementation was not used"
    assert len(calls) == 1
    assert calls[0]["patient_id"] == "P001"


def test_the_entrypoint_still_fails_closed(monkeypatch) -> None:
    """A pipeline exception must never leak a candidate answer.

    The deny-on-exception boundary is the last thing standing between an
    unexpected error and an unreviewed reply reaching a patient.
    """

    def exploding_impl(**_kwargs):
        raise RuntimeError("pipeline blew up")

    monkeypatch.setattr(agent_rag, "_run_patient_agent_pipeline_impl", exploding_impl)

    result = agent_rag.run_patient_agent_pipeline(
        db=None,
        patient_id="P001",
        query="what does CBC mean?",
        patient_context={},
        fallback_response="fallback",
    )

    serialized = str(result).lower()
    assert "patient_agent_pipeline_exception" in serialized
    assert "runtimeerror" in serialized
    assert "pipeline blew up" not in serialized, "the exception message leaked"


def test_route_intent_patch_seam_is_preserved() -> None:
    """The pre-existing indirection this module documents must still hold."""
    assert hasattr(agent_rag, "route_intent_with_local_llm")


# ─── module structure ────────────────────────────────────────────────────────


@pytest.mark.parametrize("module", EXTRACTED_MODULES)
def test_each_extracted_module_imports_standalone(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{module} failed to import alone:\n{result.stderr}"


@pytest.mark.parametrize(
    "module",
    EXTRACTED_MODULES + ("backend.services.agent_rag",),
)
def test_each_module_is_within_the_service_limit(module: str) -> None:
    path = ROOT / (module.replace(".", "/") + ".py")
    loc = len(path.read_bytes().decode("utf-8").splitlines())
    assert loc <= 500, f"{path.name} is {loc} LOC"


def test_the_extracted_modules_do_not_import_the_facade() -> None:
    """A back-import would make the facade and its parts mutually dependent."""
    for module in EXTRACTED_MODULES:
        source = (ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        assert "from backend.services.agent_rag import" not in source


def test_shaping_does_not_execute_the_pipeline() -> None:
    """Presenting a result must not be able to re-run one."""
    source = (ROOT / "backend/services/agent_result_shaping.py").read_text(encoding="utf-8")
    assert "agent_pipeline_runner" not in source


def test_source_path_readers_still_find_their_evidence() -> None:
    """Sibling modules cite `agent_rag.py` in their docstrings as provenance.

    Twice in this repository a path-based evidence reader was broken by a
    legitimate decomposition, so the file those citations name must keep
    existing.
    """
    assert (ROOT / ORIGINAL_PATH).is_file()
