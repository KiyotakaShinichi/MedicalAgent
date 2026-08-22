from __future__ import annotations

from pathlib import Path

from scripts.check_dependency_contract import validate_dependency_contract

# `pyproject.toml` + `uv.lock` are canonical. `requirements.txt` and
# `requirements-serving.txt` are maintained compatibility exports: the container
# installs with plain pip and never runs uv, and the serving profile omits the
# training/evaluation stack. Because they are hand-maintained, they can drift
# from the canonical source while staying perfectly pinned — which is the case
# these tests exist to catch.

_PYPROJECT = '[project]\ndependencies = ["fastapi==0.136.1", "pandas==3.0.2"]\n'


def _manifest_tree(
    tmp_path: Path, *, pyproject: str, requirements: str, serving: str
) -> Path:
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(requirements, encoding="utf-8")
    (tmp_path / "requirements-serving.txt").write_text(serving, encoding="utf-8")
    return tmp_path


def test_repository_dependency_contract_is_reproducible() -> None:
    assert validate_dependency_contract() == []


def test_dependency_contract_rejects_unpinned_compatibility_file(tmp_path: Path) -> None:
    root = _manifest_tree(
        tmp_path,
        pyproject='[project]\ndependencies = ["fastapi==0.136.1"]\n',
        requirements="fastapi\n",
        serving="fastapi==0.136.1\n",
    )
    # Membership rather than equality: the contract now also reports that the
    # unpinned line leaves the canonical runtime dependency unrepresented,
    # which is a second and correct finding about the same entry.
    assert (
        "unpinned compatibility dependency in requirements.txt: fastapi"
        in validate_dependency_contract(root)
    )


def test_contract_detects_version_drift_from_pyproject(tmp_path: Path) -> None:
    """Both manifests exact-pinned, to *different* versions.

    Pinning alone never caught this: the container would install one version
    while every test ran against another.
    """
    root = _manifest_tree(
        tmp_path,
        pyproject=_PYPROJECT,
        requirements="fastapi==0.135.0\npandas==3.0.2\n",
        serving="fastapi==0.135.0\n",
    )
    assert (
        "dependency drift for fastapi: pyproject.toml pins 0.136.1 "
        "but requirements.txt pins 0.135.0"
    ) in validate_dependency_contract(root)


def test_contract_detects_runtime_dependency_missing_from_requirements(
    tmp_path: Path,
) -> None:
    """A dependency added to pyproject but forgotten in the pip manifest."""
    root = _manifest_tree(
        tmp_path,
        pyproject=_PYPROJECT,
        requirements="fastapi==0.136.1\n",
        serving="fastapi==0.136.1\n",
    )
    assert (
        "requirements.txt is missing runtime dependency pandas==3.0.2 "
        "declared in pyproject.toml"
    ) in validate_dependency_contract(root)


def test_contract_detects_serving_profile_divergence(tmp_path: Path) -> None:
    """The serving image must be a subset of the full profile, not a variant."""
    root = _manifest_tree(
        tmp_path,
        pyproject=_PYPROJECT,
        requirements="fastapi==0.136.1\npandas==3.0.2\n",
        serving="fastapi==0.130.0\n",
    )
    assert (
        "dependency drift for fastapi: requirements.txt pins 0.136.1 "
        "but requirements-serving.txt pins 0.130.0"
    ) in validate_dependency_contract(root)


def test_contract_detects_dependency_group_drift(tmp_path: Path) -> None:
    """An extra entry that pyproject also declares must still agree with it."""
    root = _manifest_tree(
        tmp_path,
        pyproject=_PYPROJECT + '\n[dependency-groups]\ndev = ["pytest==9.0.3"]\n',
        requirements="fastapi==0.136.1\npandas==3.0.2\npytest==8.0.0\n",
        serving="fastapi==0.136.1\n",
    )
    assert (
        "dependency drift for pytest: pyproject.toml dependency group pins 9.0.3 "
        "but requirements.txt pins 8.0.0"
    ) in validate_dependency_contract(root)


def test_contract_allows_pip_only_development_extras(tmp_path: Path) -> None:
    """Guards against over-strictness.

    `requirements.txt` doubles as the pip-only development profile, so it
    legitimately carries a test runner that is not a runtime dependency. A
    contract that rejected extras outright would force the test runner out of
    the only file pip-only users install from.
    """
    root = _manifest_tree(
        tmp_path,
        pyproject=_PYPROJECT,
        requirements="fastapi==0.136.1\npandas==3.0.2\npytest==9.0.3\n",
        serving="fastapi==0.136.1\n",
    )
    assert validate_dependency_contract(root) == []
