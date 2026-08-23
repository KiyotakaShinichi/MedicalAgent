"""`pyproject.toml` + `uv.lock` are the only Python dependency source of truth.

These tests used to check that two hand-maintained requirements manifests agreed
with pyproject. That comparison existed because the manifests could drift while
staying perfectly pinned - pinning is not agreement. The manifests are gone now,
consolidated into uv dependency groups, so the contract changed shape: instead
of proving the copies match, it proves there are no copies, and that the profile
split cannot quietly shrink the default install.
"""

from __future__ import annotations

from pathlib import Path

from scripts.check_dependency_contract import (
    REMOVED_MANIFESTS,
    validate_dependency_contract,
)

_VALID_PYPROJECT = """[project]
dependencies = ["fastapi==0.136.1", "pandas==3.0.2"]

[dependency-groups]
ml = ["torch==2.13.0"]
dev = ["pytest==9.0.3"]

[tool.uv]
default-groups = ["dev", "ml"]
"""


def _tree(tmp_path: Path, pyproject: str = _VALID_PYPROJECT) -> Path:
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return tmp_path


# ─── the repository itself ───────────────────────────────────────────────────


def test_repository_dependency_contract_is_reproducible() -> None:
    assert validate_dependency_contract() == []


def test_removed_manifests_are_absent_from_the_repository() -> None:
    """The consolidation is only real while the files stay gone."""
    root = Path(__file__).resolve().parents[1]
    present = [name for name in REMOVED_MANIFESTS if (root / name).exists()]
    assert not present, f"{present} reappeared; pyproject.toml + uv.lock are canonical"


# ─── the contract's own failure modes ────────────────────────────────────────


def test_valid_tree_reports_no_issues(tmp_path: Path) -> None:
    assert validate_dependency_contract(_tree(tmp_path)) == []


def test_contract_rejects_a_reintroduced_requirements_file(tmp_path: Path) -> None:
    """The anti-reintroduction guard, and the guidance it must carry."""
    root = _tree(tmp_path)
    (root / "requirements.txt").write_text("fastapi==0.136.1\n", encoding="utf-8")

    issues = validate_dependency_contract(root)
    assert any("requirements.txt has reappeared" in issue for issue in issues)
    assert any("uv sync --frozen" in issue for issue in issues), (
        "the failure must tell the reader what to do instead"
    )


def test_contract_rejects_a_reintroduced_serving_manifest(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "requirements-serving.txt").write_text("fastapi==0.136.1\n", encoding="utf-8")

    assert any(
        "requirements-serving.txt has reappeared" in issue
        for issue in validate_dependency_contract(root)
    )


def test_contract_rejects_unpinned_project_dependency(tmp_path: Path) -> None:
    root = _tree(tmp_path, '[project]\ndependencies = ["fastapi"]\n')
    assert "unpinned project dependency: fastapi" in validate_dependency_contract(root)


def test_contract_rejects_unpinned_group_dependency(tmp_path: Path) -> None:
    """A group is a real install profile, so it is held to the same pinning."""
    root = _tree(
        tmp_path,
        '[project]\ndependencies = ["fastapi==0.136.1"]\n\n'
        '[dependency-groups]\nml = ["torch"]\n\n'
        '[tool.uv]\ndefault-groups = ["ml"]\n',
    )
    assert "unpinned dependency in group 'ml': torch" in validate_dependency_contract(root)


def test_contract_rejects_a_group_missing_from_default_groups(tmp_path: Path) -> None:
    """This is the regression that would shrink every developer and CI install.

    Moving a package into a group is safe only while the group is installed by
    default. Drop it from `default-groups` and `uv sync --frozen` stops
    installing it, with no manifest appearing to change.
    """
    root = _tree(
        tmp_path,
        '[project]\ndependencies = ["fastapi==0.136.1"]\n\n'
        '[dependency-groups]\nml = ["torch==2.13.0"]\ndev = ["pytest==9.0.3"]\n\n'
        '[tool.uv]\ndefault-groups = ["dev"]\n',
    )
    issues = validate_dependency_contract(root)
    assert any("absent from [tool.uv] default-groups" in issue for issue in issues)
    assert any("ml" in issue for issue in issues)


def test_contract_rejects_missing_default_groups_declaration(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        '[project]\ndependencies = ["fastapi==0.136.1"]\n\n'
        '[dependency-groups]\nml = ["torch==2.13.0"]\n',
    )
    assert any(
        "default-groups is not declared" in issue
        for issue in validate_dependency_contract(root)
    )


def test_contract_rejects_default_groups_naming_an_unknown_group(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        '[project]\ndependencies = ["fastapi==0.136.1"]\n\n'
        '[dependency-groups]\nml = ["torch==2.13.0"]\n\n'
        '[tool.uv]\ndefault-groups = ["ml", "typo"]\n',
    )
    assert any(
        "names groups that do not exist" in issue
        for issue in validate_dependency_contract(root)
    )


def test_contract_reports_a_missing_lockfile(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(_VALID_PYPROJECT, encoding="utf-8")
    assert "missing uv.lock" in validate_dependency_contract(tmp_path)
