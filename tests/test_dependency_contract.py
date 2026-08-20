from __future__ import annotations

from pathlib import Path

from scripts.check_dependency_contract import validate_dependency_contract


def test_repository_dependency_contract_is_reproducible() -> None:
    assert validate_dependency_contract() == []


def test_dependency_contract_rejects_unpinned_compatibility_file(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["fastapi==0.136.1"]\n', encoding="utf-8"
    )
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (tmp_path / "requirements-serving.txt").write_text(
        "fastapi==0.136.1\n", encoding="utf-8"
    )

    assert validate_dependency_contract(tmp_path) == [
        "unpinned compatibility dependency in requirements.txt: fastapi"
    ]

