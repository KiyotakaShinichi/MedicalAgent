"""The source+test discipline check accepts and rejects the right diffs.

A check like this earns its place only if it is accurate in both directions.
False rejections get it routed around within a week; false acceptances make it
decoration. So both are asserted: a service change without tests fails, and a
documentation-only change, a deletion, and a generated file all pass.

The exemption for documentation is structural — the module is parsed before and
after and the syntax trees compared with docstrings stripped — so these tests
drive that comparison with real source rather than with diff text.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_change_discipline import (  # noqa: E402
    TEST_PREFIX,
    WATCHED_PREFIX,
    evaluate,
    is_generated,
    main,
)


def _run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git repository, so the check is exercised end to end."""
    _run(["git", "init", "-q", "-b", "main"], tmp_path)
    _run(["git", "config", "user.email", "t@example.test"], tmp_path)
    _run(["git", "config", "user.name", "Test"], tmp_path)
    service = tmp_path / "backend" / "services"
    service.mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (service / "thing.py").write_text(
        '"""Original docstring."""\n\n\ndef add(a, b):\n    return a + b\n',
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_thing.py").write_text("def test_add():\n    pass\n", encoding="utf-8")
    _run(["git", "add", "-A"], tmp_path)
    _run(["git", "commit", "-q", "-m", "initial"], tmp_path)
    return tmp_path


def _commit(repo: Path, message: str) -> None:
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-q", "-m", message], repo)


def _changed(repo: Path):
    from scripts.check_change_discipline import changed_files

    return changed_files("HEAD~1", "HEAD", repo)


# ─── rejection ───────────────────────────────────────────────────────────────


def test_service_change_without_tests_is_rejected(repo: Path) -> None:
    (repo / "backend" / "services" / "thing.py").write_text(
        '"""Original docstring."""\n\n\ndef add(a, b):\n    return a + b + 1\n', encoding="utf-8"
    )
    _commit(repo, "change behaviour")

    needs_tests, exempt, test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests == ["backend/services/thing.py"]
    assert test_changes == []
    assert exempt == []


def test_new_service_file_without_tests_is_rejected(repo: Path) -> None:
    (repo / "backend" / "services" / "brand_new.py").write_text(
        "def f():\n    return 1\n", encoding="utf-8"
    )
    _commit(repo, "add service")

    needs_tests, _exempt, test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests == ["backend/services/brand_new.py"]
    assert test_changes == []


# ─── acceptance ──────────────────────────────────────────────────────────────


def test_service_change_with_a_test_change_is_accepted(repo: Path) -> None:
    (repo / "backend" / "services" / "thing.py").write_text(
        '"""Original docstring."""\n\n\ndef add(a, b):\n    return a + b + 1\n', encoding="utf-8"
    )
    (repo / "tests" / "test_thing.py").write_text(
        "def test_add():\n    assert True\n", encoding="utf-8"
    )
    _commit(repo, "change behaviour with tests")

    needs_tests, _exempt, test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests == ["backend/services/thing.py"]
    assert test_changes == ["tests/test_thing.py"]


def test_docstring_only_change_is_exempt(repo: Path) -> None:
    """The parsed structure is unchanged, so no test could be expected."""
    (repo / "backend" / "services" / "thing.py").write_text(
        '"""Rewritten docstring explaining the function better."""\n\n\n'
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    _commit(repo, "docs only")

    needs_tests, exempt, _test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests == []
    assert exempt == ["backend/services/thing.py (documentation only)"]


def test_comment_only_change_is_exempt(repo: Path) -> None:
    (repo / "backend" / "services" / "thing.py").write_text(
        '"""Original docstring."""\n\n\ndef add(a, b):\n    # why this is a sum\n    return a + b\n',
        encoding="utf-8",
    )
    _commit(repo, "comment only")

    needs_tests, exempt, _test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests == []
    assert "documentation only" in exempt[0]


def test_a_real_change_hidden_behind_a_docstring_edit_is_not_exempt(repo: Path) -> None:
    """Editing the docstring must not launder a behaviour change past the check."""
    (repo / "backend" / "services" / "thing.py").write_text(
        '"""Rewritten docstring."""\n\n\ndef add(a, b):\n    return a * b\n', encoding="utf-8"
    )
    _commit(repo, "sneaky")

    needs_tests, exempt, _test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests == ["backend/services/thing.py"]
    assert exempt == []


def test_deleted_service_file_is_exempt(repo: Path) -> None:
    (repo / "backend" / "services" / "thing.py").unlink()
    _commit(repo, "remove service")

    needs_tests, exempt, _test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests == []
    assert exempt == ["backend/services/thing.py (deleted)"]


def test_changes_outside_the_watched_prefix_are_ignored(repo: Path) -> None:
    (repo / "README.md").write_text("# docs\n", encoding="utf-8")
    (repo / "backend" / "other.py").write_text("x = 1\n", encoding="utf-8")
    _commit(repo, "unrelated")

    needs_tests, exempt, _test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests == []
    assert exempt == []


def test_generated_migrations_are_exempt() -> None:
    assert is_generated("backend/services/migrations/versions/0001_x.py")
    assert not is_generated("backend/services/agent_rag.py")


def test_any_test_file_satisfies_the_requirement(repo: Path) -> None:
    """Coverage for one service is often spread across unrelated suites.

    Requiring a name-matched test file would reject correct work, so any test
    change counts and a reviewer judges whether it is the right one.
    """
    (repo / "backend" / "services" / "thing.py").write_text(
        '"""Original docstring."""\n\n\ndef add(a, b):\n    return a + b + 1\n', encoding="utf-8"
    )
    (repo / "tests" / "test_completely_different.py").write_text(
        "def test_other():\n    assert True\n", encoding="utf-8"
    )
    _commit(repo, "change with a differently-named test")

    needs_tests, _exempt, test_changes = evaluate(_changed(repo), "HEAD~1", "HEAD", repo)
    assert needs_tests and test_changes


# ─── the CLI ─────────────────────────────────────────────────────────────────


def test_cli_reports_a_bad_ref_as_an_error_not_a_pass() -> None:
    """A broken ref in CI must not look green."""
    assert main(["--base", "definitely-not-a-ref", "--head", "HEAD"]) == 2


def test_prefixes_are_the_ones_documented() -> None:
    assert WATCHED_PREFIX == "backend/services/"
    assert TEST_PREFIX == "tests/"


def test_contributing_documents_the_rule() -> None:
    """The check and the written policy must not drift apart."""
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "check_change_discipline.py" in contributing
    assert "Source changes ship with tests" in contributing
