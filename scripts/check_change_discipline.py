"""Fail when production service code changes without a test change.

    python scripts/check_change_discipline.py --base origin/main --head HEAD

The rule is deliberately coarse: if a commit or pull request changes Python
under ``backend/services/``, it must also change something under ``tests/``.

Why not match test files to source files
----------------------------------------
Guessing that `backend/services/agent_rag.py` is covered by
`tests/test_agent_rag.py` is wrong often enough to be worse than useless. Real
coverage for one service is spread across integration suites, contract tests,
and evaluation harnesses that share no name with it, so a filename rule would
reject correct work and be routed around within a week. Requiring *some* test
change is a claim the tool can actually justify; a reviewer judges whether the
test is the right one, which is a judgement no checker makes well.

What is exempt, and how it is decided
-------------------------------------
* **Documentation-only changes.** Each changed service module is parsed before
  and after; if the syntax trees are identical once docstrings are stripped,
  the change touched only comments, docstrings, or formatting, and no test can
  be expected. This is a structural comparison, not a heuristic on the diff
  text.
* **Generated code.** Alembic revisions under ``migrations/versions/``.
* **Deletions.** Removing a module is not a behaviour change needing new tests.

Everything else that changes a service's parsed structure needs a test change
in the same diff.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WATCHED_PREFIX = "backend/services/"
TEST_PREFIX = "tests/"
GENERATED_PATH_PARTS = ("migrations/versions",)


def _git(args: list[str], root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def changed_files(base: str, head: str, root: Path = ROOT) -> list[tuple[str, str]]:
    """(status, path) for each file changed between two refs."""
    output = _git(["diff", "--name-status", f"{base}...{head}"], root)
    entries = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            entries.append((parts[0].strip(), parts[-1].strip().replace("\\", "/")))
    return entries


def is_generated(path: str) -> bool:
    return any(part in path for part in GENERATED_PATH_PARTS)


def _normalized_tree(source: str) -> str | None:
    """AST dump with docstrings removed, or None if the source will not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def _blob(ref: str, path: str, root: Path = ROOT) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=root, capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def is_documentation_only(path: str, base: str, head: str, root: Path = ROOT) -> bool:
    """True when the change did not alter the module's parsed structure."""
    before, after = _blob(base, path, root), _blob(head, path, root)
    if before is None or after is None:
        return False
    before_tree, after_tree = _normalized_tree(before), _normalized_tree(after)
    if before_tree is None or after_tree is None:
        return False
    return before_tree == after_tree


def evaluate(
    entries: list[tuple[str, str]], base: str, head: str, root: Path = ROOT
) -> tuple[list[str], list[str], list[str]]:
    """Return (service changes needing tests, exempt changes, test changes)."""
    needs_tests: list[str] = []
    exempt: list[str] = []
    test_changes: list[str] = []

    for status, path in entries:
        if path.startswith(TEST_PREFIX) and path.endswith(".py"):
            test_changes.append(path)
        if not path.startswith(WATCHED_PREFIX) or not path.endswith(".py"):
            continue
        if status.startswith("D"):
            exempt.append(f"{path} (deleted)")
        elif is_generated(path):
            exempt.append(f"{path} (generated)")
        elif status.startswith("M") and is_documentation_only(path, base, head, root):
            exempt.append(f"{path} (documentation only)")
        else:
            needs_tests.append(path)
    return needs_tests, exempt, test_changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="base ref (default: origin/main)")
    parser.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    args = parser.parse_args(argv)

    try:
        entries = changed_files(args.base, args.head)
    except subprocess.CalledProcessError as exc:
        print(
            f"change discipline: cannot diff {args.base}...{args.head}: "
            f"{exc.stderr.strip() or exc}",
            file=sys.stderr,
        )
        return 2

    needs_tests, exempt, test_changes = evaluate(entries, args.base, args.head)

    for path in exempt:
        print(f"  exempt: {path}")

    if not needs_tests:
        print(
            "Change discipline: PASSED (no production service change requiring tests"
            f"; {len(test_changes)} test file(s) changed)"
        )
        return 0

    if test_changes:
        print(
            f"Change discipline: PASSED ({len(needs_tests)} service file(s) changed, "
            f"{len(test_changes)} test file(s) changed)"
        )
        for path in needs_tests:
            print(f"  service: {path}")
        for path in test_changes:
            print(f"  test:    {path}")
        return 0

    print("Change discipline: FAILED")
    print(
        f"{len(needs_tests)} production file(s) under {WATCHED_PREFIX} changed with no "
        f"change under {TEST_PREFIX}:"
    )
    for path in needs_tests:
        print(f"  - {path}")
    print()
    print(
        "Add the test that would have failed before this change. If the change "
        "genuinely cannot carry one, say why in the pull request - see "
        "CONTRIBUTING.md, 'Source changes ship with tests'."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
