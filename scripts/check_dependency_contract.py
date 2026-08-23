"""Validate NLCare's canonical dependency declaration.

Source of truth
---------------
``pyproject.toml`` plus ``uv.lock`` are the only Python dependency source of
truth. ``uv sync --frozen`` is the only supported way to build a development or
CI environment, and the container image resolves the same lockfile rather than
a manifest maintained beside it.

Profiles are expressed as uv dependency groups, not as separate files:

* ``[project].dependencies`` is the minimal request-path runtime, which is what
  the serving image installs (``uv sync --frozen --no-default-groups``, and the
  equivalent ``uv export`` in the ``Dockerfile``);
* ``ml`` carries training, embedding, and explainability;
* ``reporting`` carries figure and document generation;
* ``dev`` carries the test and tooling stack.

``[tool.uv] default-groups`` lists every one of those groups, so a plain
``uv sync --frozen`` still installs the complete environment a developer and CI
had before the profiles were split apart. That is the property check 4 below
protects: if a group were dropped from ``default-groups``, the default install
would silently shrink and CI would start running against a different
environment than the one the lockfile describes.

Why the removed manifests are not allowed back
----------------------------------------------
``requirements.txt`` and ``requirements-serving.txt`` used to be hand-maintained
exports of the canonical source. Being exact-pinned never made them correct: two
files can both be perfectly pinned and still disagree, which shipped one
dependency set to the container and a different one to every test. The drift
could only be caught by a contract that compared them - so the contract is now
that they do not exist.

This module enforces:

1. ``pyproject.toml`` and ``uv.lock`` are present;
2. every project dependency and every dependency-group entry is exact-pinned;
3. no removed manifest has reappeared at the repository root;
4. every declared dependency group is listed in ``[tool.uv] default-groups``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Manifests that were consolidated into pyproject.toml + uv.lock. A file
# reappearing here is a second source of truth, which is the exact failure the
# consolidation removed.
REMOVED_MANIFESTS = ("requirements.txt", "requirements-serving.txt")

REINTRODUCTION_GUIDANCE = (
    "pyproject.toml + uv.lock are canonical. Install with `uv sync --frozen`; "
    "build the minimal serving profile with `uv sync --frozen --no-default-groups`. "
    "To add a dependency, edit pyproject.toml and run `uv lock` - do not add a "
    "requirements file, which becomes a second source of truth that drifts silently."
)

# name, optional extras, pinned version.
_REQUIREMENT_PARTS = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==(?P<version>[^\s;]+)"
)


def _canonical_name(name: str) -> str:
    """PEP 503 normalisation, so `types-PyYAML` and `types_pyyaml` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _pinned_versions(requirements: list[str]) -> dict[str, str]:
    """Map canonical distribution name -> pinned version, ignoring markers."""
    versions: dict[str, str] = {}
    for requirement in requirements:
        match = _REQUIREMENT_PARTS.match(requirement)
        if match:
            versions[_canonical_name(match.group("name"))] = match.group("version")
    return versions


def validate_dependency_contract(root: Path = ROOT) -> list[str]:
    issues: list[str] = []
    pyproject_path = root / "pyproject.toml"
    lock_path = root / "uv.lock"
    if not pyproject_path.is_file():
        return ["missing pyproject.toml"]
    if not lock_path.is_file():
        issues.append("missing uv.lock")

    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = project.get("project", {}).get("dependencies", [])
    if not dependencies:
        issues.append("project.dependencies is empty")
    for requirement in dependencies:
        if "==" not in str(requirement):
            issues.append(f"unpinned project dependency: {requirement}")

    groups = project.get("dependency-groups") or {}
    for group_name, entries in sorted(groups.items()):
        for requirement in entries:
            if "==" not in str(requirement):
                issues.append(
                    f"unpinned dependency in group '{group_name}': {requirement}"
                )

    issues.extend(_removed_manifest_issues(root))
    issues.extend(_default_group_issues(project, groups))
    return issues


def _removed_manifest_issues(root: Path) -> list[str]:
    """Fail if a consolidated manifest has been recreated."""
    issues: list[str] = []
    for relative_path in REMOVED_MANIFESTS:
        if (root / relative_path).exists():
            issues.append(
                f"{relative_path} has reappeared at the repository root. "
                + REINTRODUCTION_GUIDANCE
            )
    return issues


def _default_group_issues(project: dict, groups: dict) -> list[str]:
    """Every declared group must be installed by a plain `uv sync --frozen`.

    The groups exist so the *serving* image can opt out, not so the default
    developer or CI environment can quietly become smaller. A group missing
    here would drop packages from every `uv sync --frozen` without any manifest
    appearing to change.
    """
    issues: list[str] = []
    if not groups:
        return issues
    default_groups = (project.get("tool", {}).get("uv", {}) or {}).get("default-groups")
    if default_groups is None:
        issues.append(
            "[tool.uv] default-groups is not declared, so `uv sync --frozen` would "
            f"install only the 'dev' group and silently drop: "
            f"{', '.join(sorted(set(groups) - {'dev'}))}"
        )
        return issues
    missing = sorted(set(groups) - set(default_groups))
    if missing:
        issues.append(
            "dependency groups declared but absent from [tool.uv] default-groups, so "
            f"`uv sync --frozen` would not install them: {', '.join(missing)}"
        )
    unknown = sorted(set(default_groups) - set(groups))
    if unknown:
        issues.append(
            "[tool.uv] default-groups names groups that do not exist: "
            f"{', '.join(unknown)}"
        )
    return issues


def main() -> int:
    issues = validate_dependency_contract()
    if issues:
        print("Dependency contract: FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Dependency contract: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
