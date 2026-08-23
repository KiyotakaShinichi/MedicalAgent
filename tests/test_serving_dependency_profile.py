"""The serving image installs a genuinely minimal runtime profile.

This contract used to compare two hand-maintained manifests,
`requirements-serving.txt` against `requirements.txt`. Those files are gone:
`pyproject.toml` + `uv.lock` are canonical, and the profiles are uv dependency
groups. The property under test is unchanged and the exclusions are the same
set - what changed is where the answer is read from.

The exclusions are not cosmetic. torch, torchvision, sentence-transformers, and
shap dominate the image, and nothing on the request path imports them; shipping
them would mean a serving container carrying a training stack it never executes.

`pytest` is asserted absent for a different reason: a test runner inside a
deployed image is an attack-surface and a signal that the dev profile leaked
into production.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Excluded from the serving image, and asserted rather than assumed.
EXCLUDED_FROM_SERVING = {
    "torch",
    "torchvision",
    "shap",
    "matplotlib",
    "reportlab",
    "pytest",
    "sentence-transformers",
}

# The request path genuinely needs these.
REQUIRED_IN_SERVING = {"fastapi", "uvicorn", "sqlalchemy", "pandas", "scikit-learn"}


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _names(requirements) -> set[str]:
    """Distribution names from a list of PEP 508 requirement strings."""
    out = set()
    for requirement in requirements:
        match = re.match(r"^([A-Za-z0-9_.-]+)", str(requirement).strip())
        if match:
            out.add(_canonical(match.group(1)))
    return out


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_serving_profile_keeps_api_dependencies_and_excludes_offline_research():
    """`[project].dependencies` is what the serving image installs."""
    project = _pyproject()
    serving = _names(project["project"]["dependencies"])

    assert REQUIRED_IN_SERVING <= serving, (
        f"serving profile is missing request-path packages: "
        f"{sorted(REQUIRED_IN_SERVING - serving)}"
    )
    leaked = EXCLUDED_FROM_SERVING & serving
    assert not leaked, (
        f"{sorted(leaked)} reached the serving profile. Move it to a dependency "
        "group (ml, reporting, or dev) so the serving image keeps excluding it."
    )


def test_full_research_profile_remains_available():
    """The heavy stack is still installed by default - moved, not dropped.

    A "minimal serving profile" achieved by deleting the ML stack from the
    repository would pass the exclusion test above and break every training and
    safety-encoder workflow.
    """
    project = _pyproject()
    groups = project["dependency-groups"]
    grouped = _names(entry for entries in groups.values() for entry in entries)

    assert {"torch", "torchvision", "shap", "sentence-transformers"} <= grouped
    assert {"matplotlib", "reportlab"} <= grouped


def test_default_install_still_gets_everything():
    """`uv sync --frozen` must not have quietly become smaller.

    CI and the safety-encoder provisioning depend on sentence-transformers, and
    the training workflow on torch. They are only reachable by default because
    every group is listed in `default-groups`.
    """
    project = _pyproject()
    default_groups = set(project["tool"]["uv"]["default-groups"])
    assert set(project["dependency-groups"]) <= default_groups

    installed_by_default = _names(project["project"]["dependencies"]) | _names(
        entry
        for group, entries in project["dependency-groups"].items()
        if group in default_groups
        for entry in entries
    )
    assert EXCLUDED_FROM_SERVING <= installed_by_default, (
        "packages excluded from serving must still be installed by default: "
        f"{sorted(EXCLUDED_FROM_SERVING - installed_by_default)}"
    )


def test_serving_and_grouped_profiles_do_not_overlap():
    """A package in both places would make the exclusion meaningless."""
    project = _pyproject()
    serving = _names(project["project"]["dependencies"])
    for group, entries in project["dependency-groups"].items():
        overlap = serving & _names(entries)
        assert not overlap, (
            f"{sorted(overlap)} is declared both as a project dependency and in "
            f"group '{group}'; the serving image would install it regardless."
        )
