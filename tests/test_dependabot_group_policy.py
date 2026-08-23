"""Dependabot's Python groups stay aligned with the uv dependency groups.

Dependabot groups updates by dependency *name pattern*. It has no concept of
uv dependency groups, so `.github/dependabot.yml` mirrors the `ml`,
`reporting`, and `dev` groups from `pyproject.toml` by hand.

Hand-mirrored config drifts. A package moved into the `ml` group in pyproject
would keep arriving in the runtime PR, which is the outcome the split exists to
prevent: a torch bump and a starlette bump in one PR means either reviewing
model-output risk and web-framework risk together, or reviewing neither
properly.

These tests are the drift alarm. They do not test Dependabot; they test that
the mapping we maintain still describes the dependency tree we have.
"""

from __future__ import annotations

import fnmatch
import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
PYPROJECT = ROOT / "pyproject.toml"

#: uv group -> the Dependabot group that must claim its members.
GROUP_MAPPING = {
    "ml": "python-ml",
    "reporting": "python-reporting",
    "dev": "python-dev-tooling",
}
CATCH_ALL = "python-runtime"


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    return _canonical(re.split(r"[<>=!\[;\s]", str(requirement).strip())[0])


@pytest.fixture(scope="module")
def python_update() -> dict:
    config = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    for update in config["updates"]:
        if update["package-ecosystem"] == "uv":
            return update
    raise AssertionError("no uv ecosystem entry in dependabot.yml")


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _claiming_group(name: str, groups: dict) -> str | None:
    """The first non-catch-all group whose patterns match, else the catch-all."""
    for group_name, spec in groups.items():
        if group_name == CATCH_ALL:
            continue
        for pattern in spec.get("patterns", []):
            if fnmatch.fnmatch(name, pattern):
                return group_name
    for pattern in groups.get(CATCH_ALL, {}).get("patterns", []):
        if fnmatch.fnmatch(name, pattern):
            return CATCH_ALL
    return None


# ─── the mapping holds ───────────────────────────────────────────────────────


def test_every_uv_group_has_a_dependabot_group(python_update: dict, pyproject: dict) -> None:
    groups = python_update["groups"]
    for uv_group in pyproject["dependency-groups"]:
        assert uv_group in GROUP_MAPPING, (
            f"uv group '{uv_group}' has no Dependabot mapping; add one or its "
            "updates will arrive in the runtime PR"
        )
        assert GROUP_MAPPING[uv_group] in groups


@pytest.mark.parametrize("uv_group,dependabot_group", sorted(GROUP_MAPPING.items()))
def test_uv_group_members_are_claimed_by_their_dependabot_group(
    uv_group: str, dependabot_group: str, python_update: dict, pyproject: dict
) -> None:
    """This is the drift alarm: a member falling through to the catch-all."""
    groups = python_update["groups"]
    members = [_requirement_name(r) for r in pyproject["dependency-groups"][uv_group]]

    misrouted = {
        name: _claiming_group(name, groups)
        for name in members
        if _claiming_group(name, groups) != dependabot_group
    }
    assert not misrouted, (
        f"these '{uv_group}' packages are not claimed by '{dependabot_group}': "
        f"{misrouted}. Add a pattern, or their updates land in the runtime PR."
    )


def test_runtime_dependencies_are_not_swallowed_by_a_specific_group(
    python_update: dict, pyproject: dict
) -> None:
    """The serving profile's own packages must stay in the runtime group.

    A pattern that is too broad would pull a request-path dependency into the
    ML PR, and the reviewer would stop seeing runtime changes.
    """
    groups = python_update["groups"]
    grouped_names = {
        _requirement_name(r)
        for entries in pyproject["dependency-groups"].values()
        for r in entries
    }
    runtime_only = [
        name
        for name in ("fastapi", "uvicorn", "sqlalchemy", "alembic", "redis", "pyjwt")
        if name not in grouped_names
    ]
    for name in runtime_only:
        assert _claiming_group(name, groups) == CATCH_ALL, (
            f"{name} is a request-path dependency but is claimed by "
            f"{_claiming_group(name, groups)}"
        )


def test_a_catch_all_group_exists(python_update: dict) -> None:
    """Without it, an unmatched package would get its own single-package PR."""
    groups = python_update["groups"]
    assert CATCH_ALL in groups
    assert "*" in groups[CATCH_ALL]["patterns"]


# ─── the policy that must not weaken ─────────────────────────────────────────


def test_major_updates_are_never_grouped(python_update: dict) -> None:
    """Majors need individual review; grouping them hides the risky one."""
    for name, spec in python_update["groups"].items():
        update_types = spec.get("update-types", [])
        assert "major" not in update_types, f"group {name} batches major updates"


def test_numerical_stack_majors_stay_excluded_from_automation(python_update: dict) -> None:
    """Upgrading these can move model outputs and invalidate frozen evidence."""
    ignored = {
        entry["dependency-name"]
        for entry in python_update.get("ignore", [])
        if "version-update:semver-major" in entry.get("update-types", [])
    }
    assert {"numpy", "pandas", "scikit-learn", "torch", "transformers"} <= ignored


def test_review_based_policy_is_unchanged(python_update: dict) -> None:
    """Every update stays review-based.

    A dependency bump in a system that runs safety evaluation can change
    refusal routing or numeric output without failing a type check, so nothing
    may merge itself. Asserted two ways: the stated policy is still in the
    file, and no workflow exists to enable Dependabot auto-merge.
    """
    config_text = DEPENDABOT.read_text(encoding="utf-8")
    assert "Nothing here auto-merges" in config_text

    workflows = ROOT / ".github" / "workflows"
    for workflow in workflows.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8").lower()
        if "dependabot" in text:
            assert "gh pr merge" not in text and "--auto" not in text, (
                f"{workflow.name} appears to auto-merge Dependabot PRs"
            )

    assert python_update["open-pull-requests-limit"] <= 5


def test_the_limitation_is_documented(python_update: dict) -> None:
    """The hand-mirroring is a real constraint and must not be silent.

    Comment prose wraps, so the text is normalised before matching rather than
    the sentence being kept artificially short to suit the test.
    """
    config_text = DEPENDABOT.read_text(encoding="utf-8")
    prose = " ".join(config_text.replace("#", " ").split())

    assert "LIMITATION" in prose
    assert "no concept of uv dependency groups" in prose
    assert "mirror" in prose, "the comment should say the mapping is hand-maintained"
