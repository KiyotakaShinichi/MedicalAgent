"""A resolved advisory must not be reintroduced by a later pin.

`pypdf==6.15.0` carried three advisories - GHSA-jp53-mhqp-8xcg (fixed in
6.16.0), GHSA-23w6-3w8w-8484 and GHSA-763m-79hh-57f2 (both fixed in 6.16.1).
CI's `dependency-audit` job caught them against the live advisory database, on
an unchanged tree: the pin was five days old and the advisories were newer.

That job stays the real detector, because it learns about advisories published
after this file was written and this file never will. What it cannot do is run
offline, and it reports on whatever version happens to be resolved rather than
on the decision recorded in the repository. So the narrow thing pinned here is
the decision itself: the floor that closed these three advisories, and the
agreement between the two files that are supposed to state it identically.

A future bump raises the floor and this test keeps passing. A downgrade below
it fails here, hermetically, instead of waiting for a network audit to notice.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Lowest release in which all three advisories above are fixed.
MINIMUM_REMEDIATED = (6, 16, 1)


def _version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split("."))


def _pyproject_pin() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = [d for d in data["project"]["dependencies"] if d.startswith("pypdf==")]
    assert len(pins) == 1, f"pypdf is pinned {len(pins)} times, expected exactly once"
    return pins[0].split("==", 1)[1]


def _lock_pin() -> str:
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(r'^name = "pypdf"\nversion = "([^"]+)"', lock, re.MULTILINE)
    assert match, "pypdf is absent from uv.lock"
    return match.group(1)


def test_the_pinned_pypdf_is_at_or_above_the_remediated_release() -> None:
    pinned = _pyproject_pin()
    assert _version_tuple(pinned) >= MINIMUM_REMEDIATED, (
        f"pypdf=={pinned} reintroduces GHSA-jp53-mhqp-8xcg, GHSA-23w6-3w8w-8484 "
        f"or GHSA-763m-79hh-57f2; 6.16.1 is the lowest release that fixes all three"
    )


def test_the_lockfile_resolves_the_version_pyproject_declares() -> None:
    """Pinning is not agreement: the audit runs against whatever uv.lock resolves."""
    assert _lock_pin() == _pyproject_pin()
