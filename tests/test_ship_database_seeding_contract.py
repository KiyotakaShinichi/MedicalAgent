"""The Ship Gate's database has to exist, have a schema, and hold demo data.

Demo credentials are not a separate account table: `create_demo_session_from_
credentials` resolves a username by looking up the patient row. So an
unmigrated database does not fail as "no such table", it fails as **401 on
/auth/demo-credential-login**, which reads like an authentication problem and
is not one. Three progressive-report tests failed that way on a runner whose
ship database was an empty file with no schema at all.

The fix is data, not auth, and these tests pin that distinction: the workflow
must initialize and seed its disposable database, and must not reach for any
switch that widens the authentication surface instead.

`scripts/reset_local_db.py` is the initializer already used by
`scripts/run_playwright_backend.py` for its own throwaway database. Reusing it
keeps one migration-and-seed path rather than a workflow-shaped copy, and
because it takes the URL from the job environment it can only ever touch the
ephemeral SQLite file the job declares.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHIP_WORKFLOW = ROOT / ".github/workflows/ship.yml"
PLAYWRIGHT_BACKEND = ROOT / "scripts/run_playwright_backend.py"

INITIALIZER = "scripts/reset_local_db.py"
GATE = "scripts/ship.py"

# Env vars that would widen the demo authentication surface rather than
# provide the data the tests actually need.
AUTH_WIDENING_VARS = ("ALLOW_DEMO_AUTH", "NLCARE_BOOTSTRAP_SYNTHETIC_DEMO")


def _ship_job() -> dict:
    return yaml.safe_load(SHIP_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["ship"]


def _steps() -> list[dict]:
    return _ship_job().get("steps") or []


def _lines() -> list[str]:
    lines: list[str] = []
    for step in _steps():
        lines.extend(str(step.get("run") or "").splitlines())
    return lines


def _first(needle: str) -> int | None:
    return next((i for i, line in enumerate(_lines()) if needle in line), None)


def _database_url() -> str:
    return str((_ship_job().get("env") or {}).get("DATABASE_URL") or "")


# --- the database is prepared, in the right order ----------------------------


def test_the_ship_gate_initializes_its_database() -> None:
    assert _first(INITIALIZER) is not None, (
        "the Ship Gate never migrates or seeds its database; demo login will "
        "401 because there is no patient row to resolve a username against"
    )


def test_the_database_directory_exists_before_initialization() -> None:
    """`reset_local_db.py` writes a SQLite file; the parent must be there."""
    lines = _lines()
    mkdir_at = next(
        (i for i, line in enumerate(lines) if "mkdir" in line and "Data/test_tmp" in line),
        None,
    )
    init_at = _first(INITIALIZER)

    assert mkdir_at is not None, "nothing creates Data/test_tmp"
    assert init_at is not None
    assert mkdir_at < init_at, (
        f"the directory is created at line {mkdir_at} but the database is "
        f"initialized at {init_at}"
    )


def test_initialization_happens_before_the_gate_runs() -> None:
    init_at = _first(INITIALIZER)
    gate_at = _first(GATE)

    assert init_at is not None, "no database initialization step"
    assert gate_at is not None, "no ship.py step"
    assert init_at < gate_at, (
        f"the database is initialized at line {init_at} but the gate starts at "
        f"{gate_at}; the suites would still see an empty database"
    )


def test_initialization_failure_is_not_swallowed() -> None:
    for step in _steps():
        if INITIALIZER in str(step.get("run") or ""):
            assert step.get("continue-on-error") is not True, (
                "database initialization is continue-on-error; the gate would "
                "run against an unseeded database and fail confusingly"
            )


# --- it is the canonical initializer, not a second one -----------------------


def test_the_canonical_initializer_is_reused() -> None:
    """The same script the Playwright backend uses for its throwaway database."""
    assert INITIALIZER in PLAYWRIGHT_BACKEND.read_text(encoding="utf-8"), (
        f"{PLAYWRIGHT_BACKEND.name} no longer uses {INITIALIZER}; the Ship Gate "
        "would be seeding by a path nothing else exercises"
    )


def test_no_seeding_logic_is_inlined_into_the_workflow() -> None:
    """No hand-rolled SQL, no second seeder."""
    text = SHIP_WORKFLOW.read_text(encoding="utf-8")
    for marker in ("INSERT INTO", "sqlite3 ", "CREATE TABLE", "alembic upgrade"):
        assert marker not in text, (
            f"the workflow performs its own database setup ({marker!r}); it "
            f"should delegate to {INITIALIZER}"
        )


# --- it can only touch the ephemeral test database ---------------------------


def test_the_ship_database_is_an_ephemeral_sqlite_file() -> None:
    url = _database_url()
    assert url.startswith("sqlite:"), f"the ship database is not SQLite: {url!r}"
    assert "Data/test_tmp/" in url, (
        f"the ship database lives outside the disposable directory: {url!r}"
    )


def test_the_initializer_takes_the_url_from_the_job_environment() -> None:
    """So it cannot be pointed at anything but this job's own database."""
    line = next(line for line in _lines() if INITIALIZER in line)
    assert "$DATABASE_URL" in line or "--database-url" not in line, (
        f"the initializer is given a hardcoded URL: {line.strip()!r}"
    )


@pytest.mark.parametrize("scheme", ["postgres", "mysql", "mssql", "oracle"])
def test_no_server_database_can_be_targeted(scheme: str) -> None:
    """A networked database would not be disposable, and might not be ours."""
    assert scheme not in SHIP_WORKFLOW.read_text(encoding="utf-8").lower()


def test_no_generated_ship_database_is_committed() -> None:
    """The disposable database is rebuilt each run, so it must not be in git.

    Scoped to the disposable directory rather than every `*.db` in the tree:
    `Data/medicalagent.db` is a pre-existing tracked file that predates this
    workflow and is not what this contract governs.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", "Data/test_tmp"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not result.stdout.strip(), (
        f"the disposable ship database directory has tracked files: {result.stdout}"
    )

    url = _database_url()
    ship_db = url.split("///")[-1].lstrip("./") if "///" in url else ""
    if ship_db:
        tracked = subprocess.run(
            ["git", "ls-files", "--", ship_db],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert not tracked.stdout.strip(), f"the ship database is committed: {ship_db}"


# --- data, not an authentication shortcut ------------------------------------


@pytest.mark.parametrize("variable", AUTH_WIDENING_VARS)
def test_no_authentication_switch_is_enabled(variable: str) -> None:
    """The failure was missing data; widening demo auth would mask it instead.

    `ALLOW_DEMO_AUTH` forces demo login on for production-shaped profiles, and
    the synthetic-demo bootstrap flag exists to top up demo accounts. Seeding
    the database properly needs neither, so neither belongs here.
    """
    assert variable not in SHIP_WORKFLOW.read_text(encoding="utf-8"), (
        f"{variable} is set in the Ship Gate; the database seeding above "
        "already provides what the tests need, without touching auth posture"
    )


def test_demo_auth_still_resolves_through_the_patient_record() -> None:
    """The behaviour the seeding relies on, asserted so it cannot drift.

    If demo credentials stopped being resolved from the patient table, seeding
    patients would no longer be the right fix and this contract would be
    quietly describing the wrong mechanism.
    """
    source = (ROOT / "backend/services/auth.py").read_text(encoding="utf-8")
    assert "_patient_from_demo_username" in source, (
        "demo credential resolution no longer looks up a patient row"
    )


# --- offline ------------------------------------------------------------------


def test_initialization_requires_no_network() -> None:
    text = SHIP_WORKFLOW.read_text(encoding="utf-8")
    assert "NLCARE_ALLOW_TEST_NETWORK" not in text

    for script in (INITIALIZER, "seed_db.py"):
        source = (ROOT / script).read_text(encoding="utf-8")
        for marker in ("import requests", "import httpx", "urllib.request", "urlopen"):
            assert marker not in source, f"{script} appears to reach the network"
        assert not re.search(r"https?://\S+\"", source), f"{script} embeds a URL"
