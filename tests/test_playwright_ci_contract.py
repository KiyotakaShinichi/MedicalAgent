"""Playwright has to be runnable by the CI that runs it.

Two prerequisites are easy to satisfy on a developer's Windows machine and easy
to leave unsatisfied on a Linux runner, and both fail the whole gate rather
than one test:

* the browser binary. `npm ci` installs the `@playwright/test` package, which
  contains no browsers, and this project declares no `postinstall` hook. A job
  that never runs `playwright install` has a Playwright package and nothing to
  drive.

* the servers Playwright starts for itself. `playwright.config.ts` launches the
  backend through `webServer`, and a command written with a Windows
  interpreter path is not merely wrong on Linux, it is unrunnable: a POSIX
  shell reads the backslashes as escapes and looks for a file named
  `..venvScriptspython.exe`.

Neither prerequisite is visible in the workflow that fails, because the Ship
Gate reaches Playwright indirectly through `scripts/ship.py`. So the job scan
below follows that hop instead of grepping the YAML for "playwright" and
concluding the job is unrelated.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKFLOW_DIR = ROOT / ".github/workflows"
PLAYWRIGHT_CONFIG = ROOT / "frontend-react/playwright.config.ts"
SHIP_ENTRYPOINT = "scripts/ship.py"
SHIP_STEPS = ROOT / "scripts/ship_steps"

# Any of these mean "this command runs the Playwright suite". The repository is
# free to invoke it directly or through the npm script; both are valid, so the
# contract recognises both rather than pinning one spelling.
RUNS_PLAYWRIGHT = (
    re.compile(r"playwright\s+test"),
    re.compile(r"npm\s+run\s+test:e2e"),
    re.compile(r"""["']test:e2e["']"""),
)

# Likewise for provisioning: `npx playwright install`, `playwright install`,
# with or without `--with-deps`, with or without a named browser, all satisfy
# the requirement. What does not satisfy it is no install at all.
INSTALLS_BROWSER = re.compile(r"playwright\s+install\b")


def _workflows() -> list[tuple[str, dict]]:
    return [
        (path.name, yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    ]


def _ship_runs_playwright() -> bool:
    """Does `scripts/ship.py` reach Playwright through its step modules?"""
    for module in SHIP_STEPS.glob("*.py"):
        text = module.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in RUNS_PLAYWRIGHT):
            return True
    return False


def _jobs_that_reach_playwright():
    """Yield (workflow, job, steps) for every job that ends up running it.

    A job qualifies either by invoking Playwright in a `run` block or by
    invoking the ship entrypoint while that entrypoint runs Playwright.
    """
    ship_runs_it = _ship_runs_playwright()
    for filename, workflow in _workflows():
        for job_name, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            commands = [str(step.get("run") or "") for step in steps]
            direct = any(
                pattern.search(command) for command in commands for pattern in RUNS_PLAYWRIGHT
            )
            via_ship = ship_runs_it and any(SHIP_ENTRYPOINT in command for command in commands)
            if direct or via_ship:
                yield filename, job_name, steps


# --- the browser has to exist -----------------------------------------------


def test_the_scan_finds_at_least_one_playwright_job() -> None:
    """Guard against the contract quietly matching nothing and passing."""
    reached = list(_jobs_that_reach_playwright())
    assert reached, "no job was detected as running Playwright; the scan is broken"


def test_ship_entrypoint_is_known_to_run_playwright() -> None:
    """The indirection this contract depends on, asserted rather than assumed."""
    assert _ship_runs_playwright(), (
        "scripts/ship_steps no longer runs Playwright; the indirect detection "
        "above is now dead and the contract needs revisiting"
    )


def test_every_playwright_job_provisions_a_browser() -> None:
    offenders = []
    for filename, job_name, steps in _jobs_that_reach_playwright():
        if not any(INSTALLS_BROWSER.search(str(step.get("run") or "")) for step in steps):
            offenders.append(f"{filename}:{job_name}")

    assert not offenders, (
        "these jobs run Playwright without installing a browser first: "
        f"{offenders}. `npm ci` installs the test runner, not the browsers."
    )


def _command_lines(steps) -> list[str]:
    """Every `run` line in the job, flattened in execution order.

    Per-step granularity is not enough: a job may legitimately install the
    browser and use it inside one multi-line `run` block, which makes both land
    on the same step index while still being correctly ordered.
    """
    lines = []
    for step in steps:
        lines.extend(str(step.get("run") or "").splitlines())
    return lines


def test_the_browser_is_installed_before_it_is_used() -> None:
    """Ordering matters: an install after the run is an install that never ran."""
    for filename, job_name, steps in _jobs_that_reach_playwright():
        lines = _command_lines(steps)

        install_at = next(
            (i for i, line in enumerate(lines) if INSTALLS_BROWSER.search(line)),
            None,
        )
        use_at = next(
            (
                i
                for i, line in enumerate(lines)
                if any(p.search(line) for p in RUNS_PLAYWRIGHT) or SHIP_ENTRYPOINT in line
            ),
            None,
        )
        assert install_at is not None, f"{filename}:{job_name} installs no browser"
        assert use_at is not None, f"{filename}:{job_name} never uses Playwright"
        assert install_at < use_at, (
            f"{filename}:{job_name} installs the browser at line {install_at} "
            f"but uses it at line {use_at}"
        )


def test_browser_detection_rejects_a_job_with_no_install() -> None:
    """The detector must be capable of failing."""
    assert not INSTALLS_BROWSER.search("npm ci")
    assert not INSTALLS_BROWSER.search("uv run python scripts/ship.py")
    assert INSTALLS_BROWSER.search("npx playwright install chromium")
    assert INSTALLS_BROWSER.search("npx playwright install --with-deps chromium")
    assert INSTALLS_BROWSER.search("playwright install")


# --- the servers have to start on the runner's OS ---------------------------


def _web_server_commands() -> list[str]:
    """The `command:` strings inside the config's webServer block."""
    text = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    block = text[text.index("webServer") :]
    return re.findall(r"""command:\s*["'](.+?)["'],""", block)


def test_the_config_declares_web_server_commands() -> None:
    commands = _web_server_commands()
    assert commands, "no webServer commands found; the parser or config changed"


@pytest.mark.parametrize("needle", [".venv\\\\Scripts", ".venv/Scripts", "python.exe"])
def test_no_web_server_command_hardcodes_a_windows_interpreter(needle: str) -> None:
    """`.venv/Scripts/python.exe` exists only on Windows.

    On Linux the same virtualenv puts its interpreter in `.venv/bin`, so a
    hardcoded Windows path cannot resolve, and the gate fails before a single
    test runs.
    """
    offenders = [command for command in _web_server_commands() if needle in command]
    assert not offenders, (
        f"webServer command hardcodes the Windows-only {needle!r}: {offenders}. "
        "CI runs these on ubuntu-latest."
    )


def test_no_web_server_command_uses_backslash_path_separators() -> None:
    """A POSIX shell treats a backslash as an escape, not a separator.

    `.\\.venv\\Scripts\\python.exe` reaches /bin/sh as the single token
    `..venvScriptspython.exe`, which is why the failure is "command not found"
    rather than a Python error.
    """
    offenders = [command for command in _web_server_commands() if "\\" in command]
    assert not offenders, (
        f"webServer command uses backslash path separators: {offenders}. "
        "These do not survive a POSIX shell."
    )
