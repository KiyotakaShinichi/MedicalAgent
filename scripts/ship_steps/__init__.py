"""Ship-run step definitions, grouped by responsibility.

``scripts/ship.py`` owns sequencing, caching, manifests, and exit codes. This
package owns *what the steps are*. `all_steps()` concatenates the groups in
their original order, which is the ordering contract the ship run depends on.
"""

from __future__ import annotations

from scripts.ship_steps.assurance_and_release import assurance_and_release_steps
from scripts.ship_steps.backend_tests import backend_tests_steps
from scripts.ship_steps.common import FRONTEND, ROOT, Step, npm_cmd
from scripts.ship_steps.frontend import frontend_steps
from scripts.ship_steps.rag_and_data_platform import rag_and_data_platform_steps
from scripts.ship_steps.readiness_and_security import readiness_and_security_steps
from scripts.ship_steps.research_and_finetune import research_and_finetune_steps
from scripts.ship_steps.safety_and_observability import safety_and_observability_steps

# Order is the contract: the ship run executes these in sequence, and later
# steps consume evidence earlier ones produce.
STEP_GROUPS = (
    backend_tests_steps,
    frontend_steps,
    rag_and_data_platform_steps,
    readiness_and_security_steps,
    safety_and_observability_steps,
    research_and_finetune_steps,
    assurance_and_release_steps,
)


def all_steps() -> list[Step]:
    """Every ship step, in execution order."""
    steps: list[Step] = []
    for group in STEP_GROUPS:
        steps.extend(group())
    return steps


__all__ = [
    "FRONTEND",
    "ROOT",
    "STEP_GROUPS",
    "Step",
    "all_steps",
    "npm_cmd",
    *(g.__name__ for g in STEP_GROUPS),
]
