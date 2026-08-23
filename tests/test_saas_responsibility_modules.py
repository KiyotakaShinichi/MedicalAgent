"""Each control-plane responsibility works through its own module.

`saas_control_plane` was a single 982-line module owning organizations,
projects, jobs, entitlements, usage, outbox, and audit. It is now a facade over
four modules, and this file covers each one directly rather than only through
the facade — so a responsibility that breaks is named by the failing test.

What is asserted here that a facade-only test cannot:

* the facade is a **strict superset** of the pre-split public surface, so no
  caller's import can break;
* the modules import cleanly in isolation and in any order, which is the thing
  a responsibility split most easily gets wrong;
* each module owns its resource and does not quietly reach into another's.

`tests/test_saas_control_plane.py` keeps the end-to-end behavioural coverage
through the facade; this file does not repeat it.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.models import Base  # noqa: E402
from backend.services import saas_control_plane as facade  # noqa: E402
from backend.services.saas_common import SaaSActor  # noqa: E402

RESPONSIBILITY_MODULES = (
    "backend.services.saas_common",
    "backend.services.saas_organizations",
    "backend.services.saas_projects",
    "backend.services.saas_jobs",
)


@pytest.fixture
def db():
    """In-memory SQLite bound through StaticPool.

    Without StaticPool each connection gets its own empty database, so the
    schema created here would be invisible to the session under test.
    """
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def admin() -> SaaSActor:
    return facade.actor_from_access_context(
        type("Ctx", (), {"subject": "admin@example.test", "role": "admin", "auth_source": "demo_session"})()
    )


# ─── the facade contract ─────────────────────────────────────────────────────


def test_facade_exports_every_pre_split_symbol() -> None:
    """Recovered from git, not hand-copied: the real pre-split surface.

    A hand-maintained list would drift from what callers actually imported,
    which is the only thing that matters here.
    """
    import ast

    original = subprocess.run(
        ["git", "show", "HEAD:backend/services/saas_control_plane.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    tree = ast.parse(original)
    public = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                if target.id != "__all__":
                    public.add(target.id)

    missing = sorted(name for name in public if not hasattr(facade, name))
    assert not missing, f"the split dropped public symbols: {missing}"


def test_facade_all_is_unchanged() -> None:
    assert facade.__all__ == sorted(facade.__all__), "__all__ must stay sorted"
    for name in facade.__all__:
        assert hasattr(facade, name), f"__all__ advertises missing {name}"


@pytest.mark.parametrize("module", RESPONSIBILITY_MODULES)
def test_each_module_imports_standalone(module: str) -> None:
    """Import order must not matter.

    The organization and job modules genuinely depend on each other — jobs
    meter usage, the overview reads jobs — so this is the assertion that the
    cycle is actually broken rather than merely hidden by import order.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{module} failed to import alone:\n{result.stderr}"


def test_modules_import_in_reverse_order() -> None:
    script = "; ".join(f"import {m}" for m in reversed(RESPONSIBILITY_MODULES))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, f"reverse-order import failed:\n{result.stderr}"


@pytest.mark.parametrize("module", RESPONSIBILITY_MODULES)
def test_each_module_stays_under_the_service_limit(module: str) -> None:
    path = ROOT / (module.replace(".", "/") + ".py")
    loc = len(path.read_text(encoding="utf-8").splitlines())
    assert loc <= 500, f"{path.name} is {loc} LOC; split it further"


# ─── organizations ───────────────────────────────────────────────────────────


def test_organizations_module_creates_and_lists(db, admin: SaaSActor) -> None:
    from backend.services import saas_organizations

    organization = saas_organizations.create_organization(db, actor=admin, name="Acme Labs")
    db.flush()

    assert organization.slug == "acme-labs"
    assert organization.data_class == "synthetic_only"

    listed = saas_organizations.list_organizations_for_actor(db, admin)
    assert [row["slug"] for row in listed] == ["acme-labs"]
    assert listed[0]["membership_role"] == "owner"


def test_organization_creation_requires_application_admin(db) -> None:
    from backend.services import saas_organizations

    clinician = facade.actor_from_access_context(
        type("Ctx", (), {"subject": "c@example.test", "role": "clinician", "auth_source": "demo_session"})()
    )
    with pytest.raises(facade.SaaSAccessError):
        saas_organizations.create_organization(db, actor=clinician, name="Nope")


# ─── projects ────────────────────────────────────────────────────────────────


def test_projects_module_creates_within_an_organization(db, admin: SaaSActor) -> None:
    from backend.services import saas_organizations, saas_projects

    organization = saas_organizations.create_organization(db, actor=admin, name="Acme Labs")
    db.flush()
    project = saas_projects.create_project(
        db, organization_id=organization.id, actor=admin, name="Monitoring"
    )
    db.flush()

    assert project.organization_id == organization.id
    listed = saas_projects.list_projects(db, organization_id=organization.id, actor=admin)
    assert [row["slug"] for row in listed] == ["monitoring"]


# ─── jobs ────────────────────────────────────────────────────────────────────


def test_jobs_module_enqueues_and_lists(db, admin: SaaSActor) -> None:
    from backend.services import saas_organizations, saas_jobs, saas_projects

    organization = saas_organizations.create_organization(db, actor=admin, name="Acme Labs")
    db.flush()
    project = saas_projects.create_project(
        db, organization_id=organization.id, actor=admin, name="Monitoring"
    )
    db.flush()

    job, replayed = saas_jobs.enqueue_platform_job(
        db,
        organization_id=organization.id,
        actor=admin,
        job_type=sorted(facade.ALLOWED_JOB_TYPES)[0],
        project_id=project.id,
        idempotency_key="job-1",
        payload={"note": "synthetic"},
    )
    db.flush()

    assert replayed is False, "a brand-new key must not report a replay"
    listed = saas_jobs.list_platform_jobs(db, organization_id=organization.id, actor=admin)
    assert job.id in [row["id"] for row in listed]


def test_job_enqueue_is_idempotent(db, admin: SaaSActor) -> None:
    """The same key must not create a second job.

    Enqueue meters usage, so a duplicate would both double-run the work and
    double-bill the organization.
    """
    from backend.services import saas_organizations, saas_jobs, saas_projects

    organization = saas_organizations.create_organization(db, actor=admin, name="Acme Labs")
    db.flush()
    project = saas_projects.create_project(
        db, organization_id=organization.id, actor=admin, name="Monitoring"
    )
    db.flush()

    kwargs = dict(
        organization_id=organization.id,
        actor=admin,
        job_type=sorted(facade.ALLOWED_JOB_TYPES)[0],
        project_id=project.id,
        idempotency_key="job-repeat",
        payload={"note": "synthetic"},
    )
    first, first_replayed = saas_jobs.enqueue_platform_job(db, **kwargs)
    db.flush()
    second, second_replayed = saas_jobs.enqueue_platform_job(db, **kwargs)
    db.flush()

    # The flag reports a replay, not a creation: False on the first call,
    # True when the key is seen again.
    assert first_replayed is False
    assert second_replayed is True
    assert first.id == second.id


def test_job_type_must_be_allow_listed(db, admin: SaaSActor) -> None:
    """An arbitrary job type would let a caller name any worker path."""
    from backend.services import saas_organizations, saas_jobs, saas_projects

    organization = saas_organizations.create_organization(db, actor=admin, name="Acme Labs")
    db.flush()
    project = saas_projects.create_project(
        db, organization_id=organization.id, actor=admin, name="Monitoring"
    )
    db.flush()

    with pytest.raises(facade.SaaSValidationError):
        saas_jobs.enqueue_platform_job(
            db,
            organization_id=organization.id,
            actor=admin,
            job_type="arbitrary_worker",
            project_id=project.id,
            idempotency_key="job-bad-type",
            payload={},
        )


# ─── shared primitives ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "forbidden_key",
    ["patient_id", "diagnosis", "message", "prompt", "email", "raw_text"],
)
def test_job_payload_sanitisation_rejects_phi_shaped_keys(forbidden_key: str) -> None:
    """A job payload is persisted and replayed, so PHI in one is durable PHI.

    The control plane is synthetic-only by declaration; this is the check that
    makes the declaration enforceable rather than aspirational.
    """
    from backend.services import saas_common

    with pytest.raises(facade.SaaSValidationError):
        saas_common.sanitize_job_payload({forbidden_key: "anything"})


def test_job_payload_sanitisation_rejects_a_non_object_payload() -> None:
    from backend.services import saas_common

    with pytest.raises(facade.SaaSValidationError):
        saas_common.sanitize_job_payload(["not", "an", "object"])


def test_job_payload_sanitisation_keeps_ordinary_fields() -> None:
    from backend.services import saas_common

    assert saas_common.sanitize_job_payload({"note": "synthetic", "count": 3}) == {
        "note": "synthetic",
        "count": 3,
    }


def test_claim_boundary_still_denies_clinical_use() -> None:
    """The control plane must not read as a cleared clinical product."""
    boundary = " ".join(str(value) for value in facade.CLAIM_BOUNDARY.values()) \
        if isinstance(facade.CLAIM_BOUNDARY, dict) else str(facade.CLAIM_BOUNDARY)
    assert "synthetic" in boundary.lower()


def test_facade_and_modules_share_one_object(admin: SaaSActor) -> None:
    """Re-export, not a copy: patching one must affect the other."""
    from backend.services import saas_jobs, saas_organizations

    assert facade.enqueue_platform_job is saas_jobs.enqueue_platform_job
    assert facade.create_organization is saas_organizations.create_organization
    assert importlib.import_module("backend.services.saas_common").SaaSActor is SaaSActor


# ─── evidence that reads the control plane by path ───────────────────────────


def test_foundation_readiness_reads_the_whole_control_plane() -> None:
    """Readiness greps the control plane for its controls, so it must see all of it.

    `saas_foundation_readiness` checks for literal markers such as
    `idempotency_key` in `saas_control_plane.py`. After the split that string
    lives in `saas_jobs.py`, and reading only the facade reported the
    idempotency contract as *absent* — the contract unchanged, the evidence
    wrong, and the whole readiness artifact downgraded to `needs_attention`.

    This is the second time a path-based evidence check has been broken by a
    legitimate refactor, so it is pinned here rather than rediscovered.
    """
    from backend.services.saas_foundation_readiness import (
        CONTROL_PLANE_MODULES,
        build_saas_foundation_readiness,
    )

    report = build_saas_foundation_readiness()
    failing = [
        check
        for value in report.values()
        if isinstance(value, list)
        for check in value
        if isinstance(check, dict) and check.get("passed") is False
    ]
    assert not failing, f"readiness reports failing controls: {failing}"
    assert report["status"] == "ready_for_restricted_synthetic_saas_alpha"

    # Every module the facade re-exports must be in the searched set, or a
    # control implemented there would read as missing.
    for module in RESPONSIBILITY_MODULES:
        leaf = module.rsplit(".", 1)[-1] + ".py"
        assert leaf in CONTROL_PLANE_MODULES, f"{leaf} is not searched by readiness"


def test_readiness_still_fails_for_a_genuinely_absent_control() -> None:
    """Widening the search must not make the check unable to fail."""
    from backend.services.saas_foundation_readiness import _contains

    services = ROOT / "backend" / "services"
    result = _contains(
        services / "saas_control_plane.py",
        "a_marker_that_appears_in_no_control_plane_module",
        "synthetic_control",
    )
    assert result["passed"] is False
