"""Tenant-scoped control plane for NLCare's synthetic AI assurance workspace.

This module owns organization, project, environment, entitlement, usage,
durable-job, outbox, and audit boundaries. It does not make the legacy patient
demo multi-tenant and it does not authorize real patient data.

The implementation is split by responsibility:

* :mod:`backend.services.saas_common` - identity, membership, entitlements,
  payload sanitisation, outbox, and audit primitives shared by all three;
* :mod:`backend.services.saas_organizations` - organization lifecycle, usage
  metering, and the workspace overview;
* :mod:`backend.services.saas_projects` - project creation and listing;
* :mod:`backend.services.saas_jobs` - platform job enqueue, listing, cancel.

This module remains the public import surface. Every name it exported before
the split is re-exported here unchanged, so
``from backend.services.saas_control_plane import ...`` keeps working for the
API routers, the job worker, the readiness evaluations, and the tests.
"""

from __future__ import annotations

# `x as x` marks a deliberate re-export. Those names were module attributes
# of this module before the split and are kept so the facade stays a strict
# superset of the original surface, even though no caller reads them today.

from backend.services.saas_common import (
    ALLOWED_JOB_TYPES,
    CLAIM_BOUNDARY,
    DEFAULT_ENTITLEMENTS as DEFAULT_ENTITLEMENTS,
    FORBIDDEN_PAYLOAD_KEY_PARTS as FORBIDDEN_PAYLOAD_KEY_PARTS,
    MEMBERSHIP_ROLES,
    RUN_ROLES,
    WRITE_ROLES as WRITE_ROLES,
    SaaSAccessError,
    SaaSActor,
    SaaSQuotaExceeded,
    SaaSValidationError,
    actor_from_access_context,
    append_audit_event,
    append_outbox_event,
    entitlement_status,
    require_membership,
    sanitize_job_payload,
)
from backend.services.saas_jobs import (
    cancel_platform_job,
    enqueue_platform_job,
    list_platform_jobs,
    serialize_job,
)
from backend.services.saas_organizations import (
    bootstrap_demo_workspace,
    create_organization,
    list_organizations_for_actor,
    record_usage_event,
    serialize_organization as serialize_organization,
    usage_summary,
    workspace_overview,
)
from backend.services.saas_projects import (
    create_project,
    list_projects,
    serialize_project as serialize_project,
)

__all__ = [
    "ALLOWED_JOB_TYPES",
    "CLAIM_BOUNDARY",
    "MEMBERSHIP_ROLES",
    "RUN_ROLES",
    "SaaSAccessError",
    "SaaSActor",
    "SaaSQuotaExceeded",
    "SaaSValidationError",
    "actor_from_access_context",
    "append_audit_event",
    "append_outbox_event",
    "bootstrap_demo_workspace",
    "cancel_platform_job",
    "create_organization",
    "create_project",
    "enqueue_platform_job",
    "entitlement_status",
    "list_organizations_for_actor",
    "list_platform_jobs",
    "list_projects",
    "record_usage_event",
    "require_membership",
    "sanitize_job_payload",
    "serialize_job",
    "usage_summary",
    "workspace_overview",
]
