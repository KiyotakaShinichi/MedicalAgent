export interface SaaSOrganization {
  id: string;
  slug: string;
  name: string;
  status: string;
  plan_code: string;
  data_class: "synthetic_only";
  membership_role: "owner" | "admin" | "evaluator" | "viewer";
  created_at: string | null;
}

export interface SaaSEnvironment {
  id: string;
  key: string;
  name: string;
  status: string;
  retrieval_profile: string;
  data_class: "synthetic_only";
}

export interface SaaSProject {
  id: string;
  organization_id: string;
  slug: string;
  name: string;
  description: string | null;
  status: string;
  data_class: "synthetic_only";
  created_at: string | null;
  environments: SaaSEnvironment[];
}

export interface SaaSUsageMetric {
  metric_key: string;
  unit: string;
  used: number;
  soft_limit: number | null;
  hard_limit: number;
  remaining: number;
  utilization: number;
  period: string;
  billing_authoritative: false;
}

export interface SaaSPlatformJob {
  id: string;
  organization_id: string;
  project_id: string;
  environment_id: string | null;
  job_type: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | string;
  error_message: string | null;
  progress_percent: number;
  attempts: number;
  max_attempts: number;
  payload: Record<string, unknown>;
  queued_at: string | null;
  available_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  cancelled_at: string | null;
  recovery_count: number;
  billing_authoritative: false;
  clinical_validation: false;
}

export interface SaaSPlatformSession {
  actor: {
    subject: string;
    application_role: string;
    auth_source: string;
  };
  organizations: SaaSOrganization[];
  synthetic_only: true;
  clinical_validation: false;
  healthcare_production_ready: false;
  billing_enabled: false;
  claim_boundary: string;
}

export interface SaaSWorkspaceOverview {
  schema_version: string;
  organization: Omit<SaaSOrganization, "membership_role">;
  membership_role: SaaSOrganization["membership_role"];
  projects: SaaSProject[];
  recent_jobs: SaaSPlatformJob[];
  usage: SaaSUsageMetric[];
  audit_event_count: number;
  pending_outbox_event_count: number;
  synthetic_only: true;
  clinical_validation: false;
  healthcare_production_ready: false;
  billing_enabled: false;
  claim_boundary: string;
}
