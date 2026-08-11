import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import {
  Activity,
  ArrowRight,
  Boxes,
  Database,
  FileCheck2,
  FolderKanban,
  Gauge,
  LayoutDashboard,
  PlayCircle,
  Plus,
  ShieldCheck,
  Workflow,
} from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Button } from "../../components/ui/Button";
import { ErrorPane, LoadingPane } from "../../components/ui/Spinner";
import {
  createPlatformOrganization,
  createWorkspaceJob,
  createWorkspaceProject,
  getPlatformSession,
  getWorkspaceOverview,
} from "../../api/client";
import type {
  SaaSOrganization,
  SaaSPlatformSession,
  SaaSWorkspaceOverview,
} from "../../types/api";

const NAV = [
  { to: "/workspace", label: "Workspace", icon: LayoutDashboard },
  { to: "/workspace#projects", label: "Projects", icon: FolderKanban },
  { to: "/workspace#runs", label: "Evaluation runs", icon: PlayCircle },
  { to: "/workspace#usage", label: "Usage & quotas", icon: Gauge },
  { to: "/workspace#automation", label: "Automation", icon: Workflow },
  { to: "/workspace#governance", label: "Governance", icon: ShieldCheck },
  { to: "/admin", label: "Evaluation demo", icon: Boxes },
];

const JOB_LABELS: Record<string, string> = {
  rag_baseline_comparison: "RAG baseline comparison",
  adversarial_safety_eval: "Adversarial safety evaluation",
  agent_workflow_eval: "Agent workflow evaluation",
  release_gate: "Release gate",
  evidence_packet_export: "Evidence packet export",
};

export default function WorkspaceDashboard() {
  const [session, setSession] = useState<SaaSPlatformSession | null>(null);
  const [organizationId, setOrganizationId] = useState<string | null>(null);
  const [overview, setOverview] = useState<SaaSWorkspaceOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [projectName, setProjectName] = useState("");
  const [organizationName, setOrganizationName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [jobType, setJobType] = useState("rag_baseline_comparison");

  const loadOverview = useCallback(async (preferredOrganizationId: string) => {
    setLoading(true);
    setError(null);
    try {
      const nextSession = await getPlatformSession();
      setSession(nextSession);
      setOrganizationId(preferredOrganizationId);
      setOverview(await getWorkspaceOverview(preferredOrganizationId));
    } catch (nextError) {
      setError((nextError as Error).message || "Workspace failed to load.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    getPlatformSession()
      .then(async (nextSession) => {
        const selected = nextSession.organizations[0]?.id ?? null;
        const nextOverview = selected ? await getWorkspaceOverview(selected) : null;
        return { nextSession, selected, nextOverview };
      })
      .then(({ nextSession, selected, nextOverview }) => {
        if (!active) return;
        setSession(nextSession);
        setOrganizationId(selected);
        setOverview(nextOverview);
      })
      .catch((nextError) => {
        if (active) setError((nextError as Error).message || "Workspace failed to load.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, []);

  const organization = useMemo(
    () => session?.organizations.find((item) => item.id === organizationId) ?? null,
    [organizationId, session],
  );

  async function handleOrganizationCreate(event: FormEvent) {
    event.preventDefault();
    if (!organizationName.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const created = await createPlatformOrganization({ name: organizationName.trim() });
      setOrganizationName("");
      await loadOverview(created.id);
    } catch (nextError) {
      setError((nextError as Error).message || "Organization could not be created.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleProjectCreate(event: FormEvent) {
    event.preventDefault();
    if (!organizationId || !projectName.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await createWorkspaceProject(organizationId, {
        name: projectName.trim(),
        description: "Synthetic AI evaluation project; no real patient data.",
      });
      setProjectName("");
      await loadOverview(organizationId);
    } catch (nextError) {
      setError((nextError as Error).message || "Project could not be created.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRunEvaluation() {
    const project = overview?.projects[0];
    if (!organizationId || !project) return;
    setSubmitting(true);
    setError(null);
    try {
      const key = `workspace-${jobType}-${crypto.randomUUID()}`;
      await createWorkspaceJob(
        organizationId,
        project.id,
        {
          job_type: jobType,
          environment_id: project.environments[0]?.id,
          payload: {
            suite_ref: "repository-default",
            config_ref: "synthetic-staging",
            dry_run: true,
          },
        },
        key,
      );
      await loadOverview(organizationId);
    } catch (nextError) {
      setError((nextError as Error).message || "Evaluation could not be queued.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AppShell
      navItems={NAV}
      title="AI Assurance Workspace"
      subtitle="Synthetic evaluation, governance, and release evidence"
    >
      <div className="workspace-page">
        <div className="workspace-boundary">
          <ShieldCheck size={18} aria-hidden="true" />
          <div>
            <strong>Nonclinical engineering workspace</strong>
            <span>No real patient data, clinical validation, billing, or healthcare deployment authority.</span>
          </div>
        </div>

        {loading && !overview && <LoadingPane label="Loading organization workspace..." />}
        {error && <ErrorPane message={error} />}

        {!loading && session && session.organizations.length === 0 && (
          <section className="workspace-empty">
            <Database size={28} aria-hidden="true" />
            <h2>Create your first synthetic assurance organization</h2>
            <p>Organizations are isolation boundaries for projects, jobs, usage, artifacts, and audit events.</p>
            <form onSubmit={handleOrganizationCreate} className="workspace-inline-form">
              <input
                value={organizationName}
                onChange={(event) => setOrganizationName(event.target.value)}
                placeholder="Organization name"
                minLength={3}
                maxLength={120}
                required
              />
              <Button type="submit" loading={submitting} icon={<Plus size={16} />}>Create organization</Button>
            </form>
          </section>
        )}

        {overview && organization && (
          <>
            <header className="workspace-heading">
              <div>
                <span className="workspace-eyebrow">{organization.plan_code.replaceAll("_", " ")}</span>
                <h2>{organization.name}</h2>
                <p>Tenant-scoped control plane for repeatable AI evaluation and governance evidence.</p>
              </div>
              <label className="workspace-switcher">
                <span>Organization</span>
                <select
                  value={organizationId ?? ""}
                  onChange={(event) => void loadOverview(event.target.value)}
                >
                  {session?.organizations.map((item: SaaSOrganization) => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))}
                </select>
              </label>
            </header>

            <section className="workspace-kpis" aria-label="Workspace summary">
              <WorkspaceKpi label="Projects" value={overview.projects.length} detail="Synthetic-only workspaces" icon={FolderKanban} />
              <WorkspaceKpi label="Evaluation runs" value={overview.recent_jobs.length} detail="Durable, idempotent queue" icon={PlayCircle} />
              <WorkspaceKpi label="Audit events" value={overview.audit_event_count} detail="Database-backed trail" icon={FileCheck2} />
              <WorkspaceKpi label="Pending events" value={overview.pending_outbox_event_count} detail="Automation outbox" icon={Workflow} />
            </section>

            <section className="workspace-layout" id="projects">
              <article className="workspace-panel workspace-project-panel">
                <div className="workspace-panel-heading">
                  <div><span>Projects</span><h3>Evaluation environments</h3></div>
                  <strong>{overview.projects.length}</strong>
                </div>
                <div className="workspace-project-list">
                  {overview.projects.map((project) => (
                    <div className="workspace-project" key={project.id}>
                      <span className="workspace-project-icon"><FolderKanban size={18} /></span>
                      <div>
                        <strong>{project.name}</strong>
                        <p>{project.description}</p>
                        <small>{project.environments[0]?.name ?? "No environment"} - {project.environments[0]?.retrieval_profile ?? "unconfigured"}</small>
                      </div>
                      <span className="workspace-data-badge">synthetic</span>
                    </div>
                  ))}
                </div>
                <form onSubmit={handleProjectCreate} className="workspace-inline-form">
                  <input
                    value={projectName}
                    onChange={(event) => setProjectName(event.target.value)}
                    placeholder="New project name"
                    minLength={3}
                    maxLength={120}
                    required
                  />
                  <Button type="submit" variant="secondary" loading={submitting} icon={<Plus size={16} />}>Add project</Button>
                </form>
              </article>

              <article className="workspace-panel" id="runs">
                <div className="workspace-panel-heading">
                  <div><span>Run center</span><h3>Queue an evaluation</h3></div>
                  <Activity size={20} />
                </div>
                <label className="workspace-field">
                  <span>Evaluation recipe</span>
                  <select value={jobType} onChange={(event) => setJobType(event.target.value)}>
                    {Object.entries(JOB_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                </label>
                <p className="workspace-help">The control plane stores references only and rejects patient, prompt, message, and identity payload fields.</p>
                <Button
                  onClick={handleRunEvaluation}
                  loading={submitting}
                  disabled={!overview.projects.length}
                  icon={<PlayCircle size={17} />}
                >Queue dry-run evaluation</Button>
              </article>
            </section>

            <section className="workspace-layout" id="usage">
              <article className="workspace-panel workspace-usage-panel">
                <div className="workspace-panel-heading">
                  <div><span>Usage and quotas</span><h3>Engineering capacity ledger</h3></div>
                  <Gauge size={20} />
                </div>
                <div className="workspace-usage-list">
                  {overview.usage.map((metric) => (
                    <div className="workspace-usage-row" key={metric.metric_key}>
                      <div><strong>{formatMetric(metric.metric_key)}</strong><span>{formatAmount(metric.used)} / {formatAmount(metric.hard_limit)} {metric.unit}</span></div>
                      <div className="workspace-meter"><span style={{ width: `${Math.min(100, metric.utilization * 100)}%` }} /></div>
                    </div>
                  ))}
                </div>
                <p className="workspace-help">This ledger enforces preview quotas. It is not an invoice or audited billing source.</p>
              </article>

              <article className="workspace-panel" id="governance">
                <div className="workspace-panel-heading">
                  <div><span>Governance</span><h3>Hard product boundaries</h3></div>
                  <ShieldCheck size={20} />
                </div>
                <ul className="workspace-check-list">
                  <li><ShieldCheck size={16} /> Tenant-scoped projects, jobs, usage, outbox, and audit events</li>
                  <li><ShieldCheck size={16} /> Synthetic-only data class locked on preview resources</li>
                  <li><ShieldCheck size={16} /> Provider usage is not accepted as billing truth without reconciliation</li>
                  <li><ShieldCheck size={16} /> Clinical demo remains separate from the SaaS control plane</li>
                </ul>
                <a className="workspace-demo-link" href="/admin">
                  Open evaluation demo <ArrowRight size={16} />
                </a>
              </article>
            </section>

            <section className="workspace-panel" id="automation">
              <div className="workspace-panel-heading">
                <div><span>Recent runs</span><h3>Durable job activity</h3></div>
                <Workflow size={20} />
              </div>
              {overview.recent_jobs.length === 0 ? (
                <p className="workspace-help">No evaluation jobs have been queued for this organization.</p>
              ) : (
                <div className="workspace-job-table">
                  {overview.recent_jobs.slice(0, 8).map((job) => (
                    <div key={job.id}>
                      <span className={`workspace-job-status is-${job.status}`}>{job.status}</span>
                      <strong>{JOB_LABELS[job.job_type] ?? job.job_type}</strong>
                      <span title={job.error_message ?? undefined}>{job.queued_at ? new Date(job.queued_at).toLocaleString() : "Pending timestamp"}</span>
                      <small>{job.id}{job.recovery_count ? ` - recovered ${job.recovery_count}x` : ""}</small>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </AppShell>
  );
}

function WorkspaceKpi({ label, value, detail, icon: Icon }: { label: string; value: number; detail: string; icon: typeof FolderKanban }) {
  return (
    <article className="workspace-kpi">
      <div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
      <span className="workspace-kpi-icon"><Icon size={18} /></span>
    </article>
  );
}

function formatMetric(value: string) {
  return value.split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function formatAmount(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0, notation: value >= 100_000 ? "compact" : "standard" }).format(value);
}
