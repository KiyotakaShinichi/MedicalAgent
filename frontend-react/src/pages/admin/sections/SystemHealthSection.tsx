import { RefreshCw } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { Button } from "../../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { ErrorPane, LoadingPane } from "../../../components/ui/Spinner";
import { useApi } from "../../../hooks/useApi";
import { getSystemHealth, runSystemHealth } from "../../../api/client";


export function SystemHealthSection() {
  const { data, status, error, refetch } = useApi(getSystemHealth, []);

  async function rebuild() {
    await runSystemHealth();
    await refetch();
  }

  if (status === "loading") return <LoadingPane label="Checking system health..." />;
  if (status === "error") return <ErrorPane message={error ?? "Could not load system health"} onRetry={() => void refetch()} />;
  if (!data) return null;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <SectionTitle>System Health</SectionTitle>
          <Button variant="secondary" size="sm" icon={<RefreshCw size={12} />} onClick={() => void rebuild()}>
            Re-check
          </Button>
        </CardHeader>
        <div className="grid sm:grid-cols-4 gap-3">
          <HealthTile label="Overall" value={data.status} />
          <HealthTile label="Database" value={data.backend.database.status} />
          <HealthTile label="Frontend build" value={data.frontend.production_build_present ? "present" : "missing"} />
          <HealthTile label="Groq" value={data.environment.groq_configured ? "configured" : "not configured"} />
        </div>
        <p className="text-xs mt-3" style={{ color: "var(--text-dim)" }}>{data.claim_boundary}</p>
      </Card>

      <Card>
        <CardHeader><SectionTitle>Issues & Next Actions</SectionTitle></CardHeader>
        {data.issues.length === 0 ? (
          <p className="text-sm" style={{ color: "var(--green)" }}>No blocking health issues detected.</p>
        ) : (
          <div className="flex flex-col gap-2">
            {data.issues.map((issue, index) => (
              <div key={`${issue.area}-${index}`} className="flex items-start justify-between gap-3 p-2 rounded-md" style={{ background: "var(--surface2)" }}>
                <span className="text-sm">{issue.message}</span>
                <Badge variant={issue.severity === "critical" ? "red" : issue.severity === "warning" ? "amber" : "muted"}>{issue.severity}</Badge>
              </div>
            ))}
          </div>
        )}
        <ul className="mt-3 text-xs" style={{ color: "var(--text-dim)", paddingLeft: 18 }}>
          {data.next_actions.map((action) => <li key={action}>{action}</li>)}
        </ul>
      </Card>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader><SectionTitle>Dependencies</SectionTitle></CardHeader>
          <div className="flex flex-col gap-2">
            {data.dependencies.map((dep) => (
              <div key={dep.package} className="flex items-center justify-between gap-3 text-sm">
                <div>
                  <strong>{dep.package}</strong>
                  <div className="text-xs" style={{ color: "var(--text-faint)" }}>{dep.purpose}</div>
                </div>
                <Badge variant={dep.available ? "green" : "amber"}>{dep.available ? "available" : "missing"}</Badge>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <CardHeader><SectionTitle>Artifacts</SectionTitle></CardHeader>
          <div className="flex flex-col gap-2">
            {data.artifacts.map((artifact) => (
              <div key={artifact.name} className="flex items-center justify-between gap-3 text-sm">
                <div style={{ minWidth: 0 }}>
                  <strong>{artifact.name.replace(/_/g, " ")}</strong>
                  <div className="text-xs truncate" style={{ color: "var(--text-faint)" }}>{artifact.path}</div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={artifact.exists ? statusVariant(artifact.status) : "amber"}>{artifact.status}</Badge>
                  <Badge variant={artifact.freshness === "fresh" ? "green" : artifact.freshness === "stale" ? "amber" : "muted"}>
                    {artifact.freshness}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function HealthTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-lg border" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
      <div className="text-xs mb-1" style={{ color: "var(--text-faint)" }}>{label}</div>
      <Badge variant={statusVariant(value)}>{value.replace(/_/g, " ")}</Badge>
    </div>
  );
}
