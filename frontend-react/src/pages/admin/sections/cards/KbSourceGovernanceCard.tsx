import { RefreshCw, Info } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { Button } from "../../../../components/ui/Button";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { LoadingPane, EmptyPane } from "../../../../components/ui/Spinner";
import type { KbSourceGovernanceReport } from "../../../../types/api";

/**
 * Phase 8b — KB source-governance summary.
 *
 * Per-source tier (T1–T5), allowed_use (education / patient_safety /
 * monitoring_context / portal_help / clinician_only), and staleness
 * (current / aging / needs_review).  Drives what claims each chunk is
 * allowed to back.
 */
export function KbSourceGovernanceCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: KbSourceGovernanceReport | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const tierDist = report?.tier_distribution ?? {};
  const useDist = report?.allowed_use_distribution ?? {};
  const staleDist = report?.staleness_distribution ?? {};
  const issues = report?.governance_issues ?? [];
  const sources = report?.sources ?? [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Info size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>KB source governance</SectionTitle>
          <Badge variant={statusVariant(
            status === "strong" ? "strong" :
            status === "acceptable" ? "acceptable" :
            status === "missing" ? "stale" :
            status === "error" ? "failed" : "needs_attention",
          )}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Rebuilding…" : "Rebuild"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Per-source <code>tier</code> (T1 guidelines → T5 unmapped),
        <code> allowed_use</code> (education / patient_safety / monitoring_context / portal_help / clinician_only),
        and <code>staleness</code> (current / aging / needs_review) for the
        RAG knowledge base. Drives what claims each chunk is allowed to back.
      </p>

      {loading ? (
        <LoadingPane label="Loading KB governance…" />
      ) : !report || status === "missing" || status === "error" ? (
        <EmptyPane label={report?.message ?? "KB governance has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard label="Sources" value={report.source_count ?? sources.length} status="muted" />
            <MetricCard label="Chunks" value={report.chunk_count ?? 0} status="muted" />
            <MetricCard
              label="Governance issues"
              value={issues.length}
              status={issues.length === 0 ? "green" : "amber"}
            />
          </div>

          <div className="grid sm:grid-cols-3 gap-3 mb-3 text-xs">
            <DistroBlock label="Tier distribution" data={tierDist} />
            <DistroBlock label="Allowed use" data={useDist} />
            <DistroBlock label="Staleness" data={staleDist} />
          </div>

          {issues.length > 0 && (
            <div
              className="rounded-md border p-3 mb-3"
              style={{ background: "rgba(245,158,11,0.06)", borderColor: "rgba(245,158,11,0.25)" }}
            >
              <p className="text-[0.72rem] uppercase font-semibold mb-1.5" style={{ color: "var(--amber, #92400e)" }}>
                Governance issues ({issues.length})
              </p>
              <ul className="flex flex-col gap-1.5 text-xs" style={{ color: "var(--text)" }}>
                {issues.map((i, idx) => (
                  <li key={idx}>
                    <code style={{ fontWeight: 600 }}>{i.code}</code> ({i.severity}) — {i.message}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  {["Source", "Tier", "Allowed use", "Staleness", "Chunks"].map((h) => (
                    <th key={h} className="text-left font-semibold py-1.5 px-2"
                        style={{ borderBottom: "1px solid var(--border)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sources.slice(0, 12).map((s) => (
                  <tr key={s.source_id}>
                    <td className="py-1.5 px-2 font-semibold" style={{ borderBottom: "1px solid var(--border-soft)", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {s.title}
                    </td>
                    <td className="py-1.5 px-2 font-semibold" style={{ borderBottom: "1px solid var(--border-soft)", color: tierColor(s.tier) }}>
                      {s.tier}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                      {s.allowed_use.length === 0
                        ? <span style={{ color: "var(--rose)" }}>none</span>
                        : s.allowed_use.join(", ")}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: stalenessColor(s.staleness_status) }}>
                      {s.staleness_status}
                    </td>
                    <td className="py-1.5 px-2 tabular-nums" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                      {s.chunk_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {sources.length > 12 && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Showing first 12 of {sources.length} sources.
            </p>
          )}
        </>
      )}
    </Card>
  );
}

function DistroBlock({ label, data }: { label: string; data: Record<string, number> }) {
  const entries = Object.entries(data);
  return (
    <div style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, padding: 8 }}>
      <p className="text-[0.7rem] uppercase font-semibold mb-1" style={{ color: "var(--text-faint)" }}>{label}</p>
      {entries.length === 0 ? (
        <span style={{ color: "var(--text-faint)", fontSize: "0.74rem" }}>—</span>
      ) : (
        <ul className="flex flex-col gap-0.5">
          {entries.map(([k, v]) => (
            <li key={k} className="flex justify-between" style={{ fontSize: "0.74rem" }}>
              <span style={{ color: "var(--text-dim)" }}>{k}</span>
              <span className="tabular-nums" style={{ color: "var(--text)", fontWeight: 600 }}>{v}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function tierColor(tier: string): string {
  switch (tier) {
    case "T1": return "var(--green, #047857)";
    case "T2": return "var(--blue, #1e3a8a)";
    case "T3": return "var(--text)";
    case "T4": return "var(--text-dim)";
    case "T5": return "var(--rose, #b91c1c)";
    default:   return "var(--text-faint)";
  }
}

function stalenessColor(status: string): string {
  switch (status) {
    case "current":      return "var(--green, #047857)";
    case "aging":        return "var(--amber, #92400e)";
    case "needs_review": return "var(--rose, #b91c1c)";
    default:             return "var(--text-faint)";
  }
}
