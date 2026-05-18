import { RefreshCw, ShieldAlert } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { Button } from "../../../../components/ui/Button";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { LoadingPane, EmptyPane } from "../../../../components/ui/Spinner";
import type { FailureModeRegistry } from "../../../../types/api";

/**
 * Phase 6b — Failure-mode registry.
 *
 * Presentational card only. The registry is meant to expose unresolved
 * risks honestly, so `needs_attention` can be the correct state.
 */
export function FailureModeRegistryCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: FailureModeRegistry | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const summary = report?.summary ?? {};
  const entries = report?.entries ?? [];
  const high = summary.by_severity?.high ?? 0;
  const unresolved = summary.entries_with_unresolved_gap ?? 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldAlert size={14} style={{ color: "var(--amber, #92400e)" }} aria-hidden="true" />
          <SectionTitle>Failure-mode registry</SectionTitle>
          <Badge variant={statusVariant(
            status === "strong" ? "strong" :
              status === "acceptable" ? "acceptable" :
                status === "missing" ? "stale" : "needs_attention",
          )}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Rebuilding..." : "Rebuild"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Consolidates engineering risks, failure case gallery entries, safety
        red-team failures, and drift findings into one auditable list. Each
        row carries category, severity, detection method, mitigation, and
        remaining gap.
      </p>

      {loading ? (
        <LoadingPane label="Loading failure-mode registry..." />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Failure-mode registry has not been built yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard label="Entries" value={report.entry_count ?? entries.length} status="muted" />
            <MetricCard
              label="High severity"
              value={high}
              status={high > 0 ? "amber" : "green"}
            />
            <MetricCard
              label="With unresolved gap"
              value={unresolved}
              status={unresolved > 0 ? "amber" : "green"}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  {["Name", "Category", "Severity", "Detection", "Remaining gap"].map((h) => (
                    <th key={h} className="text-left font-semibold py-1.5 px-2"
                      style={{ borderBottom: "1px solid var(--border)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {entries.slice(0, 15).map((e) => {
                  const severityColor = e.severity === "high"
                    ? "var(--rose, #b91c1c)"
                    : e.severity === "medium"
                      ? "var(--amber, #92400e)"
                      : "var(--text-dim)";
                  return (
                    <tr key={e.name}>
                      <td className="py-1.5 px-2 font-semibold" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                        {e.name}
                      </td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                        {e.category}
                      </td>
                      <td className="py-1.5 px-2 font-semibold uppercase" style={{ borderBottom: "1px solid var(--border-soft)", color: severityColor, fontSize: "0.68rem" }}>
                        {e.severity}
                      </td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                        {e.detection}
                      </td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: e.remaining_gap ? "var(--amber, #92400e)" : "var(--text-faint)" }}>
                        {e.remaining_gap ?? "-"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {entries.length > 15 && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Showing first 15 of {entries.length} entries.
            </p>
          )}
        </>
      )}
    </Card>
  );
}
