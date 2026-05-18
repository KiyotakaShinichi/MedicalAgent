import { RefreshCw, ShieldCheck } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { Button } from "../../../../components/ui/Button";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { LoadingPane, EmptyPane } from "../../../../components/ui/Spinner";
import type { EvidenceAbstentionEvalReport } from "../../../../types/api";

/**
 * Phase 2 — Evidence-aware abstention eval.
 *
 * Extracted from `MleSection.tsx` so the MLE dashboard keeps moving away
 * from one giant file. The parent owns fetching/rerun state; this card is
 * presentational only.
 */
export function EvidenceAbstentionCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: EvidenceAbstentionEvalReport | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const summary = report?.summary ?? {};
  const scenarios = report?.scenarios ?? [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldCheck size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>Evidence-aware abstention eval</SectionTitle>
          <Badge variant={statusVariant(status === "strong" ? "strong" : status === "acceptable" ? "acceptable" : status === "missing" ? "stale" : "needs_attention")}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Running..." : "Rerun eval"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Sweeps modality-dropout scenarios over the test rows. Coverage is the
        fraction the system chose to score; the rest were routed to clinician
        review with <code>insufficient_evidence</code>. False-abstention rate
        flags rows where we refused but the underlying model would have been
        correct; high values mean the rules are too cautious.
      </p>

      {loading ? (
        <LoadingPane label="Loading abstention eval..." />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Abstention eval has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard
              label="Full-data coverage"
              value={formatRate(summary.full_data_coverage_rate)}
              status={(summary.full_data_coverage_rate ?? 0) >= 0.95 ? "green" : "amber"}
            />
            <MetricCard
              label="Full-data accuracy"
              value={formatRate(summary.full_data_covered_accuracy)}
              status={(summary.full_data_covered_accuracy ?? 0) >= 0.80 ? "green" : "amber"}
            />
            <MetricCard
              label="Demographics-only abstention"
              value={formatRate(summary.demographics_only_abstention_rate)}
              status={(summary.demographics_only_abstention_rate ?? 0) >= 0.95 ? "green" : "amber"}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  <th className="text-left font-semibold py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border)" }}>Scenario</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Rows</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Coverage</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Abstention</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Covered accuracy</th>
                  <th className="text-right font-semibold py-1.5 pl-2" style={{ borderBottom: "1px solid var(--border)" }}>False abstention</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((s) => (
                  <tr key={s.scenario}>
                    <td className="py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border-soft)", fontWeight: 600 }}>
                      {s.scenario}
                    </td>
                    <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>{s.rows_evaluated}</td>
                    <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.coverage_rate)}</td>
                    <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.abstention_rate)}</td>
                    <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.covered_accuracy)}</td>
                    <td className="py-1.5 pl-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.false_abstention_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {report.generated_at && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Last run: {new Date(report.generated_at).toLocaleString()} · {report.rows_evaluated ?? 0} rows
            </p>
          )}
        </>
      )}
    </Card>
  );
}

function formatRate(value: number | null | undefined): string {
  if (value == null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}
