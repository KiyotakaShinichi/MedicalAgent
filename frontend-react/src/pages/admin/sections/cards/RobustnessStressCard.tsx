import { RefreshCw, ShieldAlert } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Button } from "../../../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane, LoadingPane } from "../../../../components/ui/Spinner";
import type { RobustnessStressReport } from "../../../../types/api";

function formatRate(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function RobustnessStressCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: RobustnessStressReport | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const summary = report?.summary ?? {};
  const cases = report?.cases ?? [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldAlert size={14} style={{ color: "var(--amber, #92400e)" }} aria-hidden="true" />
          <SectionTitle>Synthetic robustness stress suite</SectionTitle>
          <Badge variant={statusVariant(
            status === "strong" ? "strong" :
            status === "acceptable" ? "acceptable" :
            status === "missing" ? "stale" : "needs_attention",
          )}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Running..." : "Rerun stress"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Fault-injection suite for missing labs, missing imaging, wrong units,
        contradictory symptoms, delayed reports, noisy tumor markers,
        incomplete family history, and ambiguous biomarkers. Passing means the
        system routes to uncertainty, abstention, or clinician review instead
        of overconfident clinical claims.
      </p>

      {loading ? (
        <LoadingPane label="Loading robustness stress report..." />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Robustness stress report has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard
              label="Pass rate"
              value={formatRate(summary.pass_rate)}
              status={(summary.pass_rate ?? 0) >= 0.95 ? "green" : "amber"}
            />
            <MetricCard label="Stress cases" value={summary.case_count ?? cases.length} status="muted" />
            <MetricCard
              label="Abstain/review route"
              value={formatRate(summary.abstention_or_review_rate)}
              status={(summary.abstention_or_review_rate ?? 0) >= 0.90 ? "green" : "amber"}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  <th className="text-left font-semibold py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border)" }}>Case</th>
                  <th className="text-left font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Category</th>
                  <th className="text-left font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Expected</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Review</th>
                  <th className="text-right font-semibold py-1.5 pl-2" style={{ borderBottom: "1px solid var(--border)" }}>Result</th>
                </tr>
              </thead>
              <tbody>
                {cases.slice(0, 10).map((c, index) => {
                  const name = c.case ?? c.case_id ?? `case_${index + 1}`;
                  const review = c.clinician_review_routed ?? c.clinician_review ?? c.abstained_any_head ?? c.abstained ?? false;
                  return (
                    <tr key={name}>
                      <td className="py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border-soft)", fontWeight: 600 }}>{name.replace(/_/g, " ")}</td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>{c.category}</td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>{(c.expected ?? c.expected_behavior ?? "safe routing").replace(/_/g, " ")}</td>
                      <td className="py-1.5 px-2 text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{review ? "yes" : "no"}</td>
                      <td className="py-1.5 pl-2 text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                        <Badge variant={c.passed ? "green" : "red"}>{c.passed ? "passed" : "failed"}</Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {report.generated_at && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Last run: {new Date(report.generated_at).toLocaleString()}
            </p>
          )}
        </>
      )}
    </Card>
  );
}
