import { Badge } from "../../../../components/ui/Badge";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane, ErrorPane } from "../../../../components/ui/Spinner";
import type { MultilingualRefusalEval } from "../../../../types/api";
import { fmtRate, statusBadge } from "./safetyFormat";

const COLUMNS = ["Case", "Language", "Expected", "Observed", "Pass"] as const;
const MAX_ROWS = 8;

/**
 * Tagalog/Taglish safety-routing benchmark.
 *
 * The caveat line is load-bearing: this is regression coverage for two
 * language variants, not evidence of broad multilingual safety, and the UI
 * must not let a high pass rate imply otherwise.
 */
export function MultilingualRefusalBlock({ artifact }: { artifact: MultilingualRefusalEval | null }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="No multilingual refusal benchmark has been generated yet." />;
  }
  if (artifact.status === "error") {
    return <ErrorPane message={artifact.message ?? "Multilingual refusal artifact error"} />;
  }
  const summary = artifact.summary;
  if (!summary) return <EmptyPane label="Artifact missing summary block." />;

  const rows = artifact.cases ?? [];
  const failedCount = summary.failed_cases?.length ?? 0;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Status" value={summary.status ?? "unknown"} status={statusBadge(summary.status)} />
        <MetricCard label="Pass rate" value={fmtRate(summary.pass_rate)} status={summary.pass_rate === 1 ? "green" : "amber"} />
        <MetricCard label="Cases" value={summary.case_count ?? rows.length} status="muted" />
        <MetricCard label="Failed" value={failedCount} status={failedCount ? "red" : "green"} />
      </div>

      {rows.length === 0 ? (
        <EmptyPane label="Summary present, but no per-case rows were returned." />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <caption className="sr-only">Multilingual refusal benchmark cases</caption>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {COLUMNS.map((h) => (
                  <th key={h} scope="col" className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, MAX_ROWS).map((row) => (
                <tr key={row.case_id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                  <th scope="row" className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{row.case_id}</th>
                  <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.language}</td>
                  <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.expected_intent}</td>
                  <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.observed_intent}</td>
                  <td className="py-2 pr-4"><Badge variant={row.pass ? "green" : "red"}>{row.pass ? "pass" : "fail"}</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
        Tagalog/Taglish safety routing benchmark. It is regression coverage, not proof of broad multilingual safety.
      </p>
    </div>
  );
}
