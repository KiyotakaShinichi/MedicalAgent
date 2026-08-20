import { Badge } from "../../../../components/ui/Badge";
import type { SafetyCenterCategorySummary } from "../../../../types/api";
import { fmtRate, statusBadge } from "./safetyFormat";

export interface CategoryRow {
  label: string;
  summary: SafetyCenterCategorySummary;
}

/**
 * Per-category pass rates for the four safety guarantees the red-team suite
 * exercises (injection, escalation, refusal, privacy).
 *
 * Uses a definition list so screen readers announce each category label paired
 * with its rate, rather than a wall of unrelated numbers.
 */
export function CategoryGrid({ rows }: { rows: CategoryRow[] }) {
  if (rows.length === 0) return null;
  return (
    <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-3">
      {rows.map(({ label, summary }) => (
        <div
          key={label}
          className="p-3 rounded-md border"
          style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
        >
          <dt className="text-xs font-medium mb-1.5" style={{ color: "var(--text)" }}>
            {label}
          </dt>
          <dd className="m-0">
            <div className="flex items-center justify-between">
              <span
                className="text-sm font-bold tabular-nums"
                style={{
                  color:
                    summary.status === "passed"
                      ? "var(--green)"
                      : summary.status === "needs_attention"
                        ? "var(--rose)"
                        : "var(--text-dim)",
                }}
              >
                {fmtRate(summary.pass_rate)}
              </span>
              <Badge variant={statusBadge(summary.status)}>{summary.status}</Badge>
            </div>
            <p className="text-xs mt-1" style={{ color: "var(--text-dim)" }}>
              {summary.case_count} case{summary.case_count !== 1 ? "s" : ""}
            </p>
          </dd>
        </div>
      ))}
    </dl>
  );
}
