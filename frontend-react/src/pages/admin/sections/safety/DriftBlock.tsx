import { Activity } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane } from "../../../../components/ui/Spinner";
import type { DriftReport } from "../../../../types/api";
import { fmtRate, fmtScore, statusBadge } from "./safetyFormat";

const MAX_SHIFT_ROWS = 8;

interface ShiftRow {
  feature?: string;
  keyword?: string;
  baseline_mean?: number;
  current_mean?: number;
  baseline_rate?: number;
  current_rate?: number;
  standardized_shift?: number;
  shift?: number;
  status: string;
}

function ShiftPanel({
  title,
  status,
  rows,
}: {
  title: string;
  status: string | undefined;
  rows: ShiftRow[];
}) {
  return (
    <section
      className="p-3 rounded-md border"
      style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
      aria-label={title}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-semibold m-0" style={{ color: "var(--text)" }}>
          {title}
        </h4>
        <Badge variant={statusBadge(status)}>{status ?? "n/a"}</Badge>
      </div>
      {rows.length === 0 ? (
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>No features available.</p>
      ) : (
        <div className="flex flex-col gap-1">
          {rows.slice(0, MAX_SHIFT_ROWS).map((row, idx) => (
            <div key={row.feature ?? row.keyword ?? idx} className="flex items-center justify-between text-xs">
              <span style={{ color: "var(--text-dim)" }}>{row.feature ?? row.keyword}</span>
              <span className="tabular-nums" style={{ color: "var(--text)" }}>
                Δ {fmtScore(row.standardized_shift ?? row.shift, 2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

/**
 * Data-drift and data-quality panel.
 *
 * `unavailable` and `not_generated` collapse to the same empty state — in both
 * cases we have no drift signal, and showing zeroed metrics would imply "no
 * drift detected" when the truth is "drift was not measured".
 */
export function DriftBlock({ report }: { report: DriftReport }) {
  if (!report || report.status === "not_generated" || report.status === "unavailable") {
    return <EmptyPane label={report?.message ?? "Drift report not generated yet."} />;
  }

  const subgroups = report.subgroup_performance_drift?.groups ?? [];

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Data source"
          value={(report.data_source ?? "—").replace(/_/g, " ")}
          status="muted"
        />
        <MetricCard
          label="Missing CBC rate"
          value={fmtRate(report.missing_cbc_rate)}
          status={(report.missing_cbc_rate ?? 0) <= 0.2 ? "green" : "amber"}
        />
        <MetricCard
          label="Data completeness"
          value={fmtRate(report.data_completeness_score)}
          status={(report.data_completeness_score ?? 0) >= 0.85 ? "green" : "amber"}
        />
        <MetricCard
          label="Calibration drift"
          value={fmtScore(report.calibration_drift?.delta_ece, 3)}
          status={statusBadge(report.calibration_drift?.status)}
          sub="ECE Δ"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <ShiftPanel
          title="Lab distribution shift"
          status={report.lab_distribution_shift?.status}
          rows={report.lab_distribution_shift?.features ?? []}
        />
        <ShiftPanel
          title="Imaging keyword shift"
          status={report.imaging_keyword_shift?.status}
          rows={report.imaging_keyword_shift?.keywords ?? []}
        />
      </div>

      {subgroups.length > 0 && (
        <section aria-label="Subgroup performance drift">
          <h4 className="text-xs font-semibold mb-2 m-0" style={{ color: "var(--text)" }}>
            <Activity size={11} aria-hidden="true" style={{ display: "inline", marginRight: 4 }} />
            Subgroup performance drift
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {subgroups.map((g, i) => (
              <div
                key={`${g.group}-${g.value}-${i}`}
                className="p-2 rounded-md border text-xs"
                style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
              >
                <div className="flex items-center justify-between">
                  <span style={{ color: "var(--text)" }}>
                    {g.group}: <strong>{g.value}</strong>
                  </span>
                  <Badge variant={statusBadge(g.status)}>{g.status}</Badge>
                </div>
                <p style={{ color: "var(--text-dim)" }}>
                  base {fmtRate(g.baseline_positive_rate)} → cur {fmtRate(g.current_positive_rate)} (Δ {fmtScore(g.shift, 2)})
                </p>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
