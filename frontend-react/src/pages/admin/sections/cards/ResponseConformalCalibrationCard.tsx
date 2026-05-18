import { RefreshCw, ShieldCheck } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Button } from "../../../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane, LoadingPane } from "../../../../components/ui/Spinner";
import type { ResponseConformalCalibrationReport } from "../../../../types/api";

function formatRate(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function ResponseConformalCalibrationCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: ResponseConformalCalibrationReport | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const adjustedMeetsNominal =
    report?.adjusted_coverage != null &&
    report?.nominal_coverage != null &&
    report.adjusted_coverage >= report.nominal_coverage - 0.01;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldCheck size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>Response-score conformal calibration</SectionTitle>
          <Badge variant={statusVariant(
            status === "strong" ? "strong" :
            status === "acceptable" ? "acceptable" :
            status === "missing" ? "stale" : "needs_attention",
          )}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Running..." : "Rerun calibration"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Split-conformal adjustment for the response-score regression interval.
        The raw quantile band is widened by a held-out residual quantile so the
        interval is calibrated as an engineering reliability signal, not a
        clinical guarantee.
      </p>

      {loading ? (
        <LoadingPane label="Loading conformal calibration..." />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Conformal calibration has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-4 gap-3 mb-3">
            <MetricCard label="Nominal coverage" value={formatRate(report.nominal_coverage)} status="muted" />
            <MetricCard
              label="Raw coverage"
              value={formatRate(report.raw_coverage)}
              status={(report.raw_coverage ?? 0) >= (report.nominal_coverage ?? 1) ? "green" : "amber"}
            />
            <MetricCard
              label="Adjusted coverage"
              value={formatRate(report.adjusted_coverage)}
              status={adjustedMeetsNominal ? "green" : "amber"}
            />
            <MetricCard
              label="qhat widen"
              value={report.qhat_percent != null ? report.qhat_percent.toFixed(3) : null}
              status="muted"
            />
          </div>
          {report.interpretation && (
            <p className="text-xs mb-2" style={{ color: "var(--text-dim)" }}>{report.interpretation}</p>
          )}
          {report.generated_at && (
            <p className="text-[0.7rem]" style={{ color: "var(--text-faint)" }}>
              Last run: {new Date(report.generated_at).toLocaleString()}
              {report.calibration_rows != null && <> · calibration rows: {report.calibration_rows}</>}
            </p>
          )}
        </>
      )}
    </Card>
  );
}
