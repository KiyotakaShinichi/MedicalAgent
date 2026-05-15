import { lazy, Suspense } from "react";
import { FlaskConical, Droplet, Heart, Shield } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { MetricCard } from "../../components/ui/MetricCard";
import { RelativeTime } from "../../components/ui/RelativeTime";
import type { PatientReport } from "../../types/api";

// Recharts is ~200 KB gzipped — code-split so the chart only loads when the
// labs panel actually renders.
const LabTrendsChart = lazy(() =>
  import("../../components/charts/LabTrendsChart").then((m) => ({ default: m.LabTrendsChart })),
);

function ChartFallback() {
  return (
    <div
      style={{
        height: 220,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--text-faint)",
        fontSize: "0.78rem",
      }}
    >
      Loading chart…
    </div>
  );
}

type Trend = "up" | "down" | "neutral";

function labStatus(val: number | null, key: string): "green" | "amber" | "red" | "muted" {
  if (val == null) return "muted";
  if (key === "wbc")        return val < 2 ? "red" : val < 4 ? "amber" : "green";
  if (key === "hemoglobin") return val < 8 ? "red" : val < 11 ? "amber" : "green";
  if (key === "platelets")  return val < 50 ? "red" : val < 100 ? "amber" : "green";
  return "muted";
}

function computeTrend(history: number[], threshold = 0.05): { trend: Trend; label: string } {
  if (history.length < 2) return { trend: "neutral", label: "Stable" };
  const last = history[history.length - 1];
  const baseline = history[0];
  if (baseline === 0) return { trend: "neutral", label: "Stable" };
  const change = (last - baseline) / Math.abs(baseline);
  if (Math.abs(change) < threshold) return { trend: "neutral", label: "Stable" };
  const pct = Math.round(Math.abs(change) * 100);
  return change > 0
    ? { trend: "up",   label: `+${pct}% vs baseline` }
    : { trend: "down", label: `-${pct}% vs baseline` };
}

interface Props { report: PatientReport; lastFetchedAt?: number | null }

export function LabsPanel({ report, lastFetchedAt }: Props) {
  const { latest_labs: labs, lab_history } = report;
  const history = lab_history ?? [];
  const wbcSeries = history.map((r) => Number(r.wbc)).filter((v) => Number.isFinite(v));
  const hgbSeries = history.map((r) => Number(r.hemoglobin)).filter((v) => Number.isFinite(v));
  const pltSeries = history.map((r) => Number(r.platelets)).filter((v) => Number.isFinite(v));
  const wbcTrend = computeTrend(wbcSeries);
  const hgbTrend = computeTrend(hgbSeries);
  const pltTrend = computeTrend(pltSeries);

  return (
    <SectionCard
      title="Lab values (CBC)"
      icon={FlaskConical}
      meta={
        <span className="flex items-center gap-2">
          <span>{history.length > 0 ? `${history.length} samples` : "no data"}</span>
          {lastFetchedAt != null && <span style={{ opacity: 0.6 }}>·</span>}
          <RelativeTime timestamp={lastFetchedAt ?? null} prefix="updated" />
        </span>
      }
    >
      <div className="grid grid-cols-3 gap-3 mb-4">
        <MetricCard
          label="WBC"
          icon={Shield}
          value={labs?.wbc != null ? labs.wbc.toFixed(1) : null}
          unit="K/uL"
          status={labStatus(labs?.wbc ?? null, "wbc")}
          trend={wbcSeries.length > 1 ? wbcTrend.trend : undefined}
          trendLabel={wbcSeries.length > 1 ? wbcTrend.label : undefined}
          range="4.0-11.0"
        />
        <MetricCard
          label="Hemoglobin"
          icon={Heart}
          value={labs?.hemoglobin != null ? labs.hemoglobin.toFixed(1) : null}
          unit="g/dL"
          status={labStatus(labs?.hemoglobin ?? null, "hemoglobin")}
          trend={hgbSeries.length > 1 ? hgbTrend.trend : undefined}
          trendLabel={hgbSeries.length > 1 ? hgbTrend.label : undefined}
          range="12.0-16.0"
        />
        <MetricCard
          label="Platelets"
          icon={Droplet}
          value={labs?.platelets != null ? Math.round(labs.platelets) : null}
          unit="K/uL"
          status={labStatus(labs?.platelets ?? null, "platelets")}
          trend={pltSeries.length > 1 ? pltTrend.trend : undefined}
          trendLabel={pltSeries.length > 1 ? pltTrend.label : undefined}
          range="150-400"
        />
      </div>
      <Suspense fallback={<ChartFallback />}>
        <LabTrendsChart data={lab_history ?? []} />
      </Suspense>
    </SectionCard>
  );
}
