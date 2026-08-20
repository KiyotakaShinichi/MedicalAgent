import { MetricCard } from "../../../../components/ui/MetricCard";
import type { SafetyCenter } from "../../../../types/api";
import { fmtScore, statusBadge } from "./safetyFormat";

type CalibrationMetrics = SafetyCenter["calibration_metrics"];

/** Expected calibration error at or below this reads as calibrated. */
const ECE_PASS_THRESHOLD = 0.05;

function eceTone(value: number | null | undefined) {
  const met = value !== null && value !== undefined && value <= ECE_PASS_THRESHOLD;
  return statusBadge(met ? "passed" : "watch");
}

/** Probability-calibration metrics for the champion model. */
export function CalibrationBlock({ calibration }: { calibration: CalibrationMetrics }) {
  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Best model" value={calibration.best_model ?? "—"} status="muted" />
        <MetricCard
          label="ECE (pre-temp)"
          value={fmtScore(calibration.ece_before)}
          status={eceTone(calibration.ece_before)}
        />
        <MetricCard
          label="ECE (post-temp)"
          value={fmtScore(calibration.ece_after)}
          status={eceTone(calibration.ece_after)}
        />
        <MetricCard label="Brier score" value={fmtScore(calibration.brier_score)} status="muted" />
      </div>
      {calibration.note && (
        <p className="text-xs mt-2" style={{ color: "var(--text-dim)" }}>
          {calibration.note}
        </p>
      )}
    </>
  );
}
