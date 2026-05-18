import { RefreshCw, ShieldCheck } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { Button } from "../../../../components/ui/Button";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { LoadingPane, EmptyPane } from "../../../../components/ui/Spinner";
import type { ModalityRobustnessComparisonReport } from "../../../../types/api";

/**
 * Phase 4 — Champion vs modality-robust comparison.
 *
 * Runs both classifiers against the same modality-dropout scenarios and
 * reports per-scenario accuracy + Brier deltas.  Positive accuracy delta
 * means the robust variant beats the original champion.
 */
export function ModalityRobustnessCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: ModalityRobustnessComparisonReport | null;
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
          <ShieldCheck size={14} style={{ color: "var(--green, #047857)" }} aria-hidden="true" />
          <SectionTitle>Champion vs modality-robust comparison</SectionTitle>
          <Badge variant={statusVariant(
            status === "robust" ? "strong" :
            status === "acceptable" ? "acceptable" :
            status === "missing" ? "stale" : "needs_attention",
          )}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Running…" : "Rerun comparison"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Runs both classifiers against the same modality-dropout scenarios.
        <strong> Force-score</strong> deltas isolate the trained model's
        intrinsic robustness (no abstention rules applied);
        <strong> with-abstention</strong> deltas match production behavior.
        Positive accuracy delta = robust variant beats the champion.
      </p>

      {loading ? (
        <LoadingPane label="Loading comparison…" />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Comparison has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard
              label="Robust wins (force-score)"
              value={summary.force_score_accuracy_wins_for_robust ?? 0}
              status="green"
            />
            <MetricCard
              label="Robust losses"
              value={summary.force_score_accuracy_losses_for_robust ?? 0}
              status={(summary.force_score_accuracy_losses_for_robust ?? 0) > 0 ? "amber" : "muted"}
            />
            <MetricCard
              label="Full-data Δaccuracy"
              value={formatDelta(summary.full_data_accuracy_delta)}
              status={(summary.full_data_accuracy_delta ?? 0) >= -0.005 ? "green" : "amber"}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  <th className="text-left font-semibold py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border)" }}>Scenario</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Champ acc.</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Robust acc.</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Δacc.</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Champ Brier</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Robust Brier</th>
                  <th className="text-right font-semibold py-1.5 pl-2" style={{ borderBottom: "1px solid var(--border)" }}>ΔBrier</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((s) => {
                  const delta = s.deltas.force_score_accuracy_robust_minus_champion ?? 0;
                  const deltaColor = delta > 0.005 ? "var(--green, #047857)" : delta < -0.005 ? "var(--rose, #b91c1c)" : "var(--text-dim)";
                  return (
                    <tr key={s.scenario}>
                      <td className="py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border-soft)", fontWeight: 600 }}>{s.scenario}</td>
                      <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.force_score.champion.accuracy)}</td>
                      <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.force_score.robust.accuracy)}</td>
                      <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)", color: deltaColor, fontWeight: 600 }}>{formatDelta(delta)}</td>
                      <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{s.force_score.champion.brier?.toFixed(3) ?? "—"}</td>
                      <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{s.force_score.robust.brier?.toFixed(3) ?? "—"}</td>
                      <td className="py-1.5 pl-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)", color: (s.deltas.force_score_brier_robust_minus_champion ?? 0) < 0 ? "var(--green, #047857)" : "var(--text-dim)" }}>{formatDelta(s.deltas.force_score_brier_robust_minus_champion)}</td>
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

function formatRate(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatDelta(value: number | null | undefined): string {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}pp`;
}
