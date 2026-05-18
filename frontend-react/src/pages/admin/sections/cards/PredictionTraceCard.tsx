import { RefreshCw, Info } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Button } from "../../../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane, LoadingPane } from "../../../../components/ui/Spinner";
import type { PredictionTraceResponse } from "../../../../types/api";

function formatRate(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function PredictionTraceCard({
  response,
  loading,
  onRefresh,
}: {
  response: PredictionTraceResponse | null;
  loading: boolean;
  onRefresh: () => void;
}) {
  const traces = response?.traces ?? [];
  const summary = response?.summary;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Info size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>Prediction trace log</SectionTitle>
          <Badge variant={statusVariant((summary?.total ?? 0) > 0 ? "strong" : "stale")}>
            {summary?.total ?? 0} TRACES
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={loading} icon={<RefreshCw size={13} />}>
          Refresh
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        One row per live evidence-aware prediction. Each trace records the
        model, feature-set, threshold, calibration, modalities present,
        abstention status, and safety-validator decision.
      </p>

      {loading ? (
        <LoadingPane label="Loading prediction traces..." />
      ) : traces.length === 0 ? (
        <EmptyPane label="No prediction traces recorded yet - they are written by live inference." />
      ) : (
        <>
          {summary && (
            <div className="grid sm:grid-cols-3 gap-3 mb-3">
              <MetricCard label="Recent traces" value={summary.total} status="muted" />
              <MetricCard
                label="Abstention rate"
                value={formatRate(summary.abstention_rate)}
                status={(summary.abstention_rate ?? 0) > 0.5 ? "amber" : "green"}
              />
              <MetricCard
                label="Model versions seen"
                value={summary.model_versions.length}
                status={summary.model_versions.length > 1 ? "amber" : "muted"}
              />
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  {["When", "Patient", "Question", "Decision", "Prob.", "Conf.", "Evidence", "Modalities", "Validator"].map((h) => (
                    <th key={h} className="text-left font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {traces.slice(0, 12).map((t) => (
                  <tr key={t.id}>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                      {t.created_at ? new Date(t.created_at).toLocaleString() : "—"}
                    </td>
                    <td className="py-1.5 px-2 font-semibold" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.patient_id ?? "—"}</td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.question}</td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: t.abstained ? "var(--amber)" : "var(--text)" }}>{t.decision}</td>
                    <td className="py-1.5 px-2 tabular-nums" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.probability == null ? "—" : t.probability.toFixed(3)}</td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.confidence ?? "—"}</td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.evidence_sufficiency ?? "—"}</td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                      {t.modalities_present.length}/{t.modalities_present.length + t.modalities_missing.length}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.validator_decision ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {traces.length > 12 && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Showing first 12 of {traces.length} recent traces.
            </p>
          )}
        </>
      )}
    </Card>
  );
}
