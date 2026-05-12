import { Cpu, TrendingUp, TrendingDown } from "lucide-react";
import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { EmptyPane } from "../../components/ui/Spinner";
import type { PatientReport } from "../../types/api";

interface Props { report: PatientReport }

function FeatureBar({ label, value, positive }: { label: string; value: number; positive: boolean }) {
  const w = Math.min(Math.abs(value) * 100, 100);
  return (
    <div className="flex items-center gap-2 py-1">
      {positive
        ? <TrendingUp size={12} style={{ color: "var(--green)", flexShrink: 0 }} />
        : <TrendingDown size={12} style={{ color: "var(--rose)", flexShrink: 0 }} />
      }
      <span className="text-xs truncate flex-1" style={{ color: "var(--text-dim)" }}>{label}</span>
      <div className="w-24 h-1.5 rounded-full" style={{ background: "var(--border)" }}>
        <div
          className="h-1.5 rounded-full"
          style={{ width: `${w}%`, background: positive ? "var(--green)" : "var(--rose)" }}
        />
      </div>
      <span className="text-xs tabular-nums w-10 text-right" style={{ color: "var(--text-dim)" }}>
        {value >= 0 ? "+" : ""}{value.toFixed(3)}
      </span>
    </div>
  );
}

export function ModelSignalPanel({ report }: Props) {
  const pred = report.synthetic_model_prediction;
  const expl = report.synthetic_model_explanation;
  const mma = report.multimodal_assessment;

  if (!pred && !mma) return null;

  const hybrid = pred?.hybrid_mle_signal;
  const score = hybrid?.hybrid_score;
  const scoreColor =
    score == null ? "var(--text-faint)" :
    score >= 70 ? "var(--green)" :
    score >= 40 ? "var(--amber)" : "var(--rose)";

  return (
    <Card>
      <CardHeader>
        <SectionTitle>Model Signal</SectionTitle>
        <Cpu size={14} style={{ color: "var(--text-faint)" }} />
      </CardHeader>

      <p
        className="text-xs px-3 py-2 rounded-md mb-3 border"
        style={{ background: "rgba(245,158,11,0.06)", borderColor: "rgba(245,158,11,0.2)", color: "var(--amber)" }}
      >
        Exploratory engineering signal only - not a clinical prediction.
      </p>

      {hybrid && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div>
            <p className="text-xs" style={{ color: "var(--text-faint)" }}>Hybrid MLE score</p>
            <p className="text-2xl font-bold tabular-nums" style={{ color: scoreColor }}>
              {score != null ? score.toFixed(0) : "-"}
            </p>
          </div>
          <div>
            <p className="text-xs" style={{ color: "var(--text-faint)" }}>Classification probability</p>
            <p className="text-2xl font-bold tabular-nums" style={{ color: "var(--text)" }}>
              {hybrid.classification_probability != null
                ? `${(hybrid.classification_probability * 100).toFixed(1)}%`
                : "-"}
            </p>
          </div>
        </div>
      )}

      {mma && (
        <div className="flex flex-col gap-1 mb-4">
          {Object.entries(mma.signals).map(([key, sig]) => sig ? (
            <div key={key} className="flex items-center gap-2 py-1 border-b last:border-0" style={{ borderColor: "var(--border)" }}>
              <Badge variant={
                sig.status?.includes("stable") || sig.status?.includes("low") ? "green" :
                sig.status?.includes("concern") || sig.status?.includes("warn") ? "amber" : "muted"
              }>
                {key.replace(/_/g, " ")}
              </Badge>
              <span className="text-xs flex-1" style={{ color: "var(--text-dim)" }}>{sig.message}</span>
            </div>
          ) : null)}
        </div>
      )}

      {expl && (
        <div>
          <p className="text-xs font-medium mb-1.5" style={{ color: "var(--text-faint)" }}>Feature contributions (SHAP)</p>
          {[...(expl.positive_contributions ?? []).slice(0, 4).map(f => ({ ...f, pos: true })),
            ...(expl.negative_contributions ?? []).slice(0, 2).map(f => ({ ...f, pos: false }))]
            .map((f, i) => (
              <FeatureBar key={i} label={f.feature} value={f.shap_value} positive={f.pos} />
            ))}
          {(expl.positive_contributions?.length ?? 0) + (expl.negative_contributions?.length ?? 0) === 0 && (
            <EmptyPane label="No explanations available" />
          )}
        </div>
      )}
    </Card>
  );
}
