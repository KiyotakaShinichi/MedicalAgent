import { Cpu, TrendingUp, TrendingDown, AlertTriangle, Info } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { SafetyBanner } from "../../components/ui/SafetyBanner";
import { EmptyState } from "../../components/ui/states";
import type { PatientReport } from "../../types/api";

interface Props { report: PatientReport }

function FeatureBar({ label, value, positive }: { label: string; value: number; positive: boolean }) {
  const w = Math.min(Math.abs(value) * 100, 100);
  const color = positive ? "#059669" : "#dc2626";
  const bgTint = positive ? "rgba(5,150,105,0.10)" : "rgba(220,38,38,0.10)";
  return (
    <div className="flex items-center gap-2.5 py-1.5">
      {positive
        ? <TrendingUp size={13} style={{ color, flexShrink: 0 }} />
        : <TrendingDown size={13} style={{ color, flexShrink: 0 }} />
      }
      <span className="text-[0.8rem] truncate flex-1" style={{ color: "var(--text)" }}>{label}</span>
      <div className="rounded-full overflow-hidden" style={{ width: 84, height: 5, background: bgTint }}>
        <div style={{ width: `${w}%`, height: 5, background: color, borderRadius: 999 }} />
      </div>
      <span
        className="text-[0.74rem] tabular-nums font-medium"
        style={{ color: "var(--text-dim)", minWidth: 44, textAlign: "right" }}
      >
        {value >= 0 ? "+" : ""}{value.toFixed(3)}
      </span>
    </div>
  );
}

export function ModelSignalPanel({ report }: Props) {
  const pred = report.synthetic_model_prediction;
  const expl = report.synthetic_model_explanation;
  const xai = report.xai_explanation_envelope;
  const mma = report.multimodal_assessment;

  // Show the card even with no signal so the patient surface stays consistent
  // and the calibration footer ("synthetic engineering signal, not a clinical
  // prediction, for clinician review") remains visible. Empty-state lives at
  // the bottom of the card so reviewers can see why nothing rendered.
  const hasAnySignal = Boolean(pred || mma);

  const hybrid = pred?.hybrid_mle_signal;
  const score = hybrid?.hybrid_score;
  const scoreColor =
    score == null ? "var(--text-faint)" :
    score >= 70 ? "#059669" :
    score >= 40 ? "#d97706" : "#dc2626";

  const explCount = (expl?.positive_contributions?.length ?? 0) + (expl?.negative_contributions?.length ?? 0);

  return (
    <SectionCard title="Model signal" icon={Cpu} meta="synthetic-only">
      <SafetyBanner tone="warning" compact className="mb-4">
        Synthetic engineering signal · Not a clinical prediction · For clinician review.
      </SafetyBanner>

      <details className="model-number-guide mb-4">
        <summary><Info size={12} aria-hidden="true" /> What the model numbers mean</summary>
        <div>
          <p><strong>Hybrid score:</strong> 65% calibrated classifier probability plus 35% normalized regression output when both are available.</p>
          <p><strong>Classification probability:</strong> model confidence for a simulator-built label, not the patient's personal outcome probability.</p>
          <p><strong>Feature contribution:</strong> how a field pushed this model output up or down. It has no clinical unit and does not prove causation.</p>
        </div>
      </details>

      {xai && (
        <div className="model-explanation-envelope mb-4" aria-label="Synthetic model explanation">
          <div className="model-explanation-envelope__header">
            <Info size={14} aria-hidden="true" />
            <strong>{xai.status === "abstained" ? "Why the model abstained" : "How to read this output"}</strong>
          </div>
          <p>{xai.output.meaning}</p>
          <p><strong>How calculated:</strong> {xai.output.calculation}</p>
          <dl>
            <div>
              <dt>Records used</dt>
              <dd>{xai.evidence.inputs_used.length ? xai.evidence.inputs_used.join(", ") : "No sufficient modality recorded"}</dd>
            </div>
            <div>
              <dt>Missing or limited</dt>
              <dd>{xai.evidence.inputs_missing.length ? xai.evidence.inputs_missing.join(", ") : "None reported by the evidence gate"}</dd>
            </div>
            <div>
              <dt>Uncertainty</dt>
              <dd>{xai.evidence.abstain_reason || xai.uncertainty.explanation}</dd>
            </div>
          </dl>
          {xai.safe_next_steps[0] && (
            <p className="model-explanation-envelope__next"><strong>Safe next step:</strong> {xai.safe_next_steps[0]}</p>
          )}
        </div>
      )}

      {hybrid && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div
            className="rounded-lg border p-3"
            style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
          >
            <p className="text-[0.7rem] uppercase tracking-wider font-semibold" style={{ color: "var(--text-faint)" }}>
              Synthetic hybrid score
            </p>
            <p className="text-2xl font-bold tabular-nums mt-1" style={{ color: scoreColor }}>
              {score != null ? score.toFixed(0) : "-"}
            </p>
          </div>
          <div
            className="rounded-lg border p-3"
            style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
          >
            <p className="text-[0.7rem] uppercase tracking-wider font-semibold" style={{ color: "var(--text-faint)" }}>
              Synthetic class probability
            </p>
            <p className="text-2xl font-bold tabular-nums mt-1" style={{ color: "var(--text-strong)" }}>
              {hybrid.classification_probability != null
                ? `${(hybrid.classification_probability * 100).toFixed(1)}%`
                : "-"}
            </p>
          </div>
        </div>
      )}

      {mma && (
        <div className="mb-4">
          <p className="text-[0.72rem] uppercase tracking-wider font-semibold mb-2" style={{ color: "var(--text-faint)" }}>
            Modal signals
          </p>
          <div className="flex flex-col gap-1.5">
            {Object.entries(mma.signals).map(([key, sig]) => sig ? (
              <div
                key={key}
                className="flex items-start gap-2.5 py-1.5"
              >
                <StatusBadge
                  tone={
                    sig.status?.includes("stable") || sig.status?.includes("low") ? "success" :
                    sig.status?.includes("concern") || sig.status?.includes("warn") ? "warning" :
                    "neutral"
                  }
                  size="sm"
                >
                  {key.replace(/_/g, " ")}
                </StatusBadge>
                <span className="text-[0.82rem] flex-1 leading-relaxed" style={{ color: "var(--text-dim)" }}>
                  {sig.message}
                </span>
              </div>
            ) : null)}
          </div>
        </div>
      )}

      {expl && explCount > 0 && (
        <div>
          <p
            className="text-[0.72rem] uppercase tracking-wider font-semibold mb-2 flex items-center gap-1.5"
            style={{ color: "var(--text-faint)" }}
          >
            Relative model contributions (SHAP)
          </p>
          <div>
            {[
              ...(expl.positive_contributions ?? []).slice(0, 4).map(f => ({ ...f, pos: true })),
              ...(expl.negative_contributions ?? []).slice(0, 3).map(f => ({ ...f, pos: false })),
            ].map((f, i) => (
              <FeatureBar key={i} label={f.feature} value={f.shap_value} positive={f.pos} />
            ))}
          </div>
        </div>
      )}

      {expl && explCount === 0 && (
        <EmptyState label="No SHAP explanations available." />
      )}

      {!hasAnySignal && (
        <div
          className="flex items-start gap-2 text-sm rounded-md border p-3"
          style={{
            color: "var(--text-dim)",
            background: "var(--surface2)",
            borderColor: "var(--border)",
          }}
        >
          <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1, color: "var(--text-faint)" }} />
          <span>
            <strong style={{ color: "var(--text-strong)" }}>No model output yet.</strong>{" "}
            The hybrid monitoring stack abstains until at least one modality
            (CBC, imaging, or symptoms) has enough recent data.
          </span>
        </div>
      )}
      {!expl && !hybrid && !mma && hasAnySignal && (
        <div className="flex items-center gap-2 text-sm" style={{ color: "var(--text-dim)" }}>
          <AlertTriangle size={14} />
          No model output available for this patient.
        </div>
      )}
    </SectionCard>
  );
}
