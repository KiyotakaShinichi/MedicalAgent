import { ShieldCheck, ShieldAlert, Info, Activity, TrendingUp, AlertTriangle } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { EmptyPane } from "../../components/ui/Spinner";
import type { HybridPrediction, EvidenceAwarePrediction, EvidenceAwareRegression } from "../../types/api";

interface Props { hybrid: HybridPrediction | null | undefined }

/**
 * Hybrid evidence-aware monitoring signal.  Renders three slots:
 *   - Response classification ("favorable / concerning / uncertain / insufficient evidence")
 *   - Response-score regression ("strong / moderate / weak signal" + uncertainty band)
 *   - Toxicity classification ("low / moderate / high toxicity signal")
 *
 * Each slot has its own evidence envelope, its own model version, and its
 * own abstention behavior.  When the rules abstain on one head, that head
 * shows "insufficient evidence" while the others may still score — this is
 * what makes the hybrid honest under partial inputs.
 */
export function HybridPredictionCard({ hybrid }: Props) {
  if (hybrid === undefined) return null;
  if (!hybrid) {
    return (
      <SectionCard title="Hybrid monitoring signal" icon={Info}>
        <EmptyPane label="No live hybrid prediction available for this patient yet." />
      </SectionCard>
    );
  }

  return (
    <SectionCard
      title="Hybrid monitoring signal"
      icon={ShieldCheck}
      meta={<span style={{ color: "var(--text-faint)", fontSize: "0.74rem" }}>3 signals</span>}
      footer={
        <span className="inline-flex items-start gap-1.5">
          <Info size={11} aria-hidden="true" style={{ marginTop: 2, flexShrink: 0 }} />
          <span>{hybrid.claim_boundary}</span>
        </span>
      }
    >
      <div className="flex flex-col gap-3">
        <ClassificationSlot title="Response classification" icon={Activity} prediction={hybrid.classification} />
        <RegressionSlot title="Response strength" icon={TrendingUp} prediction={hybrid.response_score} />
        <ClassificationSlot title="Toxicity signal" icon={AlertTriangle} prediction={hybrid.toxicity} />
      </div>
    </SectionCard>
  );
}

const DECISION_LABEL: Record<string, string> = {
  favorable_pattern:        "Favorable response pattern",
  concerning_pattern:       "Concerning response pattern",
  uncertain:                "Uncertain pattern",
  insufficient_evidence:    "Insufficient evidence",
  strong_response_signal:   "Strong response signal",
  moderate_response_signal: "Moderate response signal",
  weak_response_signal:     "Weak response signal",
  high_toxicity_signal:     "High toxicity signal",
  moderate_toxicity_signal: "Moderate toxicity signal",
  low_toxicity_signal:      "Low toxicity signal",
};

const DECISION_TONE: Record<string, { fg: string; bg: string; border: string }> = {
  favorable_pattern:        { fg: "#047857", bg: "#ecfdf5", border: "#a7f3d0" },
  concerning_pattern:       { fg: "#b91c1c", bg: "#fef2f2", border: "#fecaca" },
  uncertain:                { fg: "#92400e", bg: "#fffbeb", border: "#fde68a" },
  insufficient_evidence:    { fg: "var(--text-faint)", bg: "var(--surface2)", border: "var(--border)" },
  strong_response_signal:   { fg: "#047857", bg: "#ecfdf5", border: "#a7f3d0" },
  moderate_response_signal: { fg: "#92400e", bg: "#fffbeb", border: "#fde68a" },
  weak_response_signal:     { fg: "#b91c1c", bg: "#fef2f2", border: "#fecaca" },
  low_toxicity_signal:      { fg: "#047857", bg: "#ecfdf5", border: "#a7f3d0" },
  moderate_toxicity_signal: { fg: "#92400e", bg: "#fffbeb", border: "#fde68a" },
  high_toxicity_signal:     { fg: "#b91c1c", bg: "#fef2f2", border: "#fecaca" },
};

function toneFor(decision: string) {
  return DECISION_TONE[decision] ?? DECISION_TONE.insufficient_evidence;
}

function ClassificationSlot({
  title,
  icon: Icon,
  prediction,
}: {
  title: string;
  icon: typeof ShieldCheck;
  prediction: EvidenceAwarePrediction;
}) {
  const tone = toneFor(prediction.decision);
  const label = DECISION_LABEL[prediction.decision] ?? prediction.decision;
  const abstained = prediction.evidence?.abstain ?? false;
  const HeaderIcon = abstained ? ShieldAlert : Icon;
  return (
    <div className="rounded-md border p-3" style={{ background: tone.bg, borderColor: tone.border }}>
      <div className="flex items-center gap-2 mb-1.5">
        <HeaderIcon size={13} style={{ color: tone.fg }} aria-hidden="true" />
        <span className="text-[0.72rem] uppercase font-semibold tracking-wide" style={{ color: tone.fg, opacity: 0.9 }}>
          {title}
        </span>
        <span style={{ marginLeft: "auto", fontSize: "0.7rem", color: tone.fg, opacity: 0.7 }}>
          {abstained ? "abstained" : `confidence: ${prediction.confidence}`}
        </span>
      </div>
      <p className="text-sm font-bold" style={{ color: tone.fg, lineHeight: 1.25 }}>
        {label}
      </p>
      {prediction.probability != null && (
        <p className="text-[0.78rem] mt-1 tabular-nums" style={{ color: tone.fg, opacity: 0.85 }}>
          Probability: <strong>{(prediction.probability * 100).toFixed(1)}%</strong>
        </p>
      )}
      <EvidenceLine prediction={prediction} />
    </div>
  );
}

function RegressionSlot({
  title,
  icon: Icon,
  prediction,
}: {
  title: string;
  icon: typeof TrendingUp;
  prediction: EvidenceAwareRegression;
}) {
  const tone = toneFor(prediction.decision);
  const label = DECISION_LABEL[prediction.decision] ?? prediction.decision;
  const abstained = prediction.evidence?.abstain ?? false;
  const HeaderIcon = abstained ? ShieldAlert : Icon;
  return (
    <div className="rounded-md border p-3" style={{ background: tone.bg, borderColor: tone.border }}>
      <div className="flex items-center gap-2 mb-1.5">
        <HeaderIcon size={13} style={{ color: tone.fg }} aria-hidden="true" />
        <span className="text-[0.72rem] uppercase font-semibold tracking-wide" style={{ color: tone.fg, opacity: 0.9 }}>
          {title}
        </span>
        <span style={{ marginLeft: "auto", fontSize: "0.7rem", color: tone.fg, opacity: 0.7 }}>
          {abstained ? "abstained" : `confidence: ${prediction.confidence}`}
        </span>
      </div>
      <p className="text-sm font-bold" style={{ color: tone.fg, lineHeight: 1.25 }}>
        {label}
      </p>
      {prediction.response_score != null && (
        <div className="mt-1.5">
          <ScoreBar score={prediction.response_score} band={prediction.uncertainty_band} tone={tone.fg} />
          <p className="text-[0.74rem] mt-1 tabular-nums" style={{ color: tone.fg, opacity: 0.85 }}>
            Score: <strong>{prediction.response_score.toFixed(2)}</strong>
            {prediction.uncertainty_band && (
              <span style={{ marginLeft: 8, opacity: 0.7 }}>
                ±{((prediction.uncertainty_band[1] - prediction.uncertainty_band[0]) / 2).toFixed(2)} band
              </span>
            )}
          </p>
        </div>
      )}
      <EvidenceLine prediction={prediction} />
    </div>
  );
}

function ScoreBar({ score, band, tone }: { score: number; band: [number, number] | null; tone: string }) {
  const clamped = Math.max(0, Math.min(1, score));
  return (
    <div
      className="relative w-full"
      style={{ height: 8, borderRadius: 4, background: "rgba(0,0,0,0.06)", overflow: "hidden" }}
      aria-label={`Response score ${(score * 100).toFixed(0)} out of 100`}
    >
      {band && (
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            left: `${band[0] * 100}%`,
            width: `${(band[1] - band[0]) * 100}%`,
            top: 0,
            bottom: 0,
            background: tone,
            opacity: 0.18,
          }}
        />
      )}
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          left: `${clamped * 100}%`,
          top: 0,
          bottom: 0,
          width: 2,
          background: tone,
        }}
      />
    </div>
  );
}

function EvidenceLine({ prediction }: { prediction: { evidence: EvidenceAwarePrediction["evidence"]; model_version: string } }) {
  const ev = prediction.evidence;
  if (!ev) return null;
  const present = ev.modalities_present.length;
  const total = present + ev.modalities_missing.length;
  return (
    <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
      Modalities used: {present}/{total} · sufficiency: <strong>{ev.sufficiency}</strong>
      {" · "}<code style={{ fontSize: "0.66rem" }}>{prediction.model_version}</code>
      {ev.abstain && ev.reason && (
        <span style={{ display: "block", marginTop: 2 }}>
          <strong>Reason:</strong> {humanReason(ev.reason)}
        </span>
      )}
    </p>
  );
}

function humanReason(reason: string): string {
  switch (reason) {
    case "no_response_signal_imaging_or_longitudinal_cbc_required":
      return "Needs imaging or a complete cycle of CBC values.";
    case "missing_minimum_context":
      return "Patient demographics weren't available.";
    case "response_signal_from_single_modality_only":
      return "Only one response modality present — confidence reduced.";
    case "toxicity_signal_from_single_modality_only":
      return "Only one toxicity modality present — confidence reduced.";
    case "no_toxicity_signal_cbc_or_symptoms_required":
      return "Toxicity signal needs CBC or symptom data.";
    case "no_acute_signal_symptoms_or_nadir_cbc_required":
      return "Urgent signal needs symptoms or nadir CBC.";
    case "urgent_signal_from_single_modality_only":
      return "Only one acute signal present — confidence reduced.";
    default:
      return reason.replace(/_/g, " ");
  }
}
