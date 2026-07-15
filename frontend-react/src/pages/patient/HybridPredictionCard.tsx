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
      title="Synthetic monitoring model"
      icon={ShieldCheck}
      meta={<span style={{ color: "var(--text-faint)", fontSize: "0.74rem" }}>3 signals</span>}
      footer={
        <span className="inline-flex items-start gap-1.5">
          <Info size={11} aria-hidden="true" style={{ marginTop: 2, flexShrink: 0 }} />
          <span>
            <strong>Synthetic engineering signal · Not a clinical prediction · For clinician review.</strong>
            {" "}
            {hybrid.claim_boundary}
          </span>
        </span>
      }
    >
      <div className="flex flex-col gap-3">
        <details className="model-number-guide">
          <summary><Info size={12} aria-hidden="true" /> How to read these outputs</summary>
          <div>
            <p><strong>Probability</strong> is confidence in a synthetic model class, not a personal chance of response.</p>
            <p><strong>Score</strong> is a 0-1 simulator target estimate, not a diagnosis or validated clinical scale.</p>
            <p><strong>Confidence</strong> reflects model uncertainty and available record types. Missing inputs can reduce it or cause abstention.</p>
          </div>
        </details>
        <ClassificationSlot title="Response-pattern grouping" icon={Activity} prediction={hybrid.classification} />
        <RegressionSlot title="Synthetic response-score output" icon={TrendingUp} prediction={hybrid.response_score} />
        <ClassificationSlot title="Support-review grouping" icon={AlertTriangle} prediction={hybrid.toxicity} />
      </div>
    </SectionCard>
  );
}

const DECISION_LABEL: Record<string, string> = {
  favorable_pattern:        "Grouped with favorable synthetic examples",
  concerning_pattern:       "Grouped with review-priority synthetic examples",
  uncertain:                "Uncertain pattern",
  insufficient_evidence:    "Insufficient evidence",
  strong_response_signal:   "Strong response signal",
  moderate_response_signal: "Moderate response signal",
  weak_response_signal:     "Weak response signal",
  high_toxicity_signal:     "Higher support-review pattern",
  moderate_toxicity_signal: "Moderate support-review pattern",
  low_toxicity_signal:      "Lower support-review pattern",
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
    <div className="signal-slot" style={{ borderLeftColor: tone.fg }}>
      <div className="signal-slot-head">
        <HeaderIcon size={13} style={{ color: tone.fg }} aria-hidden="true" />
        <span className="signal-slot-eyebrow">{title}</span>
        <span
          className="signal-slot-chip"
          style={{ background: tone.bg, color: tone.fg, borderColor: tone.border }}
        >
          {abstained ? "abstained" : `confidence: ${prediction.confidence}`}
        </span>
      </div>
      <p className="signal-slot-label">{label}</p>
      {prediction.probability != null && (
        <p className="signal-slot-metric tabular-nums">
          Synthetic class probability: <strong>{(prediction.probability * 100).toFixed(1)}%</strong>
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
    <div className="signal-slot" style={{ borderLeftColor: tone.fg }}>
      <div className="signal-slot-head">
        <HeaderIcon size={13} style={{ color: tone.fg }} aria-hidden="true" />
        <span className="signal-slot-eyebrow">{title}</span>
        <span
          className="signal-slot-chip"
          style={{ background: tone.bg, color: tone.fg, borderColor: tone.border }}
        >
          {abstained ? "abstained" : `confidence: ${prediction.confidence}`}
        </span>
      </div>
      <p className="signal-slot-label">{label}</p>
      {prediction.response_score != null && (
        <div className="signal-slot-bar">
          <ScoreBar score={prediction.response_score} band={prediction.uncertainty_band} tone={tone.fg} />
          <p className="signal-slot-metric tabular-nums">
            Synthetic score: <strong>{prediction.response_score.toFixed(2)}</strong>
            {prediction.uncertainty_band && (
              <span className="signal-slot-band">
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
      Record types used: {present}/{total} | sufficiency: <strong>{ev.sufficiency}</strong>
      {" | "}<code style={{ fontSize: "0.66rem" }}>{prediction.model_version}</code>
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
    case "response_imaging_required_for_response_pattern":
      return "Imaging evidence is required before this response signal is shown.";
    case "response_imaging_only_without_longitudinal_cbc":
      return "Imaging is present, but complete longitudinal CBC context is missing — confidence reduced.";
    case "no_response_signal_imaging_or_longitudinal_cbc_required":
      return "Needs imaging evidence; older traces used this legacy reason.";
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
