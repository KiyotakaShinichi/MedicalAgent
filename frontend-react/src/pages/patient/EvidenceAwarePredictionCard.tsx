import { ShieldCheck, ShieldAlert, Info } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { EmptyPane } from "../../components/ui/Spinner";
import type { EvidenceAwarePrediction } from "../../types/api";

interface Props { prediction: EvidenceAwarePrediction | null | undefined }

const DECISION_LABEL: Record<string, string> = {
  favorable_pattern:     "Favorable response pattern",
  concerning_pattern:    "Concerning response pattern",
  uncertain:             "Uncertain pattern",
  insufficient_evidence: "Insufficient evidence",
};

const DECISION_TONE: Record<string, { fg: string; bg: string; border: string }> = {
  favorable_pattern:     { fg: "#047857", bg: "#ecfdf5", border: "#a7f3d0" },
  concerning_pattern:    { fg: "#b91c1c", bg: "#fef2f2", border: "#fecaca" },
  uncertain:             { fg: "#92400e", bg: "#fffbeb", border: "#fde68a" },
  insufficient_evidence: { fg: "var(--text-faint)", bg: "var(--surface2)", border: "var(--border)" },
};

/**
 * Patient-facing evidence-aware monitoring signal.  Shows the abstention-
 * aware envelope produced by `predict_with_abstention`: which decision
 * the system reached, how confident it was, which modalities it actually
 * had to work with, and — most importantly — when it refused to score.
 *
 * This is NOT a treatment recommendation.  The decision is a synthetic
 * monitoring signal, the claim boundary is always shown, and the
 * "insufficient evidence" path is treated as a first-class outcome rather
 * than a fallback.
 */
export function EvidenceAwarePredictionCard({ prediction }: Props) {
  if (prediction === undefined) return null; // report still loading

  const tone = DECISION_TONE[prediction?.decision ?? "insufficient_evidence"] ?? DECISION_TONE.insufficient_evidence;
  const decisionLabel = DECISION_LABEL[prediction?.decision ?? ""] ?? prediction?.decision ?? "—";
  const abstained = prediction?.evidence?.abstain ?? false;
  const Icon = abstained ? ShieldAlert : ShieldCheck;

  if (!prediction) {
    return (
      <SectionCard title="Evidence-aware monitoring signal" icon={Info}>
        <EmptyPane label="No live model signal available for this patient yet." />
      </SectionCard>
    );
  }

  const present = prediction.evidence.modalities_present;
  const missing = prediction.evidence.modalities_missing;
  const totalModalities = present.length + missing.length;

  return (
    <SectionCard
      title="Evidence-aware monitoring signal"
      icon={Icon}
      meta={
        <span style={{ color: "var(--text-faint)", fontSize: "0.74rem" }}>
          {abstained ? "abstained" : `confidence: ${prediction.confidence}`}
        </span>
      }
      footer={
        <span className="inline-flex items-start gap-1.5">
          <Info size={11} aria-hidden="true" style={{ marginTop: 2, flexShrink: 0 }} />
          <span>{prediction.claim_boundary}</span>
        </span>
      }
    >
      <div
        className="rounded-md border p-3 mb-3"
        style={{ background: tone.bg, borderColor: tone.border }}
      >
        <p
          className="text-[0.72rem] uppercase font-semibold mb-1 tracking-wide"
          style={{ color: tone.fg, opacity: 0.85 }}
        >
          Decision
        </p>
        <p className="text-base font-bold" style={{ color: tone.fg, lineHeight: 1.3 }}>
          {decisionLabel}
        </p>
        {prediction.probability != null && (
          <p className="text-[0.78rem] mt-1 tabular-nums" style={{ color: tone.fg, opacity: 0.85 }}>
            Calibrated probability: <strong>{(prediction.probability * 100).toFixed(1)}%</strong>
            {confidenceModifierNote(prediction) && (
              <span style={{ marginLeft: 8, opacity: 0.7 }}>{confidenceModifierNote(prediction)}</span>
            )}
          </p>
        )}
        {abstained && prediction.evidence.reason && (
          <p className="text-[0.78rem] mt-2" style={{ color: tone.fg, opacity: 0.9 }}>
            <strong>Why we abstained:</strong> {humanReason(prediction.evidence.reason)}
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-start gap-3">
        <div className="flex-1 min-w-[180px]">
          <p className="text-[0.7rem] uppercase font-semibold mb-1.5" style={{ color: "var(--text-faint)" }}>
            Modalities used ({present.length} / {totalModalities})
          </p>
          {present.length === 0 ? (
            <p className="text-[0.78rem]" style={{ color: "var(--text-dim)" }}>None available.</p>
          ) : (
            <ul className="flex flex-wrap gap-1">
              {present.map((m) => (
                <li key={m}>
                  <span
                    className="inline-block text-[0.7rem] px-2 py-0.5 rounded-full font-medium"
                    style={{ background: "var(--surface2)", color: "var(--text)", border: "1px solid var(--border)" }}
                  >
                    {humanModality(m)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        {missing.length > 0 && (
          <div className="flex-1 min-w-[180px]">
            <p className="text-[0.7rem] uppercase font-semibold mb-1.5" style={{ color: "var(--text-faint)" }}>
              Missing
            </p>
            <ul className="flex flex-wrap gap-1">
              {missing.map((m) => (
                <li key={m}>
                  <span
                    className="inline-block text-[0.7rem] px-2 py-0.5 rounded-full"
                    style={{ background: "transparent", color: "var(--text-faint)", border: "1px dashed var(--border)" }}
                  >
                    {humanModality(m)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <p className="text-[0.7rem] mt-3" style={{ color: "var(--text-faint)" }}>
        Model: <code>{prediction.model_version}</code>{" "}
        {prediction.calibrated && <>· isotonic calibration applied</>}
        {" · "}sufficiency: <strong>{prediction.evidence.sufficiency}</strong>
      </p>
    </SectionCard>
  );
}

function humanModality(slug: string): string {
  switch (slug) {
    case "demographics":   return "Demographics";
    case "cbc_pre":        return "Pre-cycle CBC";
    case "cbc_nadir":      return "Nadir CBC";
    case "cbc_recovery":   return "Recovery CBC";
    case "imaging":        return "Imaging";
    case "symptoms":       return "Symptoms";
    case "interventions":  return "Interventions";
    default:               return slug;
  }
}

function humanReason(reason: string): string {
  switch (reason) {
    case "no_response_signal_imaging_or_longitudinal_cbc_required":
      return "Response classification needs imaging or a complete cycle of CBC values.";
    case "missing_minimum_context":
      return "Patient demographics weren't available.";
    case "response_signal_from_single_modality_only":
      return "Only one of imaging or longitudinal CBC was present — confidence reduced.";
    case "toxicity_signal_from_single_modality_only":
      return "Only one of CBC or symptoms was present — confidence reduced.";
    case "no_acute_signal_symptoms_or_nadir_cbc_required":
      return "Urgent-intervention signal needs current symptoms or nadir CBC.";
    case "urgent_signal_from_single_modality_only":
      return "Only one acute signal was present — confidence reduced.";
    default:
      return reason.replace(/_/g, " ");
  }
}

function confidenceModifierNote(prediction: EvidenceAwarePrediction): string {
  const modifier = prediction.evidence.confidence_modifier;
  if (modifier == null || modifier >= 0.99) return "";
  if (modifier === 0) return "";
  return `(shrunk toward prior, modifier ${modifier.toFixed(2)})`;
}
