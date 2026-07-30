import { useState } from "react";
import { Activity, ShieldCheck, FlaskConical, ListChecks, CircleHelp, ChevronRight } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { PatientReport, EvidenceAwarePrediction, HybridPrediction } from "../../types/api";

/**
 * Top-row KPI strip on the patient overview.
 *
 * Four tiles, in order:
 *   1. Monitoring score / evidence status
 *   2. Hybrid monitoring signal (best of response / toxicity slot)
 *   3. Latest CBC status (combined WBC + Hgb at-a-glance)
 *   4. Review queue (count + top item preview)
 *
 * Each tile shows label · value · short caption · sub-row.  Captions are
 * neutral wording — these are not clinical claims.  Footers carry the
 * "synthetic-only engineering signal, not a clinical prediction" line
 * where the underlying value comes from the ML stack.
 */
interface Props {
  report: PatientReport;
}

type Tone = "neutral" | "good" | "watch" | "concern";

interface Sub {
  label: string;
  value: string;
}

interface Tile {
  label: string;
  icon: LucideIcon;
  value: string;
  unit?: string;
  caption: string;
  tone: Tone;
  subs?: Sub[];
  footer?: string;
  meaning: string;
  calculation: string;
  progress?: number;
  breakdown?: Sub[];
  nextSteps?: string[];
}

function toneColor(tone: Tone): string {
  switch (tone) {
    case "good": return "#059669";
    case "watch": return "#d97706";
    case "concern": return "#dc2626";
    default: return "var(--text-strong)";
  }
}

function reviewStatusTile(report: PatientReport): Tile {
  const breakdown = report.multimodal_assessment?.score_breakdown;
  const reasons = report.ai_summary?.review_reasons ?? [];
  const count = reasons.length;
  const tone: Tone = count === 0 ? "good" : count >= 5 ? "watch" : "neutral";
  const breakdownRows: Sub[] | undefined = breakdown ? [
    { label: "Urgent-review rule matches", value: String(breakdown.urgent_review_flags) },
    { label: "Watch-rule matches", value: String(breakdown.watch_flags) },
    {
      label: "Highest recorded symptom severity",
      value: breakdown.peak_recorded_symptom_severity == null
        ? "Not available"
        : `${breakdown.peak_recorded_symptom_severity}/10`,
    },
  ] : undefined;
  return {
    label: "Items for review",
    icon: Activity,
    value: String(count),
    unit: count === 1 ? "item" : "items",
    caption: count === 0 ? "No rule-based review item is queued" : "For care-team discussion",
    tone,
    footer: "A workflow count, not a diagnosis, severity grade, or emergency score.",
    meaning: count === 0
      ? "No current record item matched the portal's review rules. This is not reassurance about health or treatment response."
      : `${count} available record item${count === 1 ? "" : "s"} matched a rule for care-team discussion. This count does not measure cancer status, treatment success, prognosis, or urgency.`,
    calculation: "Counts distinct review reasons produced from available lab, symptom, imaging, and treatment-context rules. Missing or newly added records can change the count.",
    breakdown: breakdownRows,
    nextSteps: report.multimodal_assessment?.patient_next_steps ?? [
      "Check that recent symptoms, CBC values, medications, and imaging dates are complete.",
      "Review the queue and prepare questions about its record items for your care team.",
      "Use NLCare to add missing records; do not change treatment based on portal indicators.",
    ],
  };
}

// ─── Hybrid signal: pick the most informative slot for the headline ─────

const DECISION_LABEL: Record<string, string> = {
  favorable_pattern:        "Grouped with favorable synthetic examples",
  concerning_pattern:       "Grouped with concerning synthetic examples",
  uncertain:                "Synthetic grouping is uncertain",
  insufficient_evidence:    "Insufficient evidence",
  strong_response_signal:   "Higher synthetic response-signal group",
  moderate_response_signal: "Middle synthetic response-signal group",
  weak_response_signal:     "Lower synthetic response-signal group",
  high_toxicity_signal:     "Higher synthetic review-signal group",
  moderate_toxicity_signal: "Middle synthetic review-signal group",
  low_toxicity_signal:      "Lower synthetic review-signal group",
};

const DECISION_TONE: Record<string, Tone> = {
  favorable_pattern:        "good",
  concerning_pattern:       "concern",
  uncertain:                "watch",
  insufficient_evidence:    "neutral",
  strong_response_signal:   "good",
  moderate_response_signal: "watch",
  weak_response_signal:     "concern",
  low_toxicity_signal:      "good",
  moderate_toxicity_signal: "watch",
  high_toxicity_signal:     "concern",
};

function hybridHeadline(hybrid: HybridPrediction | null | undefined): {
  label: string;
  tone: Tone;
  prob: number | null;
  confidence: string | null;
  modalities: string | null;
} {
  if (!hybrid) {
    return { label: "No live signal", tone: "neutral", prob: null, confidence: null, modalities: null };
  }
  // Headline order of priority: response classification → toxicity → response score regression.
  const cls = hybrid.classification;
  const tox = hybrid.toxicity;
  const choice: EvidenceAwarePrediction | null =
    cls?.decision && cls.decision !== "insufficient_evidence" ? cls :
    tox?.decision && tox.decision !== "insufficient_evidence" ? tox :
    cls ?? tox ?? null;
  if (!choice) {
    return { label: "Insufficient evidence", tone: "neutral", prob: null, confidence: null, modalities: null };
  }
  const ev = choice.evidence;
  const used = ev ? `${ev.modalities_present.length}/${ev.modalities_present.length + ev.modalities_missing.length}` : null;
  return {
    label: DECISION_LABEL[choice.decision] ?? choice.decision,
    tone: DECISION_TONE[choice.decision] ?? "neutral",
    prob: typeof choice.probability === "number" ? choice.probability : null,
    confidence: choice.confidence ?? null,
    modalities: used,
  };
}

function hybridSignalTile(report: PatientReport): Tile {
  const h = hybridHeadline(report.hybrid_prediction);
  const subs: Sub[] = [];
  if (h.label && h.label !== "No live signal") subs.push({ label: "Synthetic grouping", value: h.label });
  if (h.prob != null) subs.push({ label: "Model confidence", value: `${(h.prob * 100).toFixed(1)}%` });
  if (h.confidence) subs.push({ label: "Confidence", value: h.confidence });
  if (h.modalities) subs.push({ label: "Modalities", value: h.modalities });
  return {
    label: "Synthetic model pattern",
    icon: ShieldCheck,
    value: h.tone === "neutral" ? "Not available" : "Available for review",
    caption: h.tone === "neutral" ? "Awaiting more data" : "Engineering output only",
    tone: "neutral",
    subs,
    footer: "Synthetic engineering signal. Not a personal outcome probability.",
    meaning: "This groups the available record pattern against simulator-built examples. Probability is model confidence in that synthetic class, not the patient's chance of improving.",
    calculation: "When both heads are available, the legacy hybrid score uses 65% calibrated classification probability and 35% normalized regression output. Missing heads are omitted rather than invented.",
  };
}

// ─── Latest CBC tile: combined WBC + Hgb at-a-glance ────────────────────

function wbcTone(wbc: number | null): Tone {
  if (wbc == null) return "neutral";
  if (wbc < 2.0) return "concern";
  if (wbc < 4.0 || wbc > 11.0) return "watch";
  return "good";
}
function hgbTone(hgb: number | null): Tone {
  if (hgb == null) return "neutral";
  if (hgb < 8.0) return "concern";
  if (hgb < 10.0) return "watch";
  return "good";
}
function pltTone(plt: number | null): Tone {
  if (plt == null) return "neutral";
  if (plt < 50) return "concern";
  if (plt < 150 || plt > 450) return "watch";
  return "good";
}

function worstTone(...tones: Tone[]): Tone {
  if (tones.includes("concern")) return "concern";
  if (tones.includes("watch")) return "watch";
  if (tones.includes("good")) return "good";
  return "neutral";
}

function latestCbcTile(report: PatientReport): Tile {
  const labs = report.latest_labs ?? { wbc: null, hemoglobin: null, platelets: null };
  const tone = worstTone(wbcTone(labs.wbc), hgbTone(labs.hemoglobin), pltTone(labs.platelets));
  const summary: Tile["value"] =
    tone === "concern" ? "Outside demo band" :
    tone === "watch" ? "Near demo band" :
    tone === "good" ? "Inside demo band" : "—";
  const subs: Sub[] = [];
  if (labs.wbc != null)       subs.push({ label: "WBC", value: `${labs.wbc.toFixed(1)} ×10³/uL` });
  if (labs.hemoglobin != null) subs.push({ label: "Hgb", value: `${labs.hemoglobin.toFixed(1)} g/dL` });
  if (labs.platelets != null) subs.push({ label: "Plt", value: `${Math.round(labs.platelets)} ×10³/uL` });
  return {
    label: "Latest CBC",
    icon: FlaskConical,
    value: summary,
    caption: tone === "neutral" ? "No lab on file" : "Demonstration bands use population defaults",
    tone,
    subs,
    footer: "Reference ranges are not personalised. Discuss with your care team.",
    meaning: "These are the latest recorded CBC values. The summary only compares them with fixed demonstration reference bands; it does not diagnose a condition.",
    calculation: "WBC, hemoglobin, and platelets are checked independently against population-default bands. The most review-oriented band becomes the card summary.",
  };
}

// ─── Review queue tile: count of care-team items + top preview ──────────

function recordCoverageTile(report: PatientReport): Tile {
  const items = report.data_availability?.items ?? [];
  const available = items.filter((item) => item.status === "available");
  const incomplete = items.filter((item) => item.status !== "available");
  const total = items.length;
  return {
    label: "Record coverage",
    icon: ListChecks,
    value: total ? `${available.length}/${total}` : "Not assessed",
    unit: total ? "areas available" : undefined,
    caption: incomplete.length ? `${incomplete.length} area${incomplete.length === 1 ? "" : "s"} incomplete` : "All tracked areas represented",
    tone: incomplete.length ? "watch" : "good",
    subs: incomplete.slice(0, 3).map((item) => ({ label: item.name, value: item.status.replaceAll("_", " ") })),
    meaning: report.data_availability?.patient_friendly_summary
      ?? "Shows whether the portal has enough repeated records to summarize each tracked area. It does not measure health or clinical completeness.",
    calculation: "Checks six portal areas: repeated CBC values, treatment-cycle alignment, symptoms, imaging, model availability, and timeline depth. Each area is marked available, incomplete, or missing.",
  };
}

function buildTiles(report: PatientReport): Tile[] {
  return [
    reviewStatusTile(report),
    hybridSignalTile(report),
    latestCbcTile(report),
    recordCoverageTile(report),
  ];
}

export function PatientKpiStrip({ report }: Props) {
  const tiles = buildTiles(report);
  const [activeTileIndex, setActiveTileIndex] = useState(0);
  const activeTile = tiles[activeTileIndex] ?? tiles[0];
  const activeColor = toneColor(activeTile.tone);
  return (
    <section className="patient-kpi-section" aria-label="Latest monitoring indicators">
      <div className="patient-kpi-strip" role="list">
        {tiles.map((t, index) => {
          const Icon = t.icon;
          const color = toneColor(t.tone);
          const isActive = index === activeTileIndex;
          return (
            <div
              className={`patient-kpi-card${isActive ? " is-active" : ""}`}
              data-tone={t.tone}
              role="listitem"
              key={t.label}
            >
              <div className="patient-kpi-card-head">
                <span className="patient-kpi-label">{t.label}</span>
                <span className="patient-kpi-icon" aria-hidden="true">
                  <Icon size={14} />
                </span>
              </div>
              <div className="patient-kpi-value" style={{ color }}>
                <strong>{t.value}</strong>
                {t.unit && <span className="patient-kpi-unit">{t.unit}</span>}
              </div>
              {typeof t.progress === "number" && (
                <div className="patient-kpi-progress" aria-label={`${t.progress} out of 100`}>
                  <span style={{ width: `${Math.max(0, Math.min(100, t.progress))}%`, background: color }} />
                </div>
              )}
              <span
                className="patient-kpi-caption"
                title={t.caption}
                style={{ color: t.tone === "neutral" ? "var(--text-faint)" : color }}
              >
                {t.caption}
              </span>
              {t.subs && t.subs.length > 0 && (
                <ul className="patient-kpi-subs" aria-label={`${t.label} detail`}>
                  {t.subs.map((s) => (
                    <li key={s.label}>
                      <span className="patient-kpi-sub-label">{s.label}</span>
                      <span className="patient-kpi-sub-value">{s.value}</span>
                    </li>
                  ))}
                </ul>
              )}
              {t.footer && (
                <span className="patient-kpi-footer">
                  <span className="patient-kpi-footer-dot" aria-hidden="true" />
                  <span>{t.footer}</span>
                </span>
              )}
              <button
                type="button"
                className="patient-kpi-explain-button"
                aria-expanded={isActive}
                aria-controls="patient-kpi-explainer"
                onClick={() => setActiveTileIndex(index)}
              >
                <CircleHelp size={13} aria-hidden="true" />
                Explain this indicator
                <ChevronRight size={13} aria-hidden="true" />
              </button>
            </div>
          );
        })}
      </div>

      <div id="patient-kpi-explainer" className="patient-kpi-explainer" aria-live="polite">
        <div className="patient-kpi-explainer-head">
          <div>
            <span className="patient-kpi-explainer-kicker">Understanding your record</span>
            <h3>{activeTile.label}: <span style={{ color: activeColor }}>{activeTile.value}{activeTile.unit ? ` ${activeTile.unit}` : ""}</span></h3>
          </div>
          <span className="patient-kpi-boundary">Record context only</span>
        </div>
        <div className="patient-kpi-explainer-grid">
          <div className="patient-kpi-explainer-copy">
            <h4>What it means</h4>
            <p>{activeTile.meaning}</p>
          </div>
          <div className="patient-kpi-explainer-copy">
            <h4>How NLCare calculated it</h4>
            <p>{activeTile.calculation}</p>
            {activeTile.breakdown && (
              <dl className="patient-kpi-breakdown">
                {activeTile.breakdown.map((row) => (
                  <div key={row.label}>
                    <dt>{row.label}</dt>
                    <dd>{row.value}</dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
          <div className="patient-kpi-explainer-copy patient-kpi-next-steps">
            <h4>Safe next steps</h4>
            <ul>
              {(activeTile.nextSteps ?? [
                "Check that the underlying records are complete and dated correctly.",
                "Use this indicator to prepare questions for your care team, not to make treatment decisions.",
              ]).map((step) => <li key={step}>{step}</li>)}
            </ul>
          </div>
        </div>
      </div>
      <details className="patient-score-boundary">
        <summary>Why NLCare does not show a health score</summary>
        <p>
          The earlier 0-100 monitoring context index was removed from patient headlines. It combined
          rule matches and synthetic model availability, so it could be mistaken for cancer status,
          treatment response, or prognosis. The patient view now shows the underlying record items,
          data coverage, and synthetic engineering output separately.
        </p>
      </details>
    </section>
  );
}
