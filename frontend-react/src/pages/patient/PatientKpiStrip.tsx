import { Activity, ShieldCheck, FlaskConical, ListChecks } from "lucide-react";
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
}

function toneColor(tone: Tone): string {
  switch (tone) {
    case "good": return "#059669";
    case "watch": return "#d97706";
    case "concern": return "#dc2626";
    default: return "var(--text-strong)";
  }
}

function monitoringScoreTile(report: PatientReport): Tile {
  const score = report.monitoring_score;
  const tone: Tone =
    score == null ? "neutral" :
    score >= 70 ? "good" :
    score >= 40 ? "watch" : "concern";
  const caption =
    score == null ? "Insufficient data" :
    score >= 70 ? "Stable today" :
    score >= 40 ? "Some signals to watch" : "Clinician review suggested";
  return {
    label: "Monitoring score",
    icon: Activity,
    value: score != null ? String(score) : "—",
    unit: score != null ? "/ 100" : undefined,
    caption,
    tone,
    footer: "Synthetic engineering signal · Not a clinical prediction · For clinician review.",
  };
}

// ─── Hybrid signal: pick the most informative slot for the headline ─────

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
  if (h.prob != null) subs.push({ label: "Probability", value: `${(h.prob * 100).toFixed(1)}%` });
  if (h.confidence) subs.push({ label: "Confidence", value: h.confidence });
  if (h.modalities) subs.push({ label: "Modalities", value: h.modalities });
  return {
    label: "Hybrid monitoring signal",
    icon: ShieldCheck,
    value: h.label,
    caption: h.tone === "neutral" ? "Awaiting more data" : "Tracked for clinician review",
    tone: h.tone,
    subs,
    footer: "Synthetic engineering signal · Not a clinical prediction · For clinician review.",
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
    tone === "concern" ? "Below reference" :
    tone === "watch" ? "Borderline" :
    tone === "good" ? "Within reference" : "—";
  const subs: Sub[] = [];
  if (labs.wbc != null)       subs.push({ label: "WBC", value: `${labs.wbc.toFixed(1)} ×10³/uL` });
  if (labs.hemoglobin != null) subs.push({ label: "Hgb", value: `${labs.hemoglobin.toFixed(1)} g/dL` });
  if (labs.platelets != null) subs.push({ label: "Plt", value: `${Math.round(labs.platelets)} ×10³/uL` });
  return {
    label: "Latest CBC",
    icon: FlaskConical,
    value: summary,
    caption: tone === "neutral" ? "No lab on file" : "Reference bands are population defaults",
    tone,
    subs,
    footer: "Reference ranges are not personalised. Discuss with your care team.",
  };
}

// ─── Review queue tile: count of care-team items + top preview ──────────

function reviewQueueTile(report: PatientReport): Tile {
  const reasons = report.ai_summary?.review_reasons ?? [];
  const count = reasons.length;
  const tone: Tone = count === 0 ? "good" : count >= 5 ? "watch" : "neutral";
  const topPreview = reasons[0] ?? null;
  return {
    label: "Review queue",
    icon: ListChecks,
    value: String(count),
    unit: count === 1 ? "item" : "items",
    caption: count === 0 ? "Nothing flagged today" : (topPreview ?? "Pending review"),
    tone,
  };
}

function buildTiles(report: PatientReport): Tile[] {
  return [
    monitoringScoreTile(report),
    hybridSignalTile(report),
    latestCbcTile(report),
    reviewQueueTile(report),
  ];
}

export function PatientKpiStrip({ report }: Props) {
  const tiles = buildTiles(report);
  return (
    <div className="patient-kpi-strip" role="list" aria-label="Latest monitoring indicators">
      {tiles.map((t) => {
        const Icon = t.icon;
        const color = toneColor(t.tone);
        return (
          <div className="patient-kpi-card" role="listitem" key={t.label}>
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
          </div>
        );
      })}
    </div>
  );
}
