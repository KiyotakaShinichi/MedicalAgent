/* eslint-disable react-refresh/only-export-components */

export interface Band {
  label: string;
  range: string;
  color: "green" | "amber" | "red";
  note?: string;
}

export interface MetricSpec {
  name: string;
  description: string;
  whyItMatters: string;
  ideal: string;
  warning: string;
  bad: string;
  clinicalNote?: string;
  bands: Band[];
}

const BAND_STYLES: Record<string, React.CSSProperties> = {
  green: { background: "rgba(16,185,129,0.10)", borderColor: "rgba(16,185,129,0.25)", color: "var(--green)" },
  amber: { background: "rgba(245,158,11,0.10)", borderColor: "rgba(245,158,11,0.25)", color: "var(--amber)" },
  red:   { background: "rgba(244,63,94,0.10)",  borderColor: "rgba(244,63,94,0.25)",  color: "var(--rose)" },
};

export function MetricInterpretation({ spec, value }: { spec: MetricSpec; value?: number | null }) {
  return (
    <div className="rounded-lg border p-4" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
      <div className="flex items-start justify-between gap-2 mb-2">
        <div>
          <p className="text-sm font-semibold" style={{ color: "var(--text)" }}>{spec.name}</p>
          <p className="text-xs mt-0.5" style={{ color: "var(--text-dim)" }}>{spec.description}</p>
        </div>
        {value != null && (
          <span className="text-lg font-bold tabular-nums flex-shrink-0" style={{ color: "var(--text)" }}>
            {value < 1 && value > 0 ? value.toFixed(3) : value.toFixed(1)}
          </span>
        )}
      </div>

      <p className="text-xs mb-3 italic" style={{ color: "var(--text-faint)" }}>{spec.whyItMatters}</p>

      <div className="flex gap-2 flex-wrap mb-2">
        {spec.bands.map((b, i) => (
          <div
            key={i}
            className="flex-1 min-w-[80px] rounded-md border px-2 py-1.5 text-center"
            style={BAND_STYLES[b.color]}
          >
            <p className="text-xs font-bold" style={{ color: BAND_STYLES[b.color].color as string }}>{b.label}</p>
            <p className="text-xs font-mono">{b.range}</p>
            {b.note && <p className="text-xs opacity-80 mt-0.5">{b.note}</p>}
          </div>
        ))}
      </div>

      {spec.clinicalNote && (
        <p className="text-xs px-2 py-1.5 rounded-md border mt-1" style={{
          background: "rgba(59,130,246,0.07)", borderColor: "rgba(59,130,246,0.22)", color: "#93c5fd"
        }}>
          ℹ {spec.clinicalNote}
        </p>
      )}
    </div>
  );
}

export function MetricGlossary({ specs, values }: { specs: MetricSpec[]; values?: Record<string, number | null> }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {specs.map((spec) => (
        <MetricInterpretation key={spec.name} spec={spec} value={values?.[spec.name]} />
      ))}
    </div>
  );
}

// ─── Standard metric specs ───────────────────────────────────────────────────

export const AUROC_SPEC: MetricSpec = {
  name: "AUROC",
  description: "Area Under the Receiver Operating Characteristic Curve. Probability that the model ranks a positive case above a negative case.",
  whyItMatters: "Higher is better. Threshold-independent summary of discrimination power. 0.5 = random, 1.0 = perfect.",
  ideal: "≥ 0.85",
  warning: "0.70–0.85",
  bad: "< 0.70",
  clinicalNote: "On synthetic data, high AUROC is expected. What matters most is holdout and external validation AUROC.",
  bands: [
    { label: "Strong", range: "≥ 0.85", color: "green" },
    { label: "Acceptable", range: "0.70–0.85", color: "amber" },
    { label: "Weak", range: "< 0.70", color: "red", note: "near random" },
  ],
};

export const PRAUC_SPEC: MetricSpec = {
  name: "PR-AUC",
  description: "Area Under the Precision-Recall Curve. More informative than AUROC when positive class is rare.",
  whyItMatters: "On imbalanced datasets (uncommon response), PR-AUC reveals whether the model actually captures positives well.",
  ideal: "≥ 0.75",
  warning: "0.55–0.75",
  bad: "< 0.55",
  bands: [
    { label: "Strong", range: "≥ 0.75", color: "green" },
    { label: "Acceptable", range: "0.55–0.75", color: "amber" },
    { label: "Weak", range: "< 0.55", color: "red" },
  ],
};

export const BRIER_SPEC: MetricSpec = {
  name: "Brier Score",
  description: "Mean squared error between predicted probability and true outcome (0 or 1). Lower is better.",
  whyItMatters: "Measures calibration quality. A score of 0.25 = as bad as always predicting 0.5 regardless of features.",
  ideal: "< 0.10",
  warning: "0.10–0.20",
  bad: "> 0.20",
  clinicalNote: "Brier score penalises overconfident wrong predictions. Aim for < 0.10 on synthetic, check on holdout.",
  bands: [
    { label: "Well-calibrated", range: "< 0.10", color: "green" },
    { label: "Acceptable", range: "0.10–0.20", color: "amber" },
    { label: "Poor", range: "> 0.20", color: "red", note: "≈ random at 0.25" },
  ],
};

export const ECE_SPEC: MetricSpec = {
  name: "ECE",
  description: "Expected Calibration Error. Average gap between predicted probability and observed frequency across confidence bins.",
  whyItMatters: "A model that says 0.8 should be right ~80% of the time. ECE measures how far off those confidence estimates are.",
  ideal: "< 0.05",
  warning: "0.05–0.12",
  bad: "> 0.12",
  clinicalNote: "Isotonic regression or Platt scaling typically reduces ECE. Compare raw vs calibrated ECE in the calibration section.",
  bands: [
    { label: "Well-calibrated", range: "< 0.05", color: "green" },
    { label: "Acceptable", range: "0.05–0.12", color: "amber" },
    { label: "Poorly calibrated", range: "> 0.12", color: "red" },
  ],
};

export const SENSITIVITY_SPEC: MetricSpec = {
  name: "Sensitivity (Recall)",
  description: "Fraction of true positives correctly identified. TP / (TP + FN).",
  whyItMatters: "In cancer monitoring, a missed positive (false negative) is more dangerous than a false alarm. Sensitivity should be prioritised.",
  ideal: "≥ 0.80",
  warning: "0.65–0.80",
  bad: "< 0.65",
  clinicalNote: "This system uses a cost-sensitive threshold: FN is assumed costlier than FP. The operating threshold is set to favour recall over precision.",
  bands: [
    { label: "Strong", range: "≥ 0.80", color: "green", note: "FN cost controlled" },
    { label: "Acceptable", range: "0.65–0.80", color: "amber" },
    { label: "High FN risk", range: "< 0.65", color: "red" },
  ],
};

export const SPECIFICITY_SPEC: MetricSpec = {
  name: "Specificity",
  description: "Fraction of true negatives correctly identified. TN / (TN + FP).",
  whyItMatters: "High false positive rate increases unnecessary clinician review burden. Balance with sensitivity.",
  ideal: "≥ 0.75",
  warning: "0.60–0.75",
  bad: "< 0.60",
  bands: [
    { label: "Good", range: "≥ 0.75", color: "green" },
    { label: "Acceptable", range: "0.60–0.75", color: "amber" },
    { label: "High FP burden", range: "< 0.60", color: "red" },
  ],
};

export const FNR_SPEC: MetricSpec = {
  name: "False Negative Rate",
  description: "Fraction of true positives missed. FN / (FN + TP). Complement of sensitivity.",
  whyItMatters: "The primary risk metric: a missed positive case in cancer monitoring can delay critical intervention.",
  ideal: "< 0.20",
  warning: "0.20–0.35",
  bad: "> 0.35",
  clinicalNote: "FNR is the clinical safety floor. The chosen threshold explicitly trades some FP for lower FNR.",
  bands: [
    { label: "Acceptable", range: "< 0.20", color: "green", note: "FN risk low" },
    { label: "Watch", range: "0.20–0.35", color: "amber" },
    { label: "Unsafe", range: "> 0.35", color: "red", note: "too many missed" },
  ],
};

export const MAE_SPEC: MetricSpec = {
  name: "MAE (Regression)",
  description: "Mean Absolute Error for the continuous response-score regressor.",
  whyItMatters: "Lower is better. Measures average absolute deviation from the true response score (0–1 scale).",
  ideal: "< 0.10",
  warning: "0.10–0.18",
  bad: "> 0.18",
  bands: [
    { label: "Good", range: "< 0.10", color: "green" },
    { label: "Acceptable", range: "0.10–0.18", color: "amber" },
    { label: "Poor", range: "> 0.18", color: "red" },
  ],
};

export const ALL_METRIC_SPECS = [
  AUROC_SPEC, PRAUC_SPEC, BRIER_SPEC, ECE_SPEC,
  SENSITIVITY_SPEC, SPECIFICITY_SPEC, FNR_SPEC, MAE_SPEC,
];
