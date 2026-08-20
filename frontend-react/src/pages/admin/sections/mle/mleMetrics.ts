/**
 * Metric grading for the MLE evidence panels.
 *
 * This logic was previously written inline inside two `.map()` callbacks in
 * `MleSection`, keyed off string comparisons against the visible label
 * ("Brier score", "MAE", …). That made it untestable and meant renaming a
 * label silently flipped a metric's grading direction. Here the direction is
 * declared as data alongside the threshold.
 *
 * These bands are engineering heuristics for a synthetic-data PoC. They are
 * not clinical validation thresholds, and nothing here should be read as one.
 */

export type MetricTone = "green" | "amber" | "red" | "muted";

/**
 * Narrow an artifact envelope's `result` field, which the API types as
 * `unknown` because its shape is owned by a training script rather than the
 * schema. Anything that is not a plain object becomes `undefined`, so a
 * malformed payload renders as "no report" instead of throwing on field access.
 */
export function toRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

/** Whether a larger number is better (AUROC) or worse (Brier, MAE, RMSE). */
export type MetricDirection = "higher-is-better" | "lower-is-better";

export interface MetricSpec {
  label: string;
  direction: MetricDirection;
  /** Value at or beyond which the metric is green. `null` = never graded. */
  threshold: number | null;
  /** Multiplier applied to `threshold` to find the amber/red boundary. */
  amberFactor: number;
  sub?: string;
}

/**
 * Parse a metric that may arrive as a number, a numeric string, null, or a
 * non-numeric label (`best_classifier` is a model name, not a score).
 *
 * Returns `null` rather than `NaN` so callers cannot accidentally propagate
 * `NaN` into a rendered "NaN%".
 */
export function parseMetricValue(raw: unknown): number | null {
  if (typeof raw === "number") return Number.isFinite(raw) ? raw : null;
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Grade a value against a spec.
 *
 * An ungraded or unparseable metric is **muted, never green** — the same rule
 * the safety surface uses. "We could not measure this" must not look like
 * "this passed".
 */
export function gradeMetric(raw: unknown, spec: MetricSpec): MetricTone {
  const value = parseMetricValue(raw);
  if (value === null || spec.threshold === null) return "muted";

  if (spec.direction === "lower-is-better") {
    if (value <= spec.threshold) return "green";
    return value <= spec.threshold * spec.amberFactor ? "amber" : "red";
  }
  if (value >= spec.threshold) return "green";
  return value >= spec.threshold * spec.amberFactor ? "amber" : "red";
}

/** Value as shown on a MetricCard; `null` renders as the card's own dash. */
export function displayMetricValue(raw: unknown): string | null {
  if (raw === null || raw === undefined) return null;
  return String(raw);
}

/**
 * Synthetic training report. `test_patients` and the two `best_*` fields are
 * descriptive, not scored, so they carry a null threshold.
 */
export const TRAINING_REPORT_METRICS: readonly MetricSpec[] = [
  { label: "Test patients", direction: "higher-is-better", threshold: null, amberFactor: 1 },
  { label: "Best classifier", direction: "higher-is-better", threshold: null, amberFactor: 1 },
  { label: "Best regressor", direction: "higher-is-better", threshold: null, amberFactor: 1 },
  { label: "AUROC", direction: "higher-is-better", threshold: 0.85, amberFactor: 0.85 },
  { label: "Brier score", direction: "lower-is-better", threshold: 0.1, amberFactor: 2 },
  { label: "MAE", direction: "lower-is-better", threshold: 0.1, amberFactor: 2 },
  { label: "RMSE", direction: "lower-is-better", threshold: 0.15, amberFactor: 2 },
] as const;

/** Frozen synthetic holdout split — looser bands than the training report. */
export const HOLDOUT_METRICS: readonly MetricSpec[] = [
  { label: "AUROC", direction: "higher-is-better", threshold: 0.8, amberFactor: 0.85, sub: "Higher = better discrimination" },
  { label: "Brier", direction: "lower-is-better", threshold: 0.12, amberFactor: 1.6, sub: "Lower = better calibration" },
  { label: "Sensitivity", direction: "higher-is-better", threshold: 0.75, amberFactor: 0.85, sub: "Higher = fewer missed positives" },
  { label: "MAE", direction: "lower-is-better", threshold: 0.12, amberFactor: 1.6, sub: "Lower = better regression fit" },
] as const;

/** Field on the artifact that supplies each spec's value. */
export const TRAINING_REPORT_FIELDS: Record<string, string> = {
  "Test patients": "test_patients",
  "Best classifier": "best_classifier",
  "Best regressor": "best_regressor",
  AUROC: "auroc",
  "Brier score": "brier_score",
  MAE: "mae",
  RMSE: "rmse",
};

export const HOLDOUT_FIELDS: Record<string, string> = {
  AUROC: "auroc",
  Brier: "brier_score",
  Sensitivity: "sensitivity",
  MAE: "mae",
};

/**
 * Values fed to the metric glossary. Uses `parseMetricValue` so a missing or
 * non-numeric field becomes `null` instead of `NaN`.
 *
 * The previous implementation used `parseFloat(String(x)) || null`, which
 * mapped a legitimate **0** to `null` because `0` is falsy — a genuinely
 * measured zero was displayed as "not measured".
 */
export function buildGlossaryValues(
  result: Record<string, unknown> | undefined,
): Record<string, number | null> {
  if (!result) return {};
  return {
    AUROC: parseMetricValue(result.auroc),
    "Brier Score": parseMetricValue(result.brier_score),
    ECE: parseMetricValue(result.ece),
    "Sensitivity (Recall)": parseMetricValue(result.sensitivity),
    "MAE (Regression)": parseMetricValue(result.mae),
  };
}
