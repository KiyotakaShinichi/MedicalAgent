/**
 * Formatting and status-mapping helpers shared by the Safety & Evaluation
 * Center blocks.
 *
 * These were previously duplicated inside SafetyCenterSection.tsx. They are
 * pure functions with no React dependency, which is what makes the individual
 * blocks cheap to unit test.
 */
import type { EvalIntegrityStatus } from "../../../../components/ui/EvalIntegrityFooter";

export type BadgeTone = "green" | "amber" | "red" | "muted";

/**
 * Map a backend status string onto a badge tone.
 *
 * Unknown statuses deliberately fall through to "muted" rather than "green".
 * In a safety surface, an unrecognised state must never read as a pass.
 */
export function statusBadge(status: string | undefined): BadgeTone {
  if (!status) return "muted";
  if (["passed", "strong", "available", "ok"].includes(status)) return "green";
  if (["acceptable", "watch", "needs_attention", "partial"].includes(status)) return "amber";
  if (["failed", "unideal", "error"].includes(status)) return "red";
  return "muted";
}

/** Percentage with no decimals; em dash for absent values. */
export function fmtRate(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

/** Fixed-precision score; em dash for absent values. */
export function fmtScore(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(digits);
}

/**
 * Tone for a metric where higher is better. Absent stays "muted" — an
 * unmeasured metric is not a passing metric.
 */
export function highIsGood(
  value: number | null | undefined,
  good = 0.9,
  warn = 0.8,
): BadgeTone {
  if (value === null || value === undefined) return "muted";
  if (value >= good) return "green";
  if (value >= warn) return "amber";
  return "red";
}

/** Tone for a metric where lower is better (leak rates, unsafe rates). */
export function lowIsGood(
  value: number | null | undefined,
  good = 0.0,
  warn = 0.02,
): BadgeTone {
  if (value === null || value === undefined) return "muted";
  if (value <= good) return "green";
  if (value <= warn) return "amber";
  return "red";
}

export function coerceIntegrityStatus(value: string | undefined): EvalIntegrityStatus {
  switch (value) {
    case "passed":
    case "acceptable":
    case "needs_attention":
    case "failed":
    case "skipped":
    case "missing":
      return value;
    default:
      return "unknown";
  }
}

// ── Loosely typed artifact readers ───────────────────────────────────────────
// Some eval artifacts arrive as `Record<string, unknown>` because their shape
// is owned by an evaluation script rather than the API schema. These readers
// are the runtime trust boundary for that data: anything that isn't the
// expected primitive becomes `undefined` rather than being cast blindly.

export function readRecord(
  record: Record<string, unknown> | undefined,
  key: string,
): Record<string, unknown> | undefined {
  const value = record?.[key];
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

export function readString(
  record: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  const value = record?.[key];
  return typeof value === "string" ? value : undefined;
}

export function readNumber(
  record: Record<string, unknown> | undefined,
  key: string,
): number | undefined {
  const value = record?.[key];
  // NaN is not a usable metric — treat it as absent so it renders as "—"
  // instead of "NaN%".
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function readArray(
  record: Record<string, unknown> | undefined,
  key: string,
): unknown[] {
  const value = record?.[key];
  return Array.isArray(value) ? value : [];
}
