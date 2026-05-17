import type { PatientReport, TimelineEvent } from "../../types/api";

export interface ReviewItem {
  title: string;
  detail: string;
  tone: "warning" | "info";
}

const SEVERITY_RANK: Record<string, number> = {
  critical: 4,
  high: 3,
  warning: 3,
  moderate: 2,
  medium: 2,
  elevated: 2,
  low: 1,
  info: 0,
};

function rank(sev: string | null | undefined): number {
  if (!sev) return 0;
  return SEVERITY_RANK[sev.toLowerCase()] ?? 0;
}

/**
 * Build a compact "next clinician review" list from a PatientReport without
 * fetching anything new.  Pulls from:
 *   - `ai_summary.review_reasons` (top 2 model-flagged items)
 *   - multimodal signals whose status mentions concern / warning / alert
 *   - timeline events ranked moderate or above (top 2 most recent)
 *
 * Returns up to 4 deduplicated items, ordered review_reasons → signals →
 * timeline, with warning tone for high-severity rows and info tone for the
 * rest.
 */
export function deriveReviewItems(report: PatientReport): ReviewItem[] {
  const out: ReviewItem[] = [];

  const reasons = report.ai_summary?.review_reasons ?? [];
  for (const r of reasons.slice(0, 2)) {
    out.push({ title: "AI flagged for discussion", detail: r, tone: "warning" });
  }

  const signals = report.multimodal_assessment?.signals ?? {};
  for (const [key, sig] of Object.entries(signals)) {
    if (!sig) continue;
    const status = sig.status?.toLowerCase() ?? "";
    if (status.includes("concern") || status.includes("warning") || status.includes("alert")) {
      out.push({
        title: key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        detail: sig.message,
        tone: "warning",
      });
    }
  }

  const sortedTimeline: TimelineEvent[] = [...(report.timeline ?? [])]
    .filter((ev) => rank(ev.severity) >= 2)
    .sort((a, b) => (a.date < b.date ? 1 : -1));
  for (const ev of sortedTimeline.slice(0, 2)) {
    out.push({
      title: ev.title,
      detail: ev.summary || ev.detail?.message || ev.date,
      tone: rank(ev.severity) >= 3 ? "warning" : "info",
    });
  }

  const seen = new Set<string>();
  const unique: ReviewItem[] = [];
  for (const item of out) {
    const k = `${item.title}::${item.detail}`;
    if (seen.has(k)) continue;
    seen.add(k);
    unique.push(item);
  }
  return unique.slice(0, 4);
}
