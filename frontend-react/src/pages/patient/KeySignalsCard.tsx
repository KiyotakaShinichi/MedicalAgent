import { CheckCircle, Info } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import type { AiSummary } from "../../types/api";

interface Props { summary: AiSummary | null }

function toArray(v: string | string[] | undefined): string[] {
  if (!v) return [];
  return Array.isArray(v) ? v : [v];
}

/**
 * Patient-facing "Key signals" card.  Extracted from the older AiSummaryPanel
 * so the dashboard top row can show it side-by-side with "Review with care
 * team" and "Recent symptoms".  Reads only the LLM-generated patient
 * explanation; clinical_summary remains a clinician-only surface.
 */
export function KeySignalsCard({ summary }: Props) {
  const items = toArray(summary?.patient_explanation);
  return (
    <SectionCard
      title="Key signals"
      icon={CheckCircle}
      meta="updated just now"
      footer={
        <span className="inline-flex items-center gap-1.5">
          <Info size={11} aria-hidden="true" />
          AI summary generated from your records — not a clinical diagnosis.
        </span>
      }
    >
      {items.length > 0 ? (
        <ul className="flex flex-col gap-2 pl-0.5">
          {items.map((s, i) => (
            <li
              key={i}
              className="text-[0.85rem] leading-relaxed"
              style={{ color: "var(--text)" }}
            >
              {s}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[0.82rem]" style={{ color: "var(--text-dim)" }}>
          No summary available yet — check back after your next clinic visit.
        </p>
      )}
    </SectionCard>
  );
}
