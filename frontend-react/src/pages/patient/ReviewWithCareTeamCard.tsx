import { AlertTriangle } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import type { AiSummary } from "../../types/api";

interface Props { summary: AiSummary | null }

/**
 * Patient-facing "Review with care team" card.  Surfaces the model's
 * review_reasons list — the items the system thinks the patient should bring
 * up at their next visit.  Tonally calm (amber, not red) because these are
 * not urgent alerts; urgent flags route through the chat safety path.
 */
export function ReviewWithCareTeamCard({ summary }: Props) {
  const reasons = summary?.review_reasons ?? [];
  return (
    <SectionCard
      title="Review with care team"
      icon={AlertTriangle}
      meta={reasons.length > 0 ? `${reasons.length} items` : undefined}
    >
      {reasons.length > 0 ? (
        <ul className="flex flex-col gap-2 pl-0.5">
          {reasons.map((s, i) => (
            <li
              key={i}
              className="text-[0.85rem] leading-relaxed flex gap-2"
              style={{ color: "var(--text)" }}
            >
              <span
                aria-hidden="true"
                style={{
                  display: "inline-block",
                  width: 6, height: 6, borderRadius: 999,
                  background: "#f59e0b",
                  marginTop: 8, flexShrink: 0,
                }}
              />
              <span>{s}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[0.82rem]" style={{ color: "var(--text-dim)" }}>
          Nothing flagged today.
        </p>
      )}
    </SectionCard>
  );
}
