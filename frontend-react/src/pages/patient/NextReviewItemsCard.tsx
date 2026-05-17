import { useMemo } from "react";
import { ClipboardList } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { deriveReviewItems } from "./nextReviewItems";
import type { PatientReport } from "../../types/api";

interface Props { report: PatientReport }

/**
 * Compact "Next clinician review items" card.  Derives a short to-discuss
 * list from the existing report payload (review_reasons + concerning
 * multimodal signals + elevated timeline events) — no new backend field
 * needed.  Caps at 4 items so the card stays scannable.
 */
export function NextReviewItemsCard({ report }: Props) {
  const items = useMemo(() => deriveReviewItems(report), [report]);
  return (
    <SectionCard
      title="Next clinician review"
      icon={ClipboardList}
      meta={items.length > 0 ? `${items.length} to discuss` : undefined}
    >
      {items.length > 0 ? (
        <ul className="flex flex-col gap-2.5 pl-0.5">
          {items.map((item, i) => (
            <li
              key={i}
              className="flex flex-col gap-0.5"
              style={{
                borderLeft: `3px solid ${item.tone === "warning" ? "#f59e0b" : "var(--border-strong)"}`,
                paddingLeft: 10,
              }}
            >
              <span
                className="text-[0.8rem] font-semibold"
                style={{ color: "var(--text-strong)" }}
              >
                {item.title}
              </span>
              <span
                className="text-[0.78rem] leading-relaxed"
                style={{ color: "var(--text-dim)" }}
              >
                {item.detail}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[0.82rem]" style={{ color: "var(--text-dim)" }}>
          Nothing requires extra discussion right now.
        </p>
      )}
    </SectionCard>
  );
}
