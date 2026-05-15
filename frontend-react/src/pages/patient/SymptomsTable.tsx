import { Activity } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { RelativeTime } from "../../components/ui/RelativeTime";
import { EmptyState } from "../../components/ui/states";
import type { Symptom } from "../../types/api";

interface Props {
  symptoms: Symptom[];
  /** Compact mode = used as a side panel beside another card; trims notes
   *  and shows fewer rows so it doesn't dwarf the column it sits in. */
  compact?: boolean;
  lastFetchedAt?: number | null;
}

function SeverityBar({ value, compact }: { value: number; compact?: boolean }) {
  const color =
    value >= 7 ? "#dc2626" :
    value >= 4 ? "#d97706" : "#059669";
  const label =
    value >= 7 ? "Severe" :
    value >= 4 ? "Moderate" : "Mild";
  return (
    <div className="flex items-center gap-2">
      <div
        className="flex-1 rounded-full"
        style={{ height: 4, background: "var(--surface2)", minWidth: 24 }}
      >
        <div
          className="rounded-full transition-all"
          style={{
            width: `${(value / 10) * 100}%`,
            height: 4,
            background: color,
          }}
        />
      </div>
      <span
        className="tabular-nums font-semibold"
        style={{ color, fontSize: "0.8rem", minWidth: 18, textAlign: "right" }}
      >
        {value}
      </span>
      {!compact && (
        <span
          className="font-medium"
          style={{ color: "var(--text-faint)", fontSize: "0.7rem", minWidth: 56, textAlign: "right" }}
        >
          {label}
        </span>
      )}
    </div>
  );
}

export function SymptomsTable({ symptoms, compact = false, lastFetchedAt }: Props) {
  const sorted = [...(symptoms ?? [])].sort((a, b) => b.date.localeCompare(a.date));
  const visibleLimit = compact ? 5 : 8;
  const visible = sorted.slice(0, visibleLimit);

  return (
    <SectionCard
      title="Symptom log"
      icon={Activity}
      meta={
        <span className="flex items-center gap-2">
          {sorted.length > 0 && <span>{sorted.length} total</span>}
          {sorted.length > 0 && lastFetchedAt != null && <span style={{ opacity: 0.6 }}>·</span>}
          <RelativeTime timestamp={lastFetchedAt ?? null} prefix="updated" />
        </span>
      }
    >
      {sorted.length === 0 ? (
        <EmptyState label="No symptoms recorded — add new ones from the support chat." />
      ) : (
        <ul className={compact ? "symptom-list symptom-list--compact" : "symptom-list"}>
          {visible.map((s, i) => (
            <li key={i} className={compact ? "symptom-row symptom-row--compact" : "symptom-row"}>
              <span className="symptom-date">{s.date?.slice(5, 10)}</span>
              <div className="symptom-info">
                <p className="symptom-name">{s.symptom}</p>
                {!compact && s.notes && <p className="symptom-notes">{s.notes}</p>}
              </div>
              <div className="symptom-severity">
                <SeverityBar value={s.severity} compact={compact} />
              </div>
            </li>
          ))}
          {sorted.length > visibleLimit && (
            <p className="symptom-more">+ {sorted.length - visibleLimit} more in record</p>
          )}
        </ul>
      )}
    </SectionCard>
  );
}
