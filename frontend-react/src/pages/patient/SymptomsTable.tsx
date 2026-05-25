import { useState } from "react";
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
  const [expanded, setExpanded] = useState(false);
  const initialLimit = 5;
  const visible = expanded ? sorted : sorted.slice(0, initialLimit);
  const hiddenCount = Math.max(0, sorted.length - visible.length);

  // Dashboard-style summary row: latest symptom name + total + max severity.
  const latest = sorted[0] ?? null;
  const maxSeverity = sorted.length > 0 ? Math.max(...sorted.map((s) => s.severity ?? 0)) : 0;
  const summaryTone =
    maxSeverity >= 7 ? "concern" :
    maxSeverity >= 4 ? "watch" : "good";

  return (
    <SectionCard
      title="Symptom log"
      icon={Activity}
      collapsible
      collapseId="patient-symptom-log"
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
        <>
          {latest && (
            <div className={`symptom-summary symptom-summary--${summaryTone}`}>
              <div className="symptom-summary-main">
                <span className="symptom-summary-eyebrow">Latest</span>
                <p className="symptom-summary-name">{latest.symptom}</p>
                <span className="symptom-summary-meta">
                  {latest.date?.slice(0, 10)} · {sorted.length} total record{sorted.length === 1 ? "" : "s"}
                </span>
              </div>
              <div className="symptom-summary-bar">
                <SeverityBar value={maxSeverity} compact />
                <span className="symptom-summary-bar-label">peak severity</span>
              </div>
            </div>
          )}
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
          </ul>
          {(hiddenCount > 0 || expanded) && (
            <button
              type="button"
              className="symptom-view-all"
              onClick={() => setExpanded((v) => !v)}
              aria-expanded={expanded}
            >
              {expanded
                ? `Show latest ${initialLimit} only`
                : `View all ${sorted.length} records`}
            </button>
          )}
        </>
      )}
    </SectionCard>
  );
}
