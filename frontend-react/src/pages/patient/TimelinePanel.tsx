import {
  CalendarDays,
  FlaskConical,
  Pill,
  Activity,
  ScanLine,
  AlertTriangle,
  Brain,
  Sparkles,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { RelativeTime } from "../../components/ui/RelativeTime";
import { EmptyState } from "../../components/ui/states";
import { AIGeneratedLabel } from "../../components/ui/AIGeneratedLabel";
import type { TimelineEvent } from "../../types/api";

interface Props {
  events: TimelineEvent[];
  lastFetchedAt?: number | null;
}

type Tone = "rose" | "blue" | "amber" | "purple" | "green" | "neutral";

const toneStyle: Record<Tone, { bg: string; fg: string }> = {
  rose:    { bg: "var(--rose-pale)", fg: "var(--rose-deep)" },
  blue:    { bg: "#dbeafe",          fg: "#1e3a8a" },
  amber:   { bg: "#fef3c7",          fg: "#92400e" },
  purple:  { bg: "#ede9fe",          fg: "#5b21b6" },
  green:   { bg: "#d1fae5",          fg: "#065f46" },
  neutral: { bg: "var(--surface2)",  fg: "var(--text-dim)" },
};

function eventIcon(type: string | undefined): { Icon: LucideIcon; tone: Tone; label: string } {
  const t = (type ?? "").toLowerCase();
  if (t.includes("lab"))                                       return { Icon: FlaskConical,   tone: "blue",   label: "Lab" };
  if (t.includes("medication") || t.includes("med"))           return { Icon: Pill,            tone: "purple", label: "Medication" };
  if (t.includes("symptom"))                                   return { Icon: Activity,        tone: "amber",  label: "Symptom" };
  if (t.includes("imaging") || t.includes("mri") || t.includes("ct")) return { Icon: ScanLine, tone: "rose",   label: "Imaging" };
  if (t.includes("risk") || t.includes("flag"))                return { Icon: AlertTriangle,   tone: "amber",  label: "Risk flag" };
  if (t.includes("ai"))                                        return { Icon: Brain,           tone: "rose",   label: "AI" };
  if (t.includes("treatment"))                                 return { Icon: Sparkles,        tone: "green",  label: "Treatment" };
  return { Icon: CalendarDays, tone: "neutral", label: type ?? "Event" };
}

function severityTone(severity?: string): "danger" | "warning" | "success" | "neutral" {
  const s = (severity ?? "").toLowerCase();
  if (s.includes("urgent")) return "danger";
  if (s.includes("warn") || s.includes("watch") || s === "review") return "warning";
  if (s.includes("normal") || s.includes("stable") || s.includes("low")) return "success";
  return "neutral";
}

function isAiFlagEvent(event: TimelineEvent): boolean {
  if (event.ai_generated) return true;
  const type = (event.type ?? "").toLowerCase();
  return type.includes("ai_risk_flag") || type.includes("risk_flag") || type.includes("ai_summary");
}

function formatHeading(dateIso: string): { primary: string; secondary?: string } {
  if (!dateIso) return { primary: "" };
  const date = new Date(dateIso);
  if (Number.isNaN(date.getTime())) return { primary: dateIso.slice(0, 10) };
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const target = new Date(date);
  target.setHours(0, 0, 0, 0);
  const diffDays = Math.round((today.getTime() - target.getTime()) / 86400000);
  const formatted = date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  if (diffDays === 0) return { primary: "Today",          secondary: formatted };
  if (diffDays === 1) return { primary: "Yesterday",      secondary: formatted };
  if (diffDays < 7)   return { primary: `${diffDays}d ago`, secondary: formatted };
  return { primary: date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) };
}

interface TimelineDateGroupProps {
  primary: string;
  secondary?: string;
  count: number;
  children: React.ReactNode;
}

/** Date header + events grouped under one date. Pure CSS-grid, no negative margins. */
function TimelineDateGroup({ primary, secondary, count, children }: TimelineDateGroupProps) {
  return (
    <section className="timeline-date-group">
      <header className="timeline-date-header">
        <span className="timeline-date-primary">{primary}</span>
        {secondary && <span className="timeline-date-secondary">{secondary}</span>}
        <span className="timeline-date-count">{count} {count === 1 ? "event" : "events"}</span>
      </header>
      <ul className="timeline-events">{children}</ul>
    </section>
  );
}

interface TimelineEventCardProps {
  event: TimelineEvent;
}

function TimelineEventCard({ event }: TimelineEventCardProps) {
  const { Icon, tone, label } = eventIcon(event.type);
  const palette = toneStyle[tone];
  const isAi = isAiFlagEvent(event);
  const sevTone = severityTone(event.severity);
  const uncertainty = event.uncertainty;
  return (
    <li className="timeline-event-card">
      <span
        className="timeline-event-icon"
        style={{ background: palette.bg, color: palette.fg }}
        aria-hidden="true"
      >
        <Icon size={15} />
      </span>
      <div className="timeline-event-body">
        <div className="timeline-event-headline">
          <span className="timeline-event-title">{event.title}</span>
          <StatusBadge tone="accent" size="sm">{label}</StatusBadge>
          {event.severity && sevTone !== "neutral" && (
            <StatusBadge tone={sevTone} size="sm">{event.severity}</StatusBadge>
          )}
        </div>
        {event.summary && (
          <p className="timeline-event-summary">{event.summary}</p>
        )}

        {isAi && (
          <AIGeneratedLabel
            className="mt-2"
            confidence={uncertainty?.confidence_level ?? null}
            uncertaintyReason={uncertainty?.uncertainty_reason ?? null}
            clinicianReviewRequired={uncertainty?.clinician_review_required ?? null}
            timestamp={event.date}
            source={event.evidence_source ?? "risk_engine"}
            modelVersion={event.model_version ?? null}
          />
        )}
        {!isAi && uncertainty?.missing_data_indicators?.length ? (
          <p className="timeline-event-missing">
            Missing data: {uncertainty.missing_data_indicators.join(", ")}
          </p>
        ) : null}
      </div>
    </li>
  );
}

export function TimelinePanel({ events, lastFetchedAt }: Props) {
  const sorted = [...(events ?? [])].sort((a, b) => b.date.localeCompare(a.date));

  // Group by YYYY-MM-DD bucket so the timeline reads like a clinical record.
  const groups: { key: string; heading: { primary: string; secondary?: string }; items: TimelineEvent[] }[] = [];
  for (const ev of sorted) {
    const key = (ev.date ?? "").slice(0, 10);
    const last = groups[groups.length - 1];
    if (last && last.key === key) {
      last.items.push(ev);
    } else {
      groups.push({ key, heading: formatHeading(ev.date), items: [ev] });
    }
  }

  return (
    <SectionCard
      title="Treatment timeline"
      icon={CalendarDays}
      meta={
        <span className="flex items-center gap-2">
          {sorted.length > 0 && <span>{sorted.length} events · {groups.length} days</span>}
          {sorted.length > 0 && lastFetchedAt != null && <span style={{ opacity: 0.6 }}>·</span>}
          <RelativeTime timestamp={lastFetchedAt ?? null} prefix="updated" />
        </span>
      }
    >
      {sorted.length === 0 ? (
        <EmptyState label="No timeline events yet — symptoms, labs, and imaging will appear here." />
      ) : (
        <div className="timeline-root">
          {groups.map((group) => (
            <TimelineDateGroup
              key={group.key}
              primary={group.heading.primary}
              secondary={group.heading.secondary}
              count={group.items.length}
            >
              {group.items.map((ev, i) => <TimelineEventCard key={i} event={ev} />)}
            </TimelineDateGroup>
          ))}
        </div>
      )}
    </SectionCard>
  );
}
