import {
  CalendarDays,
  FlaskConical,
  Pill,
  Activity,
  ScanLine,
  AlertTriangle,
  Brain,
  Sparkles,
  X,
  ExternalLink,
  FileText,
  Image as ImageIcon,
} from "lucide-react";
import { useEffect, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { RelativeTime } from "../../components/ui/RelativeTime";
import { EmptyState } from "../../components/ui/states";
import { AIGeneratedLabel } from "../../components/ui/AIGeneratedLabel";
import { getAuthenticatedObjectUrl } from "../../api/client";
import type { TimelineEvent, TimelineMedia } from "../../types/api";

interface Props {
  events: TimelineEvent[];
  lastFetchedAt?: number | null;
}

/** Latest-N events shown by default; the rest is gated behind "View all". */
const DEFAULT_VISIBLE_EVENTS = 5;

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
  onSelect: (event: TimelineEvent) => void;
}

function TimelineEventCard({ event, onSelect }: TimelineEventCardProps) {
  const { Icon, tone, label } = eventIcon(event.type);
  const palette = toneStyle[tone];
  const isAi = isAiFlagEvent(event);
  const sevTone = severityTone(event.severity);
  const uncertainty = event.uncertainty;
  return (
    <li className="timeline-event-card">
      <button
        type="button"
        className="timeline-event-open"
        onClick={() => onSelect(event)}
        aria-label={`Open details for ${event.title}`}
      >
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
      </button>
    </li>
  );
}

function renderValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Not recorded";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function AuthenticatedTimelineMediaCard({ item, index }: { item: TimelineMedia; index: number }) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [artifactError, setArtifactError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let objectUrl: string | null = null;
    if (!item.artifact_url || !item.previewable) return undefined;

    getAuthenticatedObjectUrl(item.artifact_url)
      .then((url) => {
        objectUrl = url;
        if (active) setPreviewUrl(url);
      })
      .catch(() => {
        if (active) setArtifactError("Preview unavailable");
      });

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [item.artifact_url, item.previewable]);

  async function openArtifact() {
    if (!item.artifact_url) return;
    try {
      setArtifactError(null);
      const url = previewUrl ?? await getAuthenticatedObjectUrl(item.artifact_url);
      window.open(url, "_blank", "noopener,noreferrer");
      if (!previewUrl) window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch {
      setArtifactError("Artifact unavailable");
    }
  }

  return (
    <article className="timeline-media-card" key={`${item.label ?? "media"}-${index}`}>
      {previewUrl && item.previewable ? (
        <img src={previewUrl} alt={item.label ?? "Timeline media preview"} />
      ) : (
        <div className="timeline-media-placeholder">
          {item.previewable ? <ImageIcon size={22} /> : <FileText size={22} />}
        </div>
      )}
      <div>
        <strong>{item.label ?? "Uploaded file"}</strong>
        <span>{item.modality ?? item.content_type ?? "Record"}</span>
        {item.notes && <p>{item.notes}</p>}
        {item.artifact_url && (
          <button type="button" className="timeline-media-open" onClick={openArtifact}>
            Open artifact <ExternalLink size={13} />
          </button>
        )}
        {artifactError && <span className="timeline-media-error">{artifactError}</span>}
      </div>
    </article>
  );
}

function TimelineDetailModal({ event, onClose }: { event: TimelineEvent; onClose: () => void }) {
  const detail = event.detail;
  const fields = Object.entries(detail?.fields ?? {}).filter(([, value]) => value !== undefined);
  const media = detail?.media ?? [];

  return (
    <div className="timeline-detail-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="timeline-detail-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="timeline-detail-title"
        onMouseDown={(event_) => event_.stopPropagation()}
      >
        <header className="timeline-detail-header">
          <div>
            <p className="timeline-detail-kicker">{event.type} - {String(event.date).slice(0, 10)}</p>
            <h3 id="timeline-detail-title">{detail?.title ?? event.title}</h3>
          </div>
          <button type="button" className="timeline-detail-close" onClick={onClose} aria-label="Close details">
            <X size={18} />
          </button>
        </header>

        {event.summary && <p className="timeline-detail-summary">{event.summary}</p>}

        {fields.length > 0 && (
          <dl className="timeline-detail-fields">
            {fields.map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>{renderValue(value)}</dd>
              </div>
            ))}
          </dl>
        )}

        {detail?.findings && (
          <section className="timeline-detail-section">
            <h4>Findings</h4>
            <p>{detail.findings}</p>
          </section>
        )}

        {detail?.impression && (
          <section className="timeline-detail-section">
            <h4>Impression</h4>
            <p>{detail.impression}</p>
          </section>
        )}

        {detail?.message && (
          <section className="timeline-detail-section">
            <h4>Signal</h4>
            <p>{detail.message}</p>
          </section>
        )}

        {media.length > 0 && (
          <section className="timeline-detail-section">
            <h4>Attached media / reports</h4>
            <div className="timeline-media-grid">
              {media.map((item, index) => (
                <AuthenticatedTimelineMediaCard
                  item={item}
                  index={index}
                  key={`${item.label ?? "media"}-${index}`}
                />
              ))}
            </div>
          </section>
        )}

        {detail?.notes && (
          <p className="timeline-detail-note">{detail.notes}</p>
        )}
      </section>
    </div>
  );
}

export function TimelinePanel({ events, lastFetchedAt }: Props) {
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [expanded, setExpanded] = useState(false);
  const sorted = [...(events ?? [])].sort((a, b) => b.date.localeCompare(a.date));
  // Limit to the most recent N events by default so the timeline does NOT
  // create a 4000px-tall card on a long patient record.  "View all"
  // expands in place; the section-card collapsible chevron still hides
  // the whole panel when the patient wants the dashboard quiet.
  const visible = expanded ? sorted : sorted.slice(0, DEFAULT_VISIBLE_EVENTS);
  const hiddenCount = Math.max(0, sorted.length - visible.length);

  // Group by YYYY-MM-DD bucket so the timeline reads like a clinical record.
  const groups: { key: string; heading: { primary: string; secondary?: string }; items: TimelineEvent[] }[] = [];
  for (const ev of visible) {
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
      collapsible
      collapseId="patient-treatment-timeline"
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
        <>
          <div className={`timeline-root${expanded ? " timeline-root--expanded" : ""}`}>
            {groups.map((group) => (
              <TimelineDateGroup
                key={group.key}
                primary={group.heading.primary}
                secondary={group.heading.secondary}
                count={group.items.length}
              >
                {group.items.map((ev, i) => (
                  <TimelineEventCard key={i} event={ev} onSelect={setSelectedEvent} />
                ))}
              </TimelineDateGroup>
            ))}
          </div>
          {(hiddenCount > 0 || expanded) && (
            <button
              type="button"
              className="timeline-view-all"
              onClick={() => setExpanded((v) => !v)}
              aria-expanded={expanded}
            >
              {expanded
                ? `Show latest ${DEFAULT_VISIBLE_EVENTS} only`
                : `View all ${sorted.length} events`}
            </button>
          )}
        </>
      )}
      {selectedEvent && (
        <TimelineDetailModal event={selectedEvent} onClose={() => setSelectedEvent(null)} />
      )}
    </SectionCard>
  );
}
