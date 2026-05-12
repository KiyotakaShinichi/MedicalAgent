import { CalendarDays } from "lucide-react";
import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { statusVariant } from "../../components/ui/badgeUtils";
import { EmptyPane } from "../../components/ui/Spinner";
import type { TimelineEvent } from "../../types/api";

interface Props { events: TimelineEvent[] }

export function TimelinePanel({ events }: Props) {
  const sorted = [...(events ?? [])].sort((a, b) => b.date.localeCompare(a.date));
  return (
    <Card>
      <CardHeader>
        <SectionTitle>Treatment Timeline</SectionTitle>
        <CalendarDays size={14} style={{ color: "var(--text-faint)" }} />
      </CardHeader>
      {sorted.length === 0 ? (
        <EmptyPane label="No timeline events" />
      ) : (
        <div className="flex flex-col gap-0">
          {sorted.map((ev, i) => (
            <div key={i} className="flex gap-3 py-2.5 border-b last:border-0" style={{ borderColor: "var(--border)" }}>
              <div className="flex flex-col items-center" style={{ width: 32, flexShrink: 0 }}>
                <div
                  className="w-2 h-2 rounded-full mt-1"
                  style={{
                    background:
                      ev.severity === "urgent" ? "var(--rose)" :
                      ev.severity === "warning" ? "var(--amber)" : "var(--green)",
                  }}
                />
                {i < sorted.length - 1 && (
                  <div className="flex-1 w-px mt-1" style={{ background: "var(--border)" }} />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-medium" style={{ color: "var(--text)" }}>{ev.title}</span>
                  <Badge variant={statusVariant(ev.severity ?? "")}>{ev.type}</Badge>
                </div>
                <p className="text-xs mt-0.5" style={{ color: "var(--text-dim)" }}>{ev.summary}</p>
                <p className="text-xs mt-0.5" style={{ color: "var(--text-faint)" }}>{ev.date?.slice(0, 10)}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
