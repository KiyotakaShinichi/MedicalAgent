import { CalendarDays } from "lucide-react";
import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { statusVariant } from "../../components/ui/badgeUtils";
import { EmptyState } from "../../components/ui/states";
import { AIGeneratedLabel } from "../../components/ui/AIGeneratedLabel";
import { RiskBadge } from "../../components/ui/RiskBadge";
import type { TimelineEvent } from "../../types/api";

interface Props {
  events: TimelineEvent[];
}

function isAiFlagEvent(event: TimelineEvent): boolean {
  if (event.ai_generated) return true;
  const type = (event.type ?? "").toLowerCase();
  return (
    type.includes("ai_risk_flag") ||
    type.includes("risk_flag") ||
    type.includes("ai_summary")
  );
}

export function TimelinePanel({ events }: Props) {
  const sorted = [...(events ?? [])].sort((a, b) => b.date.localeCompare(a.date));

  return (
    <Card>
      <CardHeader>
        <SectionTitle>Treatment Timeline</SectionTitle>
        <CalendarDays size={14} style={{ color: "var(--text-faint)" }} />
      </CardHeader>
      {sorted.length === 0 ? (
        <EmptyState label="No timeline events yet — symptoms, labs, and imaging will appear here." />
      ) : (
        <div className="flex flex-col gap-0">
          {sorted.map((ev, i) => {
            const isAi = isAiFlagEvent(ev);
            const uncertainty = ev.uncertainty;
            return (
              <div
                key={i}
                className="flex gap-3 py-2.5 border-b last:border-0"
                style={{ borderColor: "var(--border)" }}
              >
                <div className="flex flex-col items-center" style={{ width: 32, flexShrink: 0 }}>
                  <div
                    className="w-2 h-2 rounded-full mt-1"
                    style={{
                      background:
                        ev.severity === "urgent" || ev.severity === "urgent_review"
                          ? "var(--rose)"
                          : ev.severity === "warning" || ev.severity === "watch"
                            ? "var(--amber)"
                            : "var(--green)",
                    }}
                    aria-hidden="true"
                  />
                  {i < sorted.length - 1 && (
                    <div className="flex-1 w-px mt-1" style={{ background: "var(--border)" }} />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-medium" style={{ color: "var(--text)" }}>
                      {ev.title}
                    </span>
                    <Badge variant={statusVariant(ev.severity ?? "")}>{ev.type}</Badge>
                    {(ev.severity === "urgent" ||
                      ev.severity === "urgent_review" ||
                      ev.severity === "watch") && (
                      <RiskBadge level={ev.severity === "urgent" ? "urgent_review" : ev.severity} />
                    )}
                  </div>
                  <p className="text-xs mt-0.5" style={{ color: "var(--text-dim)" }}>
                    {ev.summary}
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "var(--text-faint)" }}>
                    {ev.date?.slice(0, 10)}
                  </p>

                  {isAi && (
                    <AIGeneratedLabel
                      className="mt-1.5"
                      confidence={uncertainty?.confidence_level ?? null}
                      uncertaintyReason={uncertainty?.uncertainty_reason ?? null}
                      clinicianReviewRequired={uncertainty?.clinician_review_required ?? null}
                      timestamp={ev.date}
                      source={ev.evidence_source ?? "risk_engine"}
                      modelVersion={ev.model_version ?? null}
                    />
                  )}

                  {!isAi && uncertainty?.missing_data_indicators?.length ? (
                    <p className="text-xs mt-1" style={{ color: "var(--text-faint)" }}>
                      Missing data:{" "}
                      {uncertainty.missing_data_indicators.join(", ")}
                    </p>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
