import { AlertTriangle, Clock3 } from "lucide-react";
import { Badge } from "../../components/ui/Badge";
import { statusVariant } from "../../components/ui/badgeUtils";
import { EmptyPane } from "../../components/ui/Spinner";
import type { ReviewQueueItem } from "../../types/api";

interface Props {
  queue: ReviewQueueItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

/**
 * Clinician review queue.
 *
 * A list of selectable patients ordered by the backend's triage priority. This
 * component does not re-sort or re-rank: the order it receives is the order it
 * shows, so the clinical prioritisation stays owned by the backend.
 *
 * Rendered as a real list with `aria-current` on the selected entry, so screen
 * reader users get both the queue length and which patient is open — neither
 * of which a bare stack of buttons conveys.
 */
export function ReviewQueue({ queue, selectedId, onSelect }: Props) {
  if (queue.length === 0) {
    return (
      <div className="review-queue-list">
        <EmptyPane label="Queue empty" />
      </div>
    );
  }

  return (
    <ul className="review-queue-list" aria-label={`Review queue, ${queue.length} patients`}>
      {queue.map((item) => {
        const selected = selectedId === item.patient_id;
        const urgentCount = item.urgent_flags?.length ?? 0;
        // The backend may omit a status; "review" is the neutral fallback and
        // must not be styled as a resolved state.
        const status = item.overall_status ?? "review";

        return (
          <li key={item.patient_id}>
            <button
              type="button"
              onClick={() => onSelect(item.patient_id)}
              className={`review-queue-card${selected ? " is-selected" : ""}`}
              aria-current={selected ? "true" : undefined}
            >
              <div className="review-queue-card-top">
                <div className="review-queue-patient">
                  <strong>{item.patient_name}</strong>
                  <span>{item.patient_id}</span>
                </div>
                <Badge variant={statusVariant(item.overall_status ?? "")} className="review-queue-status">
                  {status.replace(/_/g, " ")}
                </Badge>
              </div>

              <div className="review-queue-meta">
                <span>
                  <Clock3 size={12} aria-hidden="true" />
                  {/* An absent priority shows a dash rather than 0, which would
                      read as "lowest priority" instead of "not scored". */}
                  Priority {item.priority_score?.toFixed(0) ?? "-"}
                </span>
                {urgentCount > 0 && (
                  <span className="is-urgent">
                    <AlertTriangle size={12} aria-hidden="true" />
                    {urgentCount} urgent
                  </span>
                )}
              </div>

              {item.latest_decision && (
                <p className="review-queue-decision">Last review: {item.latest_decision}</p>
              )}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
