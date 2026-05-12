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

export function ReviewQueue({ queue, selectedId, onSelect }: Props) {
  return (
    <div className="review-queue-list">
      {queue.length === 0 && <EmptyPane label="Queue empty" />}
      {queue.map((item) => {
        const selected = selectedId === item.patient_id;
        const urgentCount = item.urgent_flags?.length ?? 0;
        return (
          <button
            key={item.patient_id}
            type="button"
            onClick={() => onSelect(item.patient_id)}
            className={`review-queue-card${selected ? " is-selected" : ""}`}
          >
            <div className="review-queue-card-top">
              <div className="review-queue-patient">
                <strong>{item.patient_name}</strong>
                <span>{item.patient_id}</span>
              </div>
              <Badge variant={statusVariant(item.overall_status ?? "")} className="review-queue-status">
                {(item.overall_status ?? "review").replace(/_/g, " ")}
              </Badge>
            </div>

            <div className="review-queue-meta">
              <span>
                <Clock3 size={12} aria-hidden="true" />
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
        );
      })}
    </div>
  );
}
