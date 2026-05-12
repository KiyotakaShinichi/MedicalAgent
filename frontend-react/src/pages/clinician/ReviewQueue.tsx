import { AlertTriangle } from "lucide-react";
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
    <div className="flex flex-col gap-0">
      {queue.length === 0 && <EmptyPane label="Queue empty" />}
      {queue.map((item) => (
        <button
          key={item.patient_id}
          onClick={() => onSelect(item.patient_id)}
          className="w-full text-left px-3 py-3 border-b transition-colors hover:opacity-90"
          style={{
            borderColor: "var(--border)",
            background: selectedId === item.patient_id ? "rgba(244,63,94,0.08)" : "transparent",
          }}
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-semibold" style={{ color: "var(--text)" }}>
              {item.patient_name}
            </span>
            <Badge variant={statusVariant(item.overall_status ?? "")}>
              {item.overall_status}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs tabular-nums" style={{ color: "var(--text-faint)" }}>
              Priority {item.priority_score?.toFixed(0) ?? "-"}
            </span>
            {(item.urgent_flags?.length ?? 0) > 0 && (
              <span className="flex items-center gap-1 text-xs" style={{ color: "var(--rose)" }}>
                <AlertTriangle size={10} />
                {item.urgent_flags.length} urgent
              </span>
            )}
          </div>
          {item.latest_decision && (
            <p className="text-xs mt-0.5" style={{ color: "var(--text-faint)" }}>
              Last: {item.latest_decision}
            </p>
          )}
        </button>
      ))}
    </div>
  );
}
