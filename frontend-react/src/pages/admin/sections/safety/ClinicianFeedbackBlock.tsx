import { Badge } from "../../../../components/ui/Badge";
import { MetricCard } from "../../../../components/ui/MetricCard";
import type { SafetyCenter } from "../../../../types/api";

type ClinicianFeedback = SafetyCenter["clinician_feedback"];

/**
 * Human-oversight signal: how often clinicians accepted, edited, or rejected
 * AI-drafted output.
 *
 * `feedback` is optional at runtime even though the schema marks it required —
 * the endpoint omits it when no reviews exist yet — so every read is guarded
 * and falls back to zero rather than rendering "undefined".
 */
export function ClinicianFeedbackBlock({ feedback }: { feedback: ClinicianFeedback | undefined }) {
  const decisions = feedback?.decision_counts;
  const reasons = feedback?.reason_category_counts;
  const reasonEntries = reasons ? Object.entries(reasons) : [];

  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Reviews" value={feedback?.review_count ?? 0} status="muted" />
        <MetricCard label="Approved" value={decisions?.approved ?? 0} status="green" />
        <MetricCard label="Edited" value={decisions?.edited ?? 0} status="amber" />
        <MetricCard
          label="Rejected / unsafe"
          value={(decisions?.rejected ?? 0) + (decisions?.unsafe ?? 0)}
          status="red"
        />
      </div>
      {reasonEntries.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {reasonEntries.map(([reason, count]) => (
            <Badge key={reason} variant="amber">
              {reason.replace(/_/g, " ")}: {count}
            </Badge>
          ))}
        </div>
      )}
    </>
  );
}
