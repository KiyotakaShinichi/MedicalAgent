import { Badge } from "./Badge";
import type { BadgeVariant } from "./badgeUtils";
import {
  REVIEW_DECISION_LABELS,
  type ReviewDecision,
} from "../../lib/constants";

interface ReviewStatusBadgeProps {
  decision: ReviewDecision | string | null | undefined;
  className?: string;
}

function variantFor(decision: string | null | undefined): BadgeVariant {
  if (!decision) return "muted";
  if (decision === "approved") return "green";
  if (decision === "edited") return "blue";
  if (decision === "needs_followup") return "amber";
  if (decision === "missing_evidence" || decision === "wrong_escalation") return "amber";
  if (decision === "rejected" || decision === "unsafe") return "red";
  return "muted";
}

/**
 * Badge for clinician review decisions. Renders the human-readable label
 * from REVIEW_DECISION_LABELS and color-codes by category — green for
 * approvals, amber for items that need more information, red for rejected
 * or unsafe.
 */
export function ReviewStatusBadge({ decision, className }: ReviewStatusBadgeProps) {
  if (!decision) {
    return (
      <Badge variant="muted" className={className}>
        Awaiting clinician review
      </Badge>
    );
  }
  const label =
    REVIEW_DECISION_LABELS[decision as ReviewDecision] ?? decision.replace(/_/g, " ");
  return (
    <Badge variant={variantFor(decision)} className={className}>
      {label}
    </Badge>
  );
}
