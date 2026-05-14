import { Gauge } from "lucide-react";
import { Badge } from "./Badge";
import type { BadgeVariant } from "./badgeUtils";
import type { ConfidenceLevel } from "../../lib/constants";

interface ConfidenceBadgeProps {
  level: ConfidenceLevel | string | null | undefined;
  showIcon?: boolean;
  className?: string;
}

function variantFor(level: string | null | undefined): BadgeVariant {
  if (level === "high") return "green";
  if (level === "moderate") return "amber";
  if (level === "low") return "red";
  return "muted";
}

/**
 * Displays the AI/model confidence in the uncertainty block.
 *
 * We deliberately label the badge as "Confidence" not "Certainty" so
 * readers do not interpret it as a clinical certainty claim.
 */
export function ConfidenceBadge({ level, showIcon = true, className }: ConfidenceBadgeProps) {
  if (!level) return null;
  const text = `Confidence: ${level}`;
  return (
    <Badge variant={variantFor(level)} className={className}>
      {showIcon && <Gauge size={11} aria-hidden="true" />}
      {text}
    </Badge>
  );
}
