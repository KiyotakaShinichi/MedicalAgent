import { AlertTriangle, Activity, Info } from "lucide-react";
import { Badge } from "./Badge";
import type { BadgeVariant } from "./badgeUtils";
import { RISK_LEVEL_LABELS, type RiskLevel } from "../../lib/constants";

interface RiskBadgeProps {
  level: RiskLevel | string | null | undefined;
  showIcon?: boolean;
  className?: string;
}

function variantFor(level: string | null | undefined): BadgeVariant {
  if (!level) return "muted";
  if (level === "urgent_review" || level === "urgent") return "red";
  if (level === "watch") return "amber";
  if (level === "info") return "blue";
  return "muted";
}

function iconFor(level: string | null | undefined) {
  if (!level) return null;
  if (level === "urgent_review" || level === "urgent") return AlertTriangle;
  if (level === "watch") return Activity;
  return Info;
}

/**
 * Standardized risk badge for AI risk flags and clinician-review surfaces.
 *
 * Always uses the safe vocabulary — never displays "diagnosis", "cancer
 * detected", or any other clinical-claim language. Callers pass a
 * `RiskLevel` string ("info" | "watch" | "urgent_review") or an unknown
 * string, which falls back to a muted "unknown" badge.
 */
export function RiskBadge({ level, showIcon = true, className }: RiskBadgeProps) {
  const Icon = iconFor(level);
  const label = (level && RISK_LEVEL_LABELS[level as RiskLevel]) || level || "unknown";
  return (
    <Badge variant={variantFor(level)} className={className}>
      {showIcon && Icon && <Icon size={11} aria-hidden="true" />}
      {label}
    </Badge>
  );
}
