import { Clock } from "lucide-react";
import { Badge } from "./Badge";
import type { BadgeVariant } from "./badgeUtils";

interface FreshnessChipProps {
  /** Either an ISO string or the structured artifact_freshness block. */
  artifactFreshness?: {
    generated_at?: string | null;
    status?: string | null;
    ttl_seconds?: number | null;
  } | null;
  generatedAt?: string | null;
  className?: string;
}

function relativeAge(generatedAt: string | null | undefined): string {
  if (!generatedAt) return "Unknown age";
  const ts = new Date(generatedAt);
  if (Number.isNaN(ts.getTime())) return "Unknown age";
  const seconds = Math.max(0, Math.floor((Date.now() - ts.getTime()) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function variantFor(status: string | null | undefined): BadgeVariant {
  if (!status) return "muted";
  if (status === "fresh" || status === "available") return "green";
  if (status === "stale") return "amber";
  if (status === "error" || status === "not_generated") return "red";
  return "muted";
}

/**
 * Compact chip showing how recent an artifact is and whether it has
 * passed its TTL. Use it on every Safety & Evaluation Center card so
 * viewers know whether they are looking at fresh data or a stored one.
 */
export function FreshnessChip({
  artifactFreshness,
  generatedAt,
  className,
}: FreshnessChipProps) {
  const ts = artifactFreshness?.generated_at ?? generatedAt ?? null;
  const status = artifactFreshness?.status ?? null;
  return (
    <Badge variant={variantFor(status)} className={className}>
      <Clock size={11} aria-hidden="true" />
      {ts ? relativeAge(ts) : "No timestamp"}
      {status === "stale" ? " - stale" : null}
    </Badge>
  );
}
