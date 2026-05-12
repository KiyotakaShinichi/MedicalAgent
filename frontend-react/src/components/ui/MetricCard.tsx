import { clsx } from "clsx";
import type { LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number | null;
  unit?: string;
  icon?: LucideIcon;
  trend?: "up" | "down" | "neutral";
  status?: "green" | "amber" | "red" | "muted";
  sub?: string;
  className?: string;
}

const statusColors = {
  green: { text: "var(--green)", bg: "rgba(16,185,129,0.10)" },
  amber: { text: "var(--amber)", bg: "rgba(245,158,11,0.11)" },
  red: { text: "var(--rose)", bg: "rgba(244,63,94,0.11)" },
  muted: { text: "var(--text)", bg: "rgba(148,163,184,0.08)" },
};

export function MetricCard({
  label,
  value,
  unit,
  icon: Icon,
  status = "muted",
  sub,
  className,
}: MetricCardProps) {
  const colors = statusColors[status];
  return (
    <div className={clsx("metric-card", className)}>
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        {Icon && (
          <span className="metric-icon" style={{ background: colors.bg }}>
            <Icon size={15} style={{ color: colors.text }} />
          </span>
        )}
      </div>
      <div className="metric-card-value-row">
        <span className="metric-value" style={{ color: colors.text }}>
          {value ?? "-"}
        </span>
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      {sub && <p className="metric-sub">{sub}</p>}
    </div>
  );
}
