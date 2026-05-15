import { clsx } from "clsx";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { LucideIcon } from "lucide-react";

type Status = "green" | "amber" | "red" | "muted";
type Trend = "up" | "down" | "neutral";

interface MetricCardProps {
  label: string;
  value: string | number | null;
  unit?: string;
  icon?: LucideIcon;
  trend?: Trend;
  trendLabel?: string;
  status?: Status;
  sub?: string;
  range?: string;
  className?: string;
}

const statusColors: Record<Status, { text: string; iconBg: string; barColor: string; pillBg: string; pillText: string; pillBorder: string }> = {
  green: {
    text:        "#047857",
    iconBg:      "#ecfdf5",
    barColor:    "#10b981",
    pillBg:      "#ecfdf5",
    pillText:    "#047857",
    pillBorder:  "#a7f3d0",
  },
  amber: {
    text:        "#92400e",
    iconBg:      "#fffbeb",
    barColor:    "#f59e0b",
    pillBg:      "#fffbeb",
    pillText:    "#92400e",
    pillBorder:  "#fde68a",
  },
  red: {
    text:        "#b91c1c",
    iconBg:      "#fef2f2",
    barColor:    "#ef4444",
    pillBg:      "#fef2f2",
    pillText:    "#b91c1c",
    pillBorder:  "#fecaca",
  },
  muted: {
    text:        "var(--text-strong)",
    iconBg:      "var(--surface2)",
    barColor:    "var(--border-strong)",
    pillBg:      "var(--rose-pale)",
    pillText:    "var(--rose-deep)",
    pillBorder:  "var(--border-strong)",
  },
};

const trendInfo: Record<Trend, { Icon: LucideIcon; label: string; color: string }> = {
  up:      { Icon: TrendingUp,   label: "Rising",  color: "#dc2626" },
  down:    { Icon: TrendingDown, label: "Falling", color: "#0284c7" },
  neutral: { Icon: Minus,        label: "Stable",  color: "var(--text-dim)" },
};

const statusLabel: Record<Status, string> = {
  green: "In range",
  amber: "Borderline",
  red:   "Out of range",
  muted: "No data",
};

export function MetricCard({
  label,
  value,
  unit,
  icon: Icon,
  status = "muted",
  trend,
  trendLabel,
  sub,
  range,
  className,
}: MetricCardProps) {
  const colors = statusColors[status];
  const trendData = trend ? trendInfo[trend] : null;
  return (
    <div
      className={clsx("metric-card", className)}
      style={{
        position: "relative",
        overflow: "hidden",
      }}
    >
      <span
        aria-hidden="true"
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: 3,
          background: colors.barColor,
        }}
      />
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        {Icon && (
          <span className="metric-icon" style={{ background: colors.iconBg }}>
            <Icon size={13} style={{ color: colors.text }} />
          </span>
        )}
      </div>
      <div className="metric-card-value-row">
        <span className="metric-value">{value ?? "-"}</span>
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 3,
            padding: "1px 7px",
            borderRadius: 999,
            background: colors.pillBg,
            color: colors.pillText,
            border: `1px solid ${colors.pillBorder}`,
            fontSize: "0.66rem",
            fontWeight: 600,
            letterSpacing: 0.02,
          }}
        >
          {statusLabel[status]}
        </span>
        {trendData && (
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 2,
              fontSize: "0.7rem",
              color: trendData.color,
              fontWeight: 600,
            }}
          >
            <trendData.Icon size={11} />
            {trendLabel ?? trendData.label}
          </span>
        )}
        {range && (
          <span style={{ color: "var(--text-faint)", fontSize: "0.68rem", marginLeft: "auto" }}>
            ref {range}
          </span>
        )}
      </div>
      {sub && <p className="metric-sub">{sub}</p>}
    </div>
  );
}
