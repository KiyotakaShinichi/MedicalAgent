import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { LabHistoryPoint } from "../../types/api";
import { EmptyPane } from "../ui/Spinner";

const LINES = [
  { key: "wbc",        color: "#3b82f6", label: "WBC",        unit: "K/uL"        },
  { key: "hemoglobin", color: "#10b981", label: "Hemoglobin", unit: "g/dL"        },
  { key: "platelets",  color: "#f59e0b", label: "Platelets",  unit: "K/uL", scale: 0.1 },
] as const;

interface Props {
  data: LabHistoryPoint[];
}

interface TooltipPayload {
  dataKey: string;
  value: number | null;
  color: string;
}

function ChartTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "8px 10px",
        boxShadow: "0 4px 12px rgba(17,24,39,0.08)",
        fontSize: 12,
        minWidth: 140,
      }}
    >
      <p style={{ fontWeight: 600, color: "var(--text-strong)", marginBottom: 4 }}>{label}</p>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {payload.map((p) => {
          const line = LINES.find((l) => l.key === p.dataKey);
          if (!line) return null;
          const value = p.value;
          const displayValue =
            value == null
              ? "-"
              : line.key === "platelets" && "scale" in line
                ? Math.round(value / line.scale) // unscale platelets back to real value
                : value.toFixed(1);
          return (
            <div key={p.dataKey} style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--text-dim)" }}>
                <span
                  style={{
                    width: 8, height: 8, borderRadius: 999,
                    background: p.color, display: "inline-block",
                  }}
                />
                {line.label}
              </span>
              <span style={{ color: "var(--text)", fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>
                {displayValue}
                <span style={{ color: "var(--text-faint)", fontWeight: 400, marginLeft: 2 }}>
                  {value != null ? ` ${line.unit}` : ""}
                </span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function LabTrendsChart({ data }: Props) {
  if (!data.length) return <EmptyPane label="No lab history" />;

  const chartData = data.map((d) => ({
    date: d.date?.slice(5, 10) ?? "", // MM-DD only — saves horizontal space
    wbc: d.wbc,
    hemoglobin: d.hemoglobin,
    platelets: d.platelets != null ? d.platelets * 0.1 : null,
  }));

  return (
    <div>
      {/* Inline legend — small, neutral, no recharts default styling */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 14,
          paddingBottom: 8,
          fontSize: "0.74rem",
        }}
      >
        {LINES.map((line) => (
          <span
            key={line.key}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              color: "var(--text-dim)",
            }}
          >
            <span style={{
              width: 10, height: 2, borderRadius: 2, background: line.color,
            }} />
            {line.label}
            {line.key === "platelets" && (
              <span style={{ color: "var(--text-faint)", fontSize: "0.68rem" }}>· scaled ×0.1</span>
            )}
          </span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="var(--border-soft)" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: "var(--text-faint)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "var(--text-faint)" }}
            tickLine={false}
            axisLine={false}
            width={32}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--border-strong)", strokeWidth: 1 }} />
          {LINES.map(({ key, color, label }) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              name={label}
              stroke={color}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
              strokeWidth={2}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
