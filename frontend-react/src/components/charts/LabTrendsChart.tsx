import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from "recharts";
import type { LabHistoryPoint } from "../../types/api";
import { EmptyPane } from "../ui/Spinner";

const LINES = [
  { key: "wbc",        color: "#3b82f6", label: "WBC" },
  { key: "hemoglobin", color: "#10b981", label: "Hgb" },
  { key: "platelets",  color: "#f59e0b", label: "Plt ÷10", scale: 0.1 },
] as const;

interface Props {
  data: LabHistoryPoint[];
}

export function LabTrendsChart({ data }: Props) {
  if (!data.length) return <EmptyPane label="No lab history" />;

  const chartData = data.map((d) => ({
    date: d.date?.slice(0, 10) ?? "",
    wbc: d.wbc,
    hemoglobin: d.hemoglobin,
    platelets: d.platelets != null ? d.platelets * 0.1 : null,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: "var(--text-faint)" }} tickLine={false} />
        <YAxis tick={{ fontSize: 11, fill: "var(--text-faint)" }} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }}
          labelStyle={{ color: "var(--text-dim)" }}
        />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
        {LINES.map(({ key, color, label }) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            name={label}
            stroke={color}
            dot={false}
            strokeWidth={2}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
